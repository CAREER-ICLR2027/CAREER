from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from career.frontend import CandidateFrontend, CLIP, SAM, edge_density, feature_dispersion, merge_candidates
from career.types import Candidate, Localization


def test_ranking_uses_raw_components_from_paper_equation_five(monkeypatch):
    from career import frontend

    monkeypatch.setattr(frontend, "feature_dispersion", lambda features, box, size: box[0] / 10)
    monkeypatch.setattr(frontend, "edge_density", lambda crop, size, interpolation: 0.)
    clip = SimpleNamespace(relevance=lambda images, texts: np.array([.6, .5]))
    settings = SimpleNamespace(edge_size=(8, 8), edge_interpolation="bilinear")
    front = CandidateFrontend(None, clip, settings)
    image = Image.new("RGB", (30, 10))
    candidates = [Candidate("first", (0, 0, 10, 10), ("SAM",)),
                  Candidate("second", (10, 0, 10, 10), ("SAM",))]

    ranked = front._rank(image, candidates, "question", [])

    assert [candidate.id for candidate in ranked] == ["second", "first"]
    assert [candidate.score for candidate in ranked] == pytest.approx([.5, .42])


def test_feature_dispersion_selects_centers_and_average_cosine_distance():
    features = np.array([[[1, 0], [0, 0]], [[0, 1], [0, 0]]], dtype=float)
    value = feature_dispersion(features, (0, 0, 20, 10), (20, 20))
    assert value == pytest.approx(1 - 1 / np.sqrt(2), abs=1e-7)
    assert feature_dispersion(features, (0, 0, 3, 3), (20, 20)) is None


def test_edge_density_excludes_padding_boundary_and_rejects_empty_eroded_mask():
    assert edge_density(Image.new("RGB", (40, 10), "white"), (32, 32), "bilinear") == 0
    assert edge_density(Image.new("RGB", (1000, 1), "white"), (32, 32), "bilinear") is None
    horizontal_ramp = np.tile(np.arange(5, dtype=np.uint8) * 40, (5, 1))
    assert edge_density(Image.fromarray(horizontal_ramp), (5, 5), "bilinear") == pytest.approx(8 * 40 / 255)


def test_exact_geometry_merge_preserves_only_tree_edges_and_visit_history():
    tree = [Candidate("root", (0, 0, 100, 100), ("SGAP",), ("child",), visited=True),
            Candidate("child", (0, 0, 50, 50), ("SGAP",))]
    sam = [Candidate("same", (0, 0, 50, 50), ("SAM:object",), visited=True),
           Candidate("overlap", (0, 0, 51, 51), ("SAM:other",))]
    pool = {c.id: c for c in merge_candidates(tree, sam)}
    assert set(pool) == {"root", "child", "overlap"}
    assert pool["child"].visited
    assert pool["child"].sources == ("SGAP", "SAM:object")
    assert pool["root"].children == ("child",)
    assert pool["overlap"].children == ()


def test_sam_candidates_preserve_each_detected_instance():
    class Localizer:
        def encode(self, image):
            return np.ones((2, 20, 20))

        def localize(self, image, phrase):
            return Localization(((0, 0, 10, 10), (20, 10, 10, 10)), (.6, .95))

    clip = SimpleNamespace(relevance=lambda images, texts: np.ones(len(images)))
    config = SimpleNamespace(edge_size=(32, 32), edge_interpolation="bilinear")
    front = CandidateFrontend(Localizer(), clip, config)
    candidates = front.initial_candidates(Image.new("RGB", (100, 100)), ["luggage"], "Where is the luggage?")
    assert [candidate.box for candidate in candidates] == [(0, 0, 10, 10), (20, 10, 10, 10)]
    assert [candidate.localization.scores for candidate in candidates] == [(.6,), (.95,)]
    assert all(candidate.sam_prompt == "luggage" for candidate in candidates)


def test_clip_averages_normalized_text_embeddings_then_normalizes_the_mean():
    import torch

    class Batch(dict):
        def to(self, device):
            return self

    clip = CLIP("unused", "cpu")
    clip.processor = lambda **kwargs: Batch()
    clip.model = lambda **kwargs: SimpleNamespace(
        image_embeds=torch.tensor([[1., 0.], [0., 1.]]),
        text_embeds=torch.tensor([[3., 0.], [0., 5.]]),
    )
    scores = clip.relevance([Image.new("RGB", (4, 4))] * 2, ["question", "phrase"])
    assert scores.tolist() == pytest.approx([1 / np.sqrt(2)] * 2)


def test_sam_caches_image_and_exact_prompt_including_empty_result():
    import torch

    class Processor:
        def set_image(self, image):
            return {"backbone_out": {"vision_features": torch.ones((1, 3, 4, 4))}}

        def set_text_prompt(self, state, prompt):
            return {"boxes": torch.empty((0, 4)), "scores": torch.empty((0,)),
                    "masks": torch.empty((0, 10, 10))}

    sam = SAM("unused", "cpu")
    sam.processor = Processor()
    image = Image.new("RGB", (10, 10))
    assert sam.encode(image).shape == (3, 4, 4)
    assert sam.localize(image, "object") == Localization((), ())
    assert sam.localize(image.copy(), "object") == Localization((), ())
    assert sam.calls == {"image_encodings": 1, "localizations": 1}
    assert "masks" in next(iter(sam.results.values()))


def test_sgap_keeps_zero_distance_neighbors_and_handles_a_single_atom():
    from career.sgap import ConstrainedTreeBuilder

    tiny = ConstrainedTreeBuilder(np.ones((3, 1, 1)), n_atoms=1)
    assert tiny.build_tree(max_depth=1)["children"] == []
    uniform = ConstrainedTreeBuilder(np.ones((3, 4, 4)), n_atoms=16)
    assert uniform.adj_matrix.nnz > 0
    assert uniform.build_tree(max_depth=1)["children"] == []


def test_sgap_builder_defaults_match_manuscript_constants():
    from career.sgap import ConstrainedTreeBuilder

    builder = ConstrainedTreeBuilder(np.ones((3, 1, 1)))
    assert (builder.n_atoms, builder.pos_weight, builder.split_threshold,
            builder.keep_threshold, builder.use_local_normalization,
            builder.use_silhouette_score) == (600, 3.5, .30, .15, True, True)
