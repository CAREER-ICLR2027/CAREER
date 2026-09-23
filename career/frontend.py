"""Paper §3.6 / B.3: cached SAM localization, fixed SGAP atlas and ranking."""

from dataclasses import replace
import hashlib
import math

import numpy as np
from PIL import Image

from .types import Candidate, Localization, clip_box, enclose


def image_key(image):
    return hashlib.sha256(str((image.mode, image.size)).encode() + image.tobytes()).hexdigest()


class SAM:
    """Lazy SAM 3 loader with exact image/prompt caching, including empty detections."""

    def __init__(self, checkpoint, device):
        self.checkpoint, self.device = checkpoint, device
        self.processor = None
        self.states = {}
        self.results = {}
        self.calls = {"image_encodings": 0, "localizations": 0}

    def _state(self, image):
        if self.processor is None:
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
            model = build_sam3_image_model(checkpoint_path=str(self.checkpoint), device=self.device)
            model.eval().requires_grad_(False)
            self.processor = Sam3Processor(model, device=self.device)
        key = image_key(image)
        if key not in self.states:
            self.states[key] = self.processor.set_image(image)
            self.calls["image_encodings"] += 1
        return key, self.states[key]

    def encode(self, image):
        _, state = self._state(image)
        features = state["backbone_out"]["vision_features"]
        if features.ndim == 4:
            features = features[0]
        return features.detach().float().cpu().numpy()

    def localize(self, image, phrase):
        if not isinstance(phrase, str) or not phrase.strip():
            raise ValueError("localization phrase must be nonempty")
        key, state = self._state(image)
        cache_key = (key, phrase)
        if cache_key not in self.results:
            output = self.processor.set_text_prompt(state=state, prompt=phrase)
            self.calls["localizations"] += 1
            # Keep masks and original scores as well as the geometry used by the controller.
            self.results[cache_key] = {
                name: output[name].detach().cpu().clone()
                for name in ("boxes", "scores", "masks") if name in output
            }
        result = self.results[cache_key]
        boxes, scores = [], []
        for raw, score in zip(result["boxes"].tolist(), result["scores"].tolist()):
            if len(raw) != 4 or not all(math.isfinite(v) for v in (*raw, score)):
                continue
            x1, y1, x2, y2 = raw
            if x2 <= x1 or y2 <= y1:
                continue
            left, top = math.floor(x1), math.floor(y1)
            box = clip_box((left, top, math.ceil(x2) - left, math.ceil(y2) - top), image.size)
            if box is not None:
                boxes.append(box)
                scores.append(float(score))
        return Localization(tuple(boxes), tuple(scores))


class CLIP:
    def __init__(self, checkpoint, device):
        self.checkpoint, self.device = checkpoint, device
        self.model = self.processor = None

    def relevance(self, images, texts):
        import torch
        if self.model is None:
            from transformers import CLIPModel, CLIPProcessor
            self.processor = CLIPProcessor.from_pretrained(self.checkpoint, local_files_only=True)
            self.model = CLIPModel.from_pretrained(self.checkpoint, local_files_only=True).to(self.device)
            self.model.eval().requires_grad_(False)
        with torch.inference_mode():
            batch = self.processor(text=texts, images=images, return_tensors="pt", padding=True).to(self.device)
            outputs = self.model(**batch)
            images = outputs.image_embeds.float()
            texts = outputs.text_embeds.float()
            images = images / (images.norm(dim=-1, keepdim=True) + 1e-8)
            texts = texts / (texts.norm(dim=-1, keepdim=True) + 1e-8)
            query = texts.mean(dim=0)
            query = query / (query.norm() + 1e-8)
            return (images @ query).cpu().numpy()


def minmax(values):
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("ranking components must be finite")
    low, high = values.min(), values.max()
    return (values - low) / (high - low) if high > low else np.zeros_like(values)


def feature_dispersion(features, box, image_size):
    """Average cosine distance to the regional mean at receptive-field centers."""
    channels, height, width = features.shape
    x, y, w, h = box
    xs = (np.arange(width) + 0.5) * image_size[0] / width
    ys = (np.arange(height) + 0.5) * image_size[1] / height
    selected = features[:, (ys >= y) & (ys < y + h), :][:, :, (xs >= x) & (xs < x + w)]
    vectors = selected.reshape(channels, -1).T.astype(np.float64)
    if not len(vectors):
        return None
    mean = vectors.mean(axis=0)
    similarity = (vectors @ mean) / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(mean) + 1e-8)
    return max(0.0, 1.0 - float(similarity.mean()))


def edge_density(crop, size, interpolation):
    """Equations B.1–B.3; centered zero padding, one 3×3 mask erosion."""
    height, width = size
    ratio = min(width / crop.width, height / crop.height)
    new_width = max(1, min(width, round(crop.width * ratio)))
    new_height = max(1, min(height, round(crop.height * ratio)))
    if min(new_width, new_height) < 3:
        return None
    mode = {"bilinear": Image.Resampling.BILINEAR, "bicubic": Image.Resampling.BICUBIC,
            "lanczos": Image.Resampling.LANCZOS}[interpolation]
    gray = Image.fromarray(np.asarray(crop.convert("L"), dtype=np.float32) / 255)
    pixels = np.asarray(gray.resize((new_width, new_height), mode), dtype=np.float64)
    # Eroded valid pixels never touch padding, so computing their Sobel values
    # directly on the resized crop is exactly the masked padded calculation.
    gx = (-pixels[:-2, :-2] + pixels[:-2, 2:] - 2 * pixels[1:-1, :-2]
          + 2 * pixels[1:-1, 2:] - pixels[2:, :-2] + pixels[2:, 2:])
    gy = (-pixels[:-2, :-2] - 2 * pixels[:-2, 1:-1] - pixels[:-2, 2:]
          + pixels[2:, :-2] + 2 * pixels[2:, 1:-1] + pixels[2:, 2:])
    return float(np.hypot(gx, gy).mean())


def merge_candidates(tree, sam):
    """Exact geometry deduplication; SAM never creates parent-child edges."""
    merged, geometry, aliases = {}, {}, {}
    for candidate in [*tree, *sam]:
        if candidate.box in geometry:
            identifier = geometry[candidate.box]
            record = merged[identifier]
            record.sources = tuple(dict.fromkeys((*record.sources, *candidate.sources)))
            record.visited |= candidate.visited
            record.children = tuple(dict.fromkeys((*record.children, *candidate.children)))
            if record.sam_prompt is None and candidate.sam_prompt is not None:
                record.sam_prompt, record.localization = candidate.sam_prompt, candidate.localization
        else:
            identifier = candidate.id
            merged[identifier] = replace(candidate)
            geometry[candidate.box] = identifier
        aliases[candidate.id] = identifier
    for candidate in merged.values():
        candidate.children = tuple(dict.fromkeys(
            aliases[child] for child in candidate.children
            if child in aliases and aliases[child] != candidate.id
        ))
    return list(merged.values())


class CandidateFrontend:
    def __init__(self, sam, clip, config):
        self.sam, self.clip, self.config = sam, clip, config
        self.features = None
        self.minimum_size = 1
        self.calls = {"hierarchies": 0, "ranking_sets": 0}

    def localize(self, image, phrase):
        return self.sam.localize(image, phrase)

    def _rank(self, image, candidates, question, phrases):
        feasible, dispersions, edges, crops = [], [], [], []
        for candidate in candidates:
            x, y, width, height = candidate.box
            crop = image.crop((x, y, x + width, y + height))
            dispersion = feature_dispersion(self.features, candidate.box, image.size)
            edge = edge_density(crop, self.config.edge_size, self.config.edge_interpolation)
            if dispersion is None or edge is None:
                continue
            feasible.append(candidate)
            dispersions.append(dispersion)
            edges.append(edge)
            crops.append(crop)
        if not feasible:
            return []
        relevance = self.clip.relevance(crops, [question, *phrases])
        scores = 0.70 * minmax(relevance) + 0.30 * (0.50 * minmax(dispersions) + 0.50 * minmax(edges))
        ids = {candidate.id for candidate in feasible}
        for candidate, score in zip(feasible, scores):
            candidate.score = float(score)
            candidate.children = tuple(child for child in candidate.children if child in ids)
        self.calls["ranking_sets"] += 1
        return sorted(feasible, key=lambda candidate: (-candidate.score, candidate.id))

    def initial_candidates(self, image, phrases, question):
        self.features = self.sam.encode(image)
        candidates = []
        for index, phrase in enumerate(phrases):
            detections = self.localize(image, phrase)
            if detections.boxes:
                candidates.append(Candidate(f"sam{index:04d}", enclose(detections.boxes),
                                            ("SAM:" + phrase,), sam_prompt=phrase,
                                            localization=detections))
        candidates = merge_candidates([], candidates)
        root_box = (0, 0, *image.size)
        for candidate in candidates:
            candidate.visited = candidate.box == root_box
        return self._rank(image, candidates, question, phrases)

    def build_candidates(self, image, sam_candidates, question, phrases):
        from .sgap import ConstrainedTreeBuilder
        settings = self.config.sgap
        build_keys = {"max_depth", "min_splits", "max_splits", "min_region_size", "max_nodes"}
        builder = ConstrainedTreeBuilder(self.features, **{k: v for k, v in settings.items() if k not in build_keys})
        root = builder.build_tree(**{k: settings[k] for k in ("max_depth", "min_splits", "max_splits")})
        self.calls["hierarchies"] += 1
        height, width = self.features.shape[1:]
        candidates = []

        def visit(node):
            if len(candidates) >= settings["max_nodes"]:
                return None
            y1, x1, y2, x2 = node["bbox"]
            left, top = math.floor(x1 * image.width / width), math.floor(y1 * image.height / height)
            right, bottom = math.ceil(x2 * image.width / width), math.ceil(y2 * image.height / height)
            box = clip_box((left, top, right - left, bottom - top), image.size)
            if box is None or min(box[2:]) < settings["min_region_size"]:
                return None
            candidate = Candidate("sgap" + node["node_id"], box, ("SGAP",),
                                  visited=box == (0, 0, *image.size))
            candidates.append(candidate)
            candidate.children = tuple(child for child in (visit(n) for n in node["children"]) if child is not None)
            return candidate.id

        visit(root)
        # Include screened SAM geometry for inherited visit history, but keep only
        # unvisited SAM-only records in the formal candidate pool.
        merged = merge_candidates(candidates, sam_candidates)
        history = [c for c in merged if "SGAP" not in c.sources and c.visited]
        active = [c for c in merged if "SGAP" in c.sources or not c.visited]
        return self._rank(image, active, question, phrases) + history
