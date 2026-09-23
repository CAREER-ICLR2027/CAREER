from types import SimpleNamespace

import pytest
import torch
from PIL import Image

pytest.importorskip("transformers")

from career.models import Model, QwenModel, InternVLModel, LlavaModel, _Prepared
from career.types import CapacityError, OutputError


class Tokenizer:
    eos_token_id = 9
    pad_token_id = 0
    bos_token_id = 0

    def __call__(self, text, **kwargs):
        if isinstance(text, str) and text in ("A", "B", "C"):
            return {"input_ids": [{"A": 1, "B": 2, "C": 3}[text]]}
        return {"input_ids": torch.tensor([[7, 8, 7]]), "attention_mask": torch.ones(1, 3, dtype=torch.long)}

    def decode(self, tokens, **kwargs):
        values = tokens.tolist() if hasattr(tokens, "tolist") else tokens
        return "".join({1: "A", 2: "B", 3: "C", 4: '\n{"grounded":[],"missing":[]}', 9: ""}.get(int(i), "x") for i in values)


class GeneratingModel:
    device = torch.device("cpu")
    dtype = torch.float32

    def __init__(self):
        self.config = SimpleNamespace(max_position_embeddings=100)
        self.generation_config = SimpleNamespace(eos_token_id=9, pad_token_id=0, bos_token_id=0)
        self.calls = 0

    def generate(self, **kwargs):
        self.calls += 1
        assert kwargs["generation_config"].do_sample is False
        assert kwargs["generation_config"].repetition_penalty == 1
        assert kwargs["generation_config"].num_beams == 1
        tokens = torch.empty((1, 0), dtype=torch.long)
        for raw in ([-1., 1., 3., 2., 10., 0., 0., 0., 0., -1.], [-1., 1., 3., 2., 10., 0., 0., 0., 0., -1.]):
            scores = torch.tensor([raw])
            for processor in kwargs["logits_processor"]:
                scores = processor(tokens, scores)
            tokens = torch.cat((tokens, scores.argmax(-1, keepdim=True)), dim=1)
        return SimpleNamespace(sequences=tokens)


class PreparedModel(Model):
    def __init__(self, limit=100):
        super().__init__(GeneratingModel(), Tokenizer(), limit)
        self.preparations = []

    def _prepare(self, images, prompt):
        self.preparations.append((tuple(image.getpixel((0, 0)) for image in images), prompt))
        return _Prepared({"input_ids": torch.ones(1, 3 + 4 * len(images), dtype=torch.long)}, 3 + 4 * len(images), 4 * len(images))


def images():
    return [Image.new("RGB", (8, 8), color) for color in ("red", "green", "blue")]


def test_decision_token_is_constrained_then_json_continues_in_same_call():
    model = PreparedModel()
    result = model.generate(images(), "all view metadata", codes=("A", "B", "C"), max_new_tokens=8)
    assert result.text == 'B\n{"grounded":[],"missing":[]}'
    assert result.first_token_logits == (1., 3., 2.)
    assert (result.token_count, result.input_tokens, result.visual_tokens) == (2, 15, 12)
    assert model.model.calls == 1


def test_capacity_never_discards_views_and_shares_one_preparation():
    model = PreparedModel(limit=20)
    views = images()
    assert not model.fits(views, "metadata", 6)
    with pytest.raises(CapacityError):
        model.generate(views, "metadata", max_new_tokens=6)
    assert model.model.calls == 0
    assert model.fits(views, "metadata", 5)
    model.generate(views, "metadata", max_new_tokens=5)
    assert len(model.preparations) == 1
    assert model.preparations[0][0] == ((255, 0, 0), (0, 128, 0), (0, 0, 255))
    views[1].putpixel((0, 0), (1, 2, 3))
    model.fits(views, "metadata", 5)
    assert len(model.preparations) == 2
    model.fits(views, "new metadata", 5)
    model.fits(views, "metadata", 5)
    assert len(model.preparations) == 4


@pytest.mark.parametrize("codes", [("A", "A"), ("AA", "B"), (), ("A", 2)])
def test_invalid_decision_codes_fail_before_generation(codes):
    model = PreparedModel()
    with pytest.raises(OutputError):
        model.generate(images(), "metadata", codes=codes, max_new_tokens=4)
    assert model.model.calls == 0


class QwenProcessor:
    def __init__(self):
        self.tokenizer = Tokenizer()
        self.image_processor = SimpleNamespace(merge_size=2)
        self.received = []

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        return "serialized"

    def __call__(self, *, images=None, **kwargs):
        assert kwargs.get("truncation") is False
        self.received.append(list(images or ()))
        count = len(images or ())
        return {
            "input_ids": torch.tensor([[7] + [6] * (4 * count) + [8]]),
            "attention_mask": torch.ones(1, 2 + 4 * count, dtype=torch.long),
            "image_grid_thw": torch.tensor([[1, 4, 4]] * count),
        }


def test_qwen_passes_all_images_and_counts_real_processed_tokens():
    core = GeneratingModel()
    core.config.image_token_id = 6
    processor = QwenProcessor()
    model = QwenModel(core, processor, 20)
    views = images()
    assert model.fits(views, "metadata", 6)
    result = model.generate(views, "metadata", codes=("A", "B", "C"), max_new_tokens=6)
    assert processor.received == [views]
    assert result.input_tokens == 14 and result.visual_tokens == 12
    assert len([item for item in processor.messages[0]["content"] if item["type"] == "image"]) == 3


def test_nonfinite_first_decision_logit_is_an_output_error():
    model = PreparedModel()

    def invalid(**kwargs):
        for processor in kwargs["logits_processor"]:
            processor(torch.zeros(1, 0, dtype=torch.long), torch.tensor([[0., float("nan"), 0., 0., 0.]]))
        raise AssertionError("nonfinite logits accepted")

    model.model.generate = invalid
    with pytest.raises(OutputError):
        model.generate(images(), "metadata", codes=("A", "B", "C"), max_new_tokens=4)


def test_declared_context_capacity_cannot_be_overridden_upwards():
    model = PreparedModel(limit=500)
    assert model.context_limit == 100


def test_llava_checks_untruncated_embeddings_and_preserves_every_patch(monkeypatch):
    core = GeneratingModel()
    core.config.tokenizer_model_max_length = 4
    core.config.max_position_embeddings = 100
    observed = {}

    def multimodal(input_ids, positions, attention, past, labels, tensor_images, **kwargs):
        assert core.config.tokenizer_model_max_length is None
        observed["images"] = tensor_images
        observed["sizes"] = kwargs["image_sizes"]
        return None, positions, torch.ones(1, 25, dtype=torch.long), None, torch.zeros(1, 25, 4), None

    core.prepare_inputs_labels_for_multimodal = multimodal
    model = LlavaModel(core, Tokenizer(), object(), 24, "qwen_1_5")
    monkeypatch.setattr(model, "_image_inputs", lambda views: [torch.zeros(2, 3, 8, 8) for _ in views])
    monkeypatch.setattr(model, "_prompt_ids", lambda prompt, count: torch.tensor([[7, -200, -200, -200, 8]]))
    assert not model.fits(images(), "metadata", 1)
    assert core.config.tokenizer_model_max_length == 4
    assert len(observed["images"]) == 3
    assert all(item.shape[0] == 2 for item in observed["images"])
    assert observed["sizes"] == [(8, 8)] * 3


def test_internvl_expands_each_images_actual_patch_count(monkeypatch):
    pytest.importorskip("torchvision")
    class InternTokenizer(Tokenizer):
        def convert_tokens_to_ids(self, token):
            return 6 if token == "<IMG_CONTEXT>" else 9

        def __call__(self, text, **kwargs):
            assert kwargs.get("truncation") is False
            count = text.count("<IMG_CONTEXT>")
            self.text = text
            return {"input_ids": torch.tensor([[7] + [6] * count + [8]])}

    core = GeneratingModel()
    core.config.force_image_size = 8
    core.config.max_dynamic_patch = 2
    core.config.use_thumbnail = True
    core.num_image_token = 2
    tokenizer = InternTokenizer()
    model = InternVLModel(core, tokenizer, 100)
    monkeypatch.setattr(model, "_conversation", lambda text: (text, "</s>"))
    views = [Image.new("RGB", (16, 8), "red"), Image.new("RGB", (8, 8), "blue")]
    assert model.fits(views, "ordered metadata", 1)
    prepared = model._cache_value
    # Wide image: two patches + thumbnail; square image: one patch.
    assert prepared.inputs["pixel_values"].shape == (4, 3, 8, 8)
    assert prepared.visual_tokens == 8 and prepared.input_tokens == 10
    assert tokenizer.text == "<img>" + "<IMG_CONTEXT>" * 6 + "</img>\n<img>" + "<IMG_CONTEXT>" * 2 + "</img>\nordered metadata"


def test_real_transformers_generation_returns_raw_logits_despite_checkpoint_defaults():
    transformers = pytest.importorskip("transformers")
    core = transformers.Qwen2ForCausalLM(transformers.Qwen2Config(
        vocab_size=10, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64,
        eos_token_id=9, pad_token_id=0, bos_token_id=0,
    )).eval()
    core.generation_config.repetition_penalty = 5
    core.generation_config.forced_bos_token_id = 7
    core.generation_config.transformers_version = "4.57.0"
    model = PreparedModel()
    model.model = core
    ids = torch.ones(1, 3, dtype=torch.long)
    with torch.inference_mode():
        expected = core(input_ids=ids).logits[0, -1, [1, 2, 3]].tolist()
    result = model.generate([], "metadata", codes=("A", "B", "C"), max_new_tokens=3)
    assert result.first_token_logits == pytest.approx(expected, abs=1e-6)
    assert result.text[0] == ("A", "B", "C")[max(range(3), key=expected.__getitem__)]
    assert 1 <= result.token_count <= 3


def test_external_llava_generates_from_prepared_embeddings():
    from transformers import modeling_utils, pytorch_utils
    for name in ("apply_chunking_to_forward", "find_pruneable_heads_and_indices", "prune_linear_layer"):
        if not hasattr(modeling_utils, name):
            setattr(modeling_utils, name, getattr(pytorch_utils, name))
    pytest.importorskip("llava")
    from llava.model.language_model.llava_qwen import LlavaQwenConfig, LlavaQwenForCausalLM

    core = LlavaQwenForCausalLM(LlavaQwenConfig(
        vocab_size=10, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64,
        eos_token_id=9, pad_token_id=0, bos_token_id=0,
    )).eval()
    model = LlavaModel(core, Tokenizer(), None, 64, "qwen_1_5")
    ids = torch.tensor([[7, 8, 7]])
    model._prepare = lambda images, prompt: _Prepared({
        "inputs_embeds": core.get_input_embeddings()(ids), "attention_mask": torch.ones_like(ids),
    }, 3, 0)
    with torch.inference_mode():
        expected = core(input_ids=ids).logits[0, -1, [1, 2, 3]].tolist()
    result = model.generate([], "metadata", codes=("A", "B", "C"), max_new_tokens=3)
    assert result.first_token_logits == pytest.approx(expected, abs=1e-6)
    assert 1 <= result.token_count <= 3


def test_real_qwen3_generates_with_three_images():
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    config = Qwen3VLConfig(
        text_config=dict(
            vocab_size=16, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=2, head_dim=8,
            max_position_embeddings=64, eos_token_id=9, pad_token_id=0, bos_token_id=0,
            rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 2]},
        ),
        vision_config=dict(
            depth=1, hidden_size=16, intermediate_size=32, num_heads=2, patch_size=2,
            spatial_merge_size=2, temporal_patch_size=2, out_hidden_size=16,
            num_position_embeddings=16, deepstack_visual_indexes=[],
        ),
        image_token_id=6, video_token_id=10, vision_start_token_id=4, vision_end_token_id=5,
    )

    class Processor(QwenProcessor):
        def __call__(self, *, images=None, **kwargs):
            ids = torch.tensor([[7] + [4, 6, 5] * len(images) + [8]])
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids),
                    "image_grid_thw": torch.tensor([[1, 2, 2]] * len(images)),
                    "pixel_values": torch.zeros(4 * len(images), 24)}

    model = QwenModel(Qwen3VLForConditionalGeneration(config).eval(), Processor(), 64)
    result = model.generate(images(), "metadata", codes=("A", "B", "C"), max_new_tokens=3)
    assert result.first_token_logits is not None
    assert result.input_tokens == 11 and result.visual_tokens == 3
    assert 1 <= result.token_count <= 3
