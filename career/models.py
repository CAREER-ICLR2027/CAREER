"""Ordered multi-image inference and first-token decisions for CAREER.

Each backend measures the actual expanded input. No backend truncates evidence
to make it fit. Checkpoints and optional model libraries are loaded on demand.
"""

from dataclasses import dataclass
from hashlib import sha256
from importlib import import_module
import json
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image

from .types import CapacityError, Completion, OutputError


@dataclass
class _Prepared:
    inputs: dict
    input_tokens: int
    visual_tokens: int


class _FirstToken:
    def __init__(self, ids):
        self.ids = ids
        self.steps = 0
        self.logits = None

    def __call__(self, input_ids, scores):
        self.steps += 1
        if self.steps == 1 and self.ids is not None:
            values = scores[0, self.ids]
            if not torch.isfinite(values).all():
                raise OutputError("nonfinite decision logits")
            self.logits = tuple(values.float().tolist())
            constrained = torch.full_like(scores, -torch.inf)
            constrained[:, self.ids] = scores[:, self.ids]
            return constrained
        return scores


class Model:
    """One sample per call, with a one-input preprocessing cache."""

    def __init__(self, model, tokenizer, context_limit: int):
        if isinstance(context_limit, bool) or not isinstance(context_limit, int) or context_limit <= 0:
            raise ValueError("context_limit must be a positive integer")
        self.model, self.tokenizer = model, tokenizer
        freeze = getattr(model, "requires_grad_", None)
        if freeze is not None:
            freeze(False)
        self.device = model.device
        self.dtype = model.dtype
        capacities = [context_limit]
        config = model.config
        for candidate in (config, getattr(config, "text_config", None), getattr(config, "llm_config", None)):
            value = getattr(candidate, "max_position_embeddings", None)
            if isinstance(value, int) and value > 0:
                capacities.append(value)
        self.context_limit = min(capacities)
        self._cache_key = self._cache_value = None

    def _prepare(self, images, prompt) -> _Prepared:
        raise NotImplementedError

    def _prepared(self, images, prompt, max_new_tokens):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be nonempty text")
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be a positive integer")
        if any(not isinstance(image, Image.Image) for image in images):
            raise TypeError("images must contain PIL images")
        key = (prompt, tuple((image.mode, image.size, sha256(image.tobytes()).digest()) for image in images))
        if key != self._cache_key:
            # Drop the old GPU tensors before allocating another prepared input.
            self._cache_key = self._cache_value = None
            with torch.inference_mode():
                prepared = self._prepare(images, prompt)
            self._cache_value, self._cache_key = prepared, key
        return self._cache_value

    def fits(self, images: Sequence[Image.Image], prompt: str, max_new_tokens: int) -> bool:
        prepared = self._prepared(images, prompt, max_new_tokens)
        return prepared.input_tokens + max_new_tokens <= self.context_limit

    def _code_ids(self, codes):
        if codes is None:
            return None
        if not codes or any(not isinstance(code, str) or not code for code in codes):
            raise OutputError("decision codes must be nonempty strings")
        if len(set(codes)) != len(codes):
            raise OutputError("decision codes must be distinct")
        ids = []
        for code in codes:
            tokens = self.tokenizer(code, add_special_tokens=False)["input_ids"]
            if hasattr(tokens, "tolist"):
                tokens = tokens.tolist()
            if len(tokens) != 1 or not isinstance(tokens[0], int):
                raise OutputError(f"decision code {code!r} must tokenize as one token")
            if self.tokenizer.decode(tokens, skip_special_tokens=False) != code:
                raise OutputError(f"decision code {code!r} does not round-trip through the tokenizer")
            ids.append(tokens[0])
        if len(set(ids)) != len(ids):
            raise OutputError("decision codes must map to distinct tokens")
        return ids

    def _run_generation(self, prepared, **kwargs):
        return self.model.generate(**prepared.inputs, **kwargs)

    def generate(self, images: Sequence[Image.Image], prompt: str, *,
                 codes: Sequence[str] | None = None, max_new_tokens: int) -> Completion:
        from transformers import GenerationConfig, LogitsProcessorList

        ids = self._code_ids(codes)
        prepared = self._prepared(images, prompt, max_new_tokens)
        if prepared.input_tokens + max_new_tokens > self.context_limit:
            raise CapacityError(
                f"complete evidence requires {prepared.input_tokens} input + "
                f"{max_new_tokens} output tokens; context limit is {self.context_limit}"
            )
        # A fresh config prevents checkpoint repetition penalties/forced tokens
        # from altering the raw first-token logits before our only processor.
        original = getattr(self.model, "generation_config", None)
        special = {name: getattr(original, name, None) for name in ("eos_token_id", "pad_token_id", "bos_token_id")}
        for name, value in special.items():
            if value is None:
                special[name] = getattr(self.tokenizer, name, None)
        generation = GenerationConfig(
            **special, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
            repetition_penalty=1.0, use_cache=True, return_dict_in_generate=True,
        )
        first = _FirstToken(ids)
        with torch.inference_mode():
            output = self._run_generation(
                prepared, generation_config=generation,
                logits_processor=LogitsProcessorList([first]),
                use_model_defaults=False,
            )
        if first.steps < 1 or not hasattr(output, "sequences"):
            raise OutputError("model did not generate any tokens")
        tokens = output.sequences[0, -first.steps:]
        if ids is not None and (first.logits is None or int(tokens[0]) not in ids):
            raise OutputError("model did not honor the first-token decision constraint")
        text = self.tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return Completion(text, first.logits, first.steps, prepared.input_tokens, prepared.visual_tokens)


class QwenModel(Model):
    def __init__(self, model, processor, context_limit):
        super().__init__(model, processor.tokenizer, context_limit)
        self.processor = processor

    def _prepare(self, images, prompt):
        content = [{"type": "image"} for _ in images] + [{"type": "text", "text": prompt}]
        text = self.processor.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True,
        )
        inputs = dict(self.processor(
            text=[text], images=list(images) if images else None,
            padding=True, truncation=False, return_tensors="pt",
        ))
        grid = inputs.get("image_grid_thw")
        if len(images) != (len(grid) if grid is not None else 0):
            raise OutputError("processor did not preserve every image")
        merge = self.processor.image_processor.merge_size
        visual = int((grid.prod(-1) // (merge * merge)).sum()) if images else 0
        image_id = getattr(self.model.config, "image_token_id", None)
        if image_id is not None and int((inputs["input_ids"] == image_id).sum()) != visual:
            raise OutputError("processed image tokens do not match all image grids")
        inputs = {name: value.to(self.device) if torch.is_tensor(value) else value for name, value in inputs.items()}
        return _Prepared(inputs, inputs["input_ids"].shape[1], visual)


def _internvl_tiles(image, size, min_tiles, max_tiles, thumbnail):
    """InternVL's aspect-ratio grid, including its optional global thumbnail."""
    ratios = sorted({(i, j) for i in range(1, max_tiles + 1) for j in range(1, max_tiles + 1)
                     if min_tiles <= i * j <= max_tiles}, key=lambda pair: (pair[0] * pair[1], pair))
    best, difference = (1, 1), float("inf")
    aspect, area = image.width / image.height, image.width * image.height
    for ratio in ratios:
        current = abs(aspect - ratio[0] / ratio[1])
        if current < difference or (current == difference and area > 0.5 * size * size * ratio[0] * ratio[1]):
            best, difference = ratio, current
    image = image.convert("RGB")
    resized = image.resize((size * best[0], size * best[1]))
    tiles = [resized.crop((x * size, y * size, (x + 1) * size, (y + 1) * size))
             for y in range(best[1]) for x in range(best[0])]
    if thumbnail and len(tiles) > 1:
        tiles.append(image.resize((size, size)))
    return tiles


class InternVLModel(Model):
    def __init__(self, model, tokenizer, context_limit):
        super().__init__(model, tokenizer, context_limit)
        self.model.img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")

    def _conversation(self, prompt):
        module = import_module(type(self.model).__module__)
        conversation = module.get_conv_template(self.model.template)
        conversation.system_message = self.model.system_message
        conversation.append_message(conversation.roles[0], prompt)
        conversation.append_message(conversation.roles[1], None)
        return conversation.get_prompt(), conversation.sep.strip()

    def _prepare(self, images, prompt):
        from torchvision import transforms
        from torchvision.transforms import InterpolationMode

        config = self.model.config
        size = getattr(config, "force_image_size", None) or config.vision_config.image_size
        dynamic = getattr(config, "dynamic_image_size", True)
        min_tiles = getattr(config, "min_dynamic_patch", 1) if dynamic else 1
        max_tiles = getattr(config, "max_dynamic_patch", 12) if dynamic else 1
        transform = transforms.Compose([
            transforms.Resize((size, size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        counts, pixels = [], []
        for image in images:
            tiles = _internvl_tiles(image, size, min_tiles, max_tiles, getattr(config, "use_thumbnail", True))
            counts.append(len(tiles))
            pixels.extend(transform(tile) for tile in tiles)
        text, separator = self._conversation("<image>\n" * len(images) + prompt)
        for count in counts:
            text = text.replace("<image>", "<img>" + "<IMG_CONTEXT>" * (self.model.num_image_token * count) + "</img>", 1)
        if "<image>" in text:
            raise OutputError("prompt contains an unmatched image placeholder")
        inputs = dict(self.tokenizer(text, return_tensors="pt", padding=True, truncation=False))
        visual = sum(counts) * self.model.num_image_token
        if int((inputs["input_ids"] == self.model.img_context_token_id).sum()) != visual:
            raise OutputError("InternVL tokenizer did not preserve every image patch")
        inputs = {name: value.to(self.device) if torch.is_tensor(value) else value for name, value in inputs.items()}
        inputs["pixel_values"] = torch.stack(pixels).to(device=self.device, dtype=self.dtype) if pixels else None
        inputs["eos_token_id"] = self.tokenizer.convert_tokens_to_ids(separator)
        return _Prepared(inputs, inputs["input_ids"].shape[1], visual)


class LlavaModel(Model):
    def __init__(self, model, tokenizer, image_processor, context_limit, conversation):
        super().__init__(model, tokenizer, context_limit)
        self.image_processor, self.conversation = image_processor, conversation

    def _prompt_ids(self, prompt, count):
        from llava.conversation import conv_templates
        from llava.mm_utils import tokenizer_image_token

        conversation = conv_templates[self.conversation].copy()
        conversation.append_message(conversation.roles[0], "<image>\n" * count + prompt)
        conversation.append_message(conversation.roles[1], None)
        return tokenizer_image_token(conversation.get_prompt(), self.tokenizer, -200, return_tensors="pt").unsqueeze(0).to(self.device)

    def _image_inputs(self, images):
        from llava.mm_utils import process_images

        tensors = process_images(list(images), self.image_processor, self.model.config)
        if len(tensors) != len(images):
            raise OutputError("LLaVA processor did not preserve every image")
        return [tensor.to(device=self.device, dtype=self.dtype) for tensor in tensors]

    def _prepare(self, images, prompt):
        ids = self._prompt_ids(prompt, len(images))
        attention = torch.ones_like(ids)
        if int((ids == -200).sum()) != len(images):
            raise OutputError("LLaVA prompt does not have one placeholder per image")
        if not images:
            return _Prepared({"input_ids": ids, "attention_mask": attention}, ids.shape[1], 0)
        config = self.model.config
        previous = getattr(config, "tokenizer_model_max_length", None)
        try:
            # The upstream preparation otherwise silently truncates embeddings.
            config.tokenizer_model_max_length = None
            _, positions, attention, _, embeddings, _ = self.model.prepare_inputs_labels_for_multimodal(
                ids, None, attention, None, None, self._image_inputs(images),
                modalities=["image"], image_sizes=[image.size for image in images],
            )
        finally:
            config.tokenizer_model_max_length = previous
        length = embeddings.shape[1]
        visual = length - (ids.shape[1] - len(images))
        if visual <= 0:
            raise OutputError("LLaVA returned no visual embeddings")
        inputs = {"inputs_embeds": embeddings, "attention_mask": attention}
        if positions is not None:
            inputs["position_ids"] = positions
        return _Prepared(inputs, length, visual)

    def _run_generation(self, prepared, **kwargs):
        from transformers import GenerationMixin

        # Upstream's wrapper refuses prepared inputs_embeds; its inherited
        # generation path accepts them without a second vision encoder call.
        return GenerationMixin.generate(self.model, **prepared.inputs, **kwargs)


def _enable_internlm_generation(model):
    """Adapt the checkpoint's legacy tuple cache to Transformers 4.57."""
    from transformers import GenerationConfig, GenerationMixin

    language = model.language_model
    if "internlm2" not in type(language).__name__.lower():
        return
    bases = (type(language),) if isinstance(language, GenerationMixin) else (type(language), GenerationMixin)
    language.__class__ = type("InternLM2Generation", bases, {
        "_supports_default_dynamic_cache": classmethod(lambda cls: False),
    })
    if getattr(language, "generation_config", None) is None:
        language.generation_config = GenerationConfig.from_model_config(language.config)


def load_model(checkpoint: str, device: str, context_limit: int) -> Model:
    """Load a local InternVL, LLaVA or Qwen VL checkpoint for inference."""
    from transformers import AutoModel, AutoProcessor, AutoTokenizer
    from transformers.utils import is_flash_attn_2_available

    checkpoint = str(Path(checkpoint).expanduser().resolve())
    config = json.loads((Path(checkpoint) / "config.json").read_text())
    kind = (config.get("model_type", "") + " " + " ".join(config.get("architectures", []))).lower()
    dtype = torch.float32 if torch.device(device).type == "cpu" else torch.bfloat16
    common = dict(local_files_only=True, torch_dtype=dtype, low_cpu_mem_usage=True)
    if "internvl" in kind:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True, use_fast=False, local_files_only=True)
        model = AutoModel.from_pretrained(
            checkpoint, trust_remote_code=True, device_map=device,
            use_flash_attn=is_flash_attn_2_available(), **common,
        ).eval()
        _enable_internlm_generation(model)
        return InternVLModel(model, tokenizer, context_limit)
    if "llava" in kind:
        # LLaVA's QFormer imports three helpers from their pre-4.57 location.
        from transformers import modeling_utils, pytorch_utils
        for name in ("apply_chunking_to_forward", "find_pruneable_heads_and_indices", "prune_linear_layer"):
            if not hasattr(modeling_utils, name):
                setattr(modeling_utils, name, getattr(pytorch_utils, name))
        from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM
        from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM

        cls = LlavaQwenForCausalLM if "qwen" in kind else LlavaLlamaForCausalLM
        model = cls.from_pretrained(checkpoint, attn_implementation="sdpa", **common).to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=False, local_files_only=True)
        model.config.tokenizer_padding_side = "left"
        tower = model.get_vision_tower()
        if not tower.is_loaded:
            tower.load_model(device_map=device)
        tower.to(device=device, dtype=dtype)
        return LlavaModel(model, tokenizer, tower.image_processor, context_limit, "qwen_1_5" if "qwen" in kind else "v1")
    from transformers import Qwen2_5_VLForConditionalGeneration, Qwen3VLForConditionalGeneration

    if "qwen3_vl" in kind or "qwen3vl" in kind:
        cls = Qwen3VLForConditionalGeneration
    elif "qwen2_5_vl" in kind or "qwen2_5vl" in kind:
        cls = Qwen2_5_VLForConditionalGeneration
    else:
        raise ValueError(f"unsupported model checkpoint type: {kind}")
    processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True)
    model = cls.from_pretrained(checkpoint, device_map=device, attn_implementation="sdpa", **common).eval()
    return QwenModel(model, processor, context_limit)
