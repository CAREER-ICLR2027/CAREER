"""Run CAREER on one image using the manuscript method."""

import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--options", nargs="+", required=True)
    parser.add_argument("--generator", type=Path, required=True)
    parser.add_argument("--verifier", type=Path, required=True, help="paper default: Qwen3-VL-4B-Instruct")
    parser.add_argument("--sam", type=Path, required=True)
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--runtime-config", type=Path, required=True,
                        help="fixed run settings whose values are omitted from the paper; see docs/appendix.md")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--verifier-device", default=None)
    args = parser.parse_args(argv)
    if not args.image.is_file():
        parser.error(f"image does not exist: {args.image}")
    if not args.question.strip() or not 2 <= len(args.options) <= 26:
        parser.error("provide a question and between 2 and 26 options")
    if any(not option.strip() for option in args.options):
        parser.error("option text must be nonempty")
    from .config import RuntimeConfig
    try:
        config = RuntimeConfig.load(args.runtime_config)
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))

    with redirect_stdout(sys.stderr):
        from PIL import Image
        import torch
        from .frontend import CandidateFrontend, CLIP, SAM
        from .models import load_model
        from .search import run_search

        generator = load_model(args.generator, device=args.device,
                               context_limit=config.generator_context_tokens)
        verifier = load_model(args.verifier, device=args.verifier_device or args.device,
                              context_limit=config.verifier_context_tokens)
        frontend = CandidateFrontend(SAM(args.sam, args.device), CLIP(args.clip, args.device), config)
        with Image.open(args.image) as source:
            image = source.convert("RGB")
        options = {chr(65 + index): option.strip() for index, option in enumerate(args.options)}
        with torch.inference_mode():
            result = run_search(image, args.question.strip(), options, generator=generator,
                                verifier=verifier, frontend=frontend, config=config.search)
    # Print only the answer summary.
    print(json.dumps({key: result[key] for key in ("answer", "option_id", "status", "reason", "observations")},
                     ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
