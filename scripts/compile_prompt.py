from __future__ import annotations

import argparse
from pathlib import Path

from arranger_core import compile_prompt


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile a text prompt into GenerationSpec JSON.")
    parser.add_argument("prompt", nargs="?", help="Prompt text to compile.")
    parser.add_argument("--prompt", dest="prompt_option", help="Prompt text to compile.")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic generation seed.")
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    args = parser.parse_args()

    prompt = args.prompt_option or args.prompt
    if prompt is None:
        parser.error("Provide a prompt as a positional argument or with --prompt.")

    spec = compile_prompt(prompt, seed=args.seed)
    json_text = spec.model_dump_json(indent=2)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json_text + "\n", encoding="utf-8")
    else:
        print(json_text)


if __name__ == "__main__":
    main()
