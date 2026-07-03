from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-len", type=int, default=2000)
    parser.add_argument("--device", default=os.environ.get("AI_DEVICE", "auto"))
    parser.add_argument("--flan-tokenizer", default="google/flan-t5-base")
    parser.add_argument("--model-file", default="pytorch_model.bin")
    parser.add_argument("--tokenizer-file", default="vocab_remi.pkl")
    parser.add_argument("--summary", default=None)
    args = parser.parse_args()

    repo_dir = _resolve_path(args.repo_dir)
    checkpoint_dir = _resolve_path(args.checkpoint_dir)
    output_path = _resolve_path(args.output)
    model_path = _resolve_checkpoint_file(checkpoint_dir, args.model_file)
    remi_tokenizer_path = _resolve_checkpoint_file(checkpoint_dir, args.tokenizer_file)
    _validate_paths(repo_dir, model_path, remi_tokenizer_path)

    sys.path.insert(0, str(repo_dir))

    try:
        import torch
        import torch.nn as nn
        from model.transformer_model import Transformer
        from transformers import T5Tokenizer
    except ImportError as exc:
        raise SystemExit(f"Missing Text2MIDI inference dependency: {exc}") from exc

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    device = _select_device(args.device, torch)
    with remi_tokenizer_path.open("rb") as file:
        r_tokenizer = pickle.load(file)

    vocab_size = len(r_tokenizer)
    model = Transformer(vocab_size, 768, 8, 2048, 18, 1024, False, 8, device=device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(_normalize_state_dict(state_dict))
    model.to(device)
    model.eval()

    tokenizer = T5Tokenizer.from_pretrained(args.flan_tokenizer)
    inputs = tokenizer(args.prompt, return_tensors="pt", padding=True, truncation=True)
    input_ids = nn.utils.rnn.pad_sequence(
        inputs.input_ids,
        batch_first=True,
        padding_value=0,
    ).to(device)
    attention_mask = nn.utils.rnn.pad_sequence(
        inputs.attention_mask,
        batch_first=True,
        padding_value=0,
    ).to(device)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            attention_mask,
            max_len=args.max_len,
            temperature=args.temperature,
        )

    output_list = output[0].tolist()
    generated_midi = r_tokenizer.decode(output_list)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generated_midi.dump_midi(str(output_path))

    report = {
        "status": "ok" if output_path.exists() and output_path.stat().st_size > 0 else "fail",
        "repo_dir": str(repo_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "output": str(output_path),
        "bytes": output_path.stat().st_size if output_path.exists() else 0,
        "device": str(device),
        "vocab_size": vocab_size,
        "max_len": args.max_len,
        "temperature": args.temperature,
        "flan_tokenizer": args.flan_tokenizer,
    }
    if args.summary:
        summary_path = _resolve_path(args.summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "ok":
        raise SystemExit(1)


def _resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (ROOT / path).resolve()


def _resolve_checkpoint_file(checkpoint_dir: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    if path.parent != Path("."):
        return (ROOT / path).resolve()
    return checkpoint_dir / path


def _validate_paths(repo_dir: Path, model_path: Path, tokenizer_path: Path) -> None:
    missing = [
        str(path)
        for path in (repo_dir, model_path, tokenizer_path)
        if not path.exists()
    ]
    if missing:
        raise SystemExit(f"Missing Text2MIDI path(s): {missing}")
    if not (repo_dir / "model" / "transformer_model.py").exists():
        raise SystemExit(f"Text2MIDI repo has no model/transformer_model.py: {repo_dir}")


def _select_device(value: str, torch_module: Any):
    normalized = (value or "auto").strip().lower()
    if normalized in {"", "auto"}:
        if torch_module.cuda.is_available():
            return torch_module.device("cuda")
        if hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
            return torch_module.device("mps")
        return torch_module.device("cpu")
    return torch_module.device(normalized)


def _normalize_state_dict(state_dict: Any) -> dict[str, Any]:
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if not isinstance(state_dict, dict):
        raise SystemExit("Text2MIDI checkpoint did not contain a state dict")
    if any(key.startswith("module.") for key in state_dict):
        return {
            key.removeprefix("module."): value
            for key, value in state_dict.items()
        }
    return state_dict


if __name__ == "__main__":
    main()
