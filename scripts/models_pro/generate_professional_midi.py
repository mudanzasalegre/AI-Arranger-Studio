from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
for package in ("arranger_core", "dataset_tools", "model_backends", "training"):
    package_path = str(ROOT / "packages" / package)
    if package_path not in sys.path:
        sys.path.insert(0, package_path)

from arranger_core.professional import (  # noqa: E402
    ProfessionalGenerationOptions,
    ProfessionalGenerationOrchestrator,
)

DEFAULT_PROFILE = "hard_bop_minor_blues_sextet_pro"


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    profile_data = _load_profile(_repo_path(args.profile_config), args.profile)
    prompt = args.prompt or str(profile_data.get("prompt") or "")
    if not prompt:
        raise SystemExit("A prompt is required. Pass --prompt or use a profile with prompt.")

    options = ProfessionalGenerationOptions(
        prompt=prompt,
        profile=args.profile,
        seed=args.seed if args.seed is not None else int(profile_data.get("seed", 1234)),
        run_id=args.run_id,
        output_root=args.output_root,
        ai_config_path=args.ai_config,
        quality_thresholds_path=args.quality_thresholds,
        export_mode=args.export_mode,
        include_pdf=args.include_pdf,
        min_rating=str(profile_data.get("min_rating") or args.min_rating),  # type: ignore[arg-type]
        use_llm_planner=_flag(args.use_llm_planner, profile_data, "use_llm_planner", True),
        use_rule_based_base=_flag(
            args.use_rule_based_base,
            profile_data,
            "use_rule_based_base",
            True,
        ),
        use_custom_role_models=_flag(
            args.use_custom_role_models,
            profile_data,
            "use_custom_role_models",
            True,
        ),
        use_midigpt_infill=_flag(
            args.use_midigpt_infill,
            profile_data,
            "use_midigpt_infill",
            True,
        ),
        use_text2midi_sketch=_flag(
            args.use_text2midi_sketch,
            profile_data,
            "use_text2midi_sketch",
            False,
        ),
        midigpt_targets=[
            item for item in profile_data.get("midigpt_targets", []) if isinstance(item, dict)
        ],
        max_ai_attempts=args.max_ai_attempts,
        clean=not args.no_clean,
    )
    result = ProfessionalGenerationOrchestrator(repo_root=ROOT).generate(options)
    payload = result.model_dump(mode="json")
    print(json.dumps(payload, indent=2))
    if result.status != "ok":
        raise SystemExit(1)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a professional MIDI arrangement.")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--profile", default="pro")
    parser.add_argument("--profile-config", default="configs/generation_profiles.pro.yaml")
    parser.add_argument("--ai-config", default="configs/ai_models.pro.yaml")
    parser.add_argument("--quality-thresholds", default="configs/quality_thresholds.pro.yaml")
    parser.add_argument("--output-root", default="outputs/pro_benchmarks")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--export-mode", choices=["private", "commercial"], default="private")
    parser.add_argument("--min-rating", choices=["A", "B", "C", "D"], default="B")
    parser.add_argument("--max-ai-attempts", type=int, default=3)
    parser.add_argument("--include-pdf", action="store_true")
    parser.add_argument("--no-clean", action="store_true")
    parser.add_argument(
        "--use-llm-planner",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--use-rule-based-base",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--use-custom-role-models",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--use-midigpt-infill",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--use-text2midi-sketch",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    return parser.parse_args(argv)


def _load_profile(path: Path, requested: str) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    profiles = payload.get("profiles") if isinstance(payload, dict) else None
    if not isinstance(profiles, dict) or not profiles:
        return {}
    profile_id = _resolve_profile_id(profiles, requested)
    profile = profiles.get(profile_id, {})
    return dict(profile) if isinstance(profile, dict) else {}


def _resolve_profile_id(profiles: dict[str, Any], requested: str) -> str:
    if requested in profiles:
        return requested
    if requested == "pro" and DEFAULT_PROFILE in profiles:
        return DEFAULT_PROFILE
    pro_profiles = [profile_id for profile_id in sorted(profiles) if profile_id.endswith("_pro")]
    return pro_profiles[0] if pro_profiles else sorted(profiles)[0]


def _flag(
    explicit: bool | None,
    profile: dict[str, Any],
    key: str,
    default: bool,
) -> bool:
    if explicit is not None:
        return explicit
    return bool(profile.get(key, default))


def _repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (ROOT / path).resolve()


if __name__ == "__main__":
    main()
