from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load_module():
    script_path = ROOT / "scripts" / "models_pro" / "generate_professional_midi.py"
    spec = importlib.util.spec_from_file_location("generate_professional_midi", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_resolve_profile_id_supports_pro_alias():
    module = _load_module()
    profiles = {
        "hard_bop_minor_blues_sextet_pro": {},
        "jazz_ballad_quartet_pro": {},
    }

    assert module._resolve_profile_id(profiles, "pro") == "hard_bop_minor_blues_sextet_pro"
    assert module._resolve_profile_id(profiles, "jazz_ballad_quartet_pro") == (
        "jazz_ballad_quartet_pro"
    )
