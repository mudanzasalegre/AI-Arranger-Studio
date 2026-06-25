import json
import subprocess
import sys
from pathlib import Path

from arranger_core import PromptCompiler, compile_prompt, normalize_prompt

ACCEPTANCE_PROMPT = (
    "hard bop nocturno en Do menor, 132 bpm, blues menor, sexteto con saxo alto, "
    "trompeta, trombon, piano, contrabajo y bateria"
)


def test_acceptance_prompt_compiles_to_required_generation_spec():
    spec = compile_prompt(ACCEPTANCE_PROMPT, seed=1234)

    assert spec.style == "hard_bop"
    assert spec.key == "C minor"
    assert spec.tempo == 132
    assert spec.form == "minor_blues_12"
    assert spec.ensemble == "jazz_sextet"
    assert spec.duration_bars == 12
    assert spec.mood == "nocturnal"
    assert spec.instruments == [
        "drum_kit",
        "double_bass",
        "piano",
        "alto_sax",
        "trumpet_bflat",
        "trombone",
    ]
    assert spec.seed == 1234


def test_prompt_compiler_handles_english_quartet_and_tenor_sax():
    prompt = "modal jazz in D minor, 118 bpm, quartet with tenor sax, piano, bass and drums"

    spec = PromptCompiler().compile(prompt, seed=9)

    assert spec.style == "modal_jazz"
    assert spec.key == "D minor"
    assert spec.tempo == 118
    assert spec.form == "modal_vamp"
    assert spec.ensemble == "jazz_quartet_tenor"
    assert spec.instruments == ["drum_kit", "double_bass", "piano", "tenor_sax"]
    assert spec.seed == 9


def test_prompt_compiler_uses_intelligent_defaults_for_sparse_prompt():
    spec = compile_prompt("make some jazz", seed=1)

    assert spec.style == "hard_bop"
    assert spec.key == "C minor"
    assert spec.form == "minor_blues_12"
    assert spec.ensemble == "jazz_sextet"
    assert spec.tempo == 129
    assert spec.constraints["fallbacks"]["style"] is True
    assert spec.constraints["fallbacks"]["key"] is True
    assert spec.constraints["fallbacks"]["ensemble"] is True


def test_prompt_compiler_extracts_density_mood_roles_and_waltz_meter():
    prompt = "sparse lyrical jazz waltz in F major, trio, brushes and rootless comping"

    spec = compile_prompt(prompt)

    assert spec.style == "jazz_waltz"
    assert spec.key == "F major"
    assert spec.meter == "3/4"
    assert spec.form == "jazz_waltz_32"
    assert spec.ensemble == "jazz_trio"
    assert spec.density == "low"
    assert spec.mood == "lyrical"
    assert spec.constraints["roles"] == {
        "piano": "rootless_comping",
        "drums": "brushes",
    }


def test_normalize_prompt_removes_accents_for_dictionary_matching():
    assert normalize_prompt("trombón y batería en Do menor") == "trombon y bateria en do menor"


def test_compile_prompt_cli_outputs_generation_spec_json():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/compile_prompt.py"),
            "--prompt",
            ACCEPTANCE_PROMPT,
            "--seed",
            "1234",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["style"] == "hard_bop"
    assert payload["key"] == "C minor"
    assert payload["seed"] == 1234
