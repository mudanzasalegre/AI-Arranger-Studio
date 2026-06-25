import json
from pathlib import Path


def test_next_app_has_required_entrypoints():
    root = Path(__file__).resolve().parents[1]
    web_root = root / "apps/web"
    package = json.loads((web_root / "package.json").read_text(encoding="utf-8"))

    assert (web_root / "app/layout.tsx").exists()
    assert (web_root / "app/page.tsx").exists()
    assert (web_root / "app/globals.css").exists()
    assert (web_root / "next.config.mjs").exists()
    assert package["scripts"]["dev"].startswith("next dev")
    assert package["scripts"]["lint"] == "tsc --noEmit"


def test_web_studio_exposes_objective_11_views_and_api_contracts():
    root = Path(__file__).resolve().parents[1]
    page = (root / "apps/web/app/page.tsx").read_text(encoding="utf-8")

    for label in [
        "Home",
        "New project",
        "Project detail",
        "Score viewer",
        "Mixer",
        "Chord/form",
        "Validation",
        "Datasets",
        "Export",
    ]:
        assert label in page

    for endpoint in [
        "/v1/projects/generate",
        "/v1/projects/",
        "/v1/datasets/import",
        "/v1/datasets",
        "/v1/patterns/search",
    ]:
        assert endpoint in page

    assert "opensheetmusicdisplay" in page
    assert "Download ZIP" in page
    assert "Play preview" in page
    for preset_label in [
        "Hard bop sextet",
        "Bebop blues",
        "Swing AABA",
        "Ballad quartet",
        "Modal quintet",
        "Bossa quartet",
        "Jazz waltz",
        "Funk jazz",
    ]:
        assert preset_label in page
