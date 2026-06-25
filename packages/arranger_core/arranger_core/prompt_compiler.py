from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from arranger_core.catalogs import InstrumentCatalog, StyleProfileCatalog
from arranger_core.schema import DEFAULT_METER, GenerationSpec

DEFAULT_STYLE = "hard_bop"
DEFAULT_KEY = "C minor"
DEFAULT_FORM = "minor_blues_12"
DEFAULT_ENSEMBLE = "jazz_sextet"
DEFAULT_TEMPO = 132
DEFAULT_DENSITY = "medium"
DEFAULT_COMPLEXITY = 0.75

ENSEMBLE_INSTRUMENT_ORDER = [
    "drum_kit",
    "double_bass",
    "piano",
    "alto_sax",
    "tenor_sax",
    "trumpet_bflat",
    "trombone",
    "flute",
    "clarinet_bflat",
    "baritone_sax",
    "tuba",
]

KEY_ROOTS = {
    "c": "C",
    "do": "C",
    "c#": "C#",
    "c sharp": "C#",
    "do sostenido": "C#",
    "db": "Db",
    "d flat": "Db",
    "re bemol": "Db",
    "d": "D",
    "re": "D",
    "d#": "D#",
    "d sharp": "D#",
    "re sostenido": "D#",
    "eb": "Eb",
    "e flat": "Eb",
    "mi bemol": "Eb",
    "e": "E",
    "mi": "E",
    "f": "F",
    "fa": "F",
    "f#": "F#",
    "f sharp": "F#",
    "fa sostenido": "F#",
    "gb": "Gb",
    "g flat": "Gb",
    "sol bemol": "Gb",
    "g": "G",
    "sol": "G",
    "g#": "G#",
    "g sharp": "G#",
    "sol sostenido": "G#",
    "ab": "Ab",
    "a flat": "Ab",
    "la bemol": "Ab",
    "a": "A",
    "la": "A",
    "a#": "A#",
    "a sharp": "A#",
    "la sostenido": "A#",
    "bb": "Bb",
    "b flat": "Bb",
    "si bemol": "Bb",
    "b": "B",
    "si": "B",
}

STYLE_PATTERNS = [
    ("hard_bop", ("hard bop", "hard-bop")),
    ("bebop", ("bebop", "be bop", "bop")),
    ("jazz_ballad", ("jazz ballad", "ballad", "balada")),
    ("modal_jazz", ("modal jazz", "jazz modal", "modal")),
    ("bossa_nova", ("bossa nova", "bossa", "latin jazz", "latin")),
    ("funk_jazz", ("funk jazz", "jazz funk", "straight eighth", "straight-eighth")),
    ("swing", ("swing",)),
    ("jazz_waltz", ("jazz waltz", "waltz", "vals jazz", "vals")),
]

FORM_PATTERNS = [
    ("minor_blues_12", 12, ("minor blues", "blues minor", "blues menor")),
    ("jazz_blues_12", 12, ("jazz blues", "12 bar blues", "blues de 12", "blues")),
    ("rhythm_changes_like", 32, ("rhythm changes", "rhythm changes-like")),
    ("ballad_aaba_32", 32, ("ballad aaba", "balada aaba")),
    ("aaba_32", 32, ("aaba", "32 bar", "32 compases")),
    ("modal_vamp", 8, ("modal vamp", "vamp", "pedal")),
    ("bossa_32", 32, ("bossa 32", "latin 32")),
    ("jazz_waltz_32", 32, ("jazz waltz", "vals jazz", "vals")),
]

ENSEMBLE_PATTERNS = [
    ("jazz_sextet", ("sextet", "sexteto", "six piece")),
    ("jazz_quintet", ("quintet", "quinteto", "five piece")),
    ("jazz_quartet", ("quartet", "cuarteto", "four piece")),
    ("jazz_trio", ("trio", "tres instrumentos")),
    ("concert_band_lite", ("concert band", "banda", "concert-band-lite")),
]

INSTRUMENT_PATTERNS = [
    ("alto_sax", ("alto saxophone", "alto sax", "saxo alto", "sax alto")),
    ("tenor_sax", ("tenor saxophone", "tenor sax", "saxo tenor", "sax tenor")),
    ("baritone_sax", ("baritone saxophone", "baritone sax", "saxo baritono")),
    ("trumpet_bflat", ("trumpet", "trompeta")),
    ("trombone", ("trombone", "trombon")),
    ("piano", ("piano",)),
    ("double_bass", ("double bass", "upright bass", "contrabajo", "walking bass")),
    ("drum_kit", ("drum kit", "drums", "bateria", "drummer")),
    ("flute", ("flute", "flauta")),
    ("clarinet_bflat", ("clarinet", "clarinete")),
    ("tuba", ("tuba",)),
]

MOOD_PATTERNS = [
    ("nocturnal", ("nocturno", "night", "nighttime", "dark", "oscuro")),
    ("bright", ("bright", "luminoso", "alegre")),
    ("energetic", ("energetic", "energia", "energetico", "intenso")),
    ("relaxed", ("relaxed", "relajado", "suave", "laid back")),
    ("lyrical", ("lyrical", "lirico", "cantabile")),
]

DENSITY_PATTERNS = [
    ("low", ("sparse", "espaciado", "poco denso", "ligero")),
    ("low_medium", ("low medium", "medio bajo", "light")),
    ("medium_high", ("medium high", "medio alto")),
    ("high", ("dense", "denso", "busy", "full", "cargado")),
    ("medium", ("medium", "medio")),
]

ROLE_PATTERNS = {
    "bass": {
        "walking_bass": ("walking bass", "bajo caminante", "walking"),
        "two_feel": ("two feel", "two-feel"),
    },
    "piano": {
        "rootless_comping": ("rootless", "rootless comping"),
        "comping": ("comping", "acompanamiento"),
    },
    "drums": {
        "swing_ride": ("swing ride", "ride swing"),
        "brushes": ("brushes", "escobillas"),
    },
    "horns": {
        "shout_chorus": ("shout chorus", "soli", "ensemble shout"),
        "call_response": ("call and response", "call-and-response", "pregunta respuesta"),
    },
}


@dataclass(frozen=True)
class CompileDefaults:
    style: str = DEFAULT_STYLE
    key: str = DEFAULT_KEY
    form: str = DEFAULT_FORM
    ensemble: str = DEFAULT_ENSEMBLE
    tempo: int = DEFAULT_TEMPO
    density: str = DEFAULT_DENSITY
    complexity: float = DEFAULT_COMPLEXITY
    meter: str = DEFAULT_METER


class PromptCompiler:
    def __init__(
        self,
        *,
        instrument_catalog: InstrumentCatalog | None = None,
        style_catalog: StyleProfileCatalog | None = None,
        defaults: CompileDefaults | None = None,
    ) -> None:
        self.instrument_catalog = instrument_catalog or InstrumentCatalog.load_default()
        self.style_catalog = style_catalog or StyleProfileCatalog.load_default()
        self.defaults = defaults or CompileDefaults()

    def compile(self, prompt: str, *, seed: int = 0) -> GenerationSpec:
        normalized = normalize_prompt(prompt)
        style = self._extract_style(normalized)
        form, duration_bars = self._extract_form(normalized, style)
        ensemble = self._extract_ensemble(normalized)
        explicit_instruments = self._extract_instruments(normalized)
        ensemble = self._resolve_quartet_variant(ensemble, explicit_instruments)
        instruments = self._resolve_instruments(ensemble, explicit_instruments)
        tempo = self._extract_tempo(normalized) or self._default_tempo(style)
        key = self._extract_key(normalized) or self.defaults.key
        density = self._extract_density(normalized) or self._default_density(style)
        mood = self._extract_mood(normalized)
        meter = self._extract_meter(normalized, style)
        complexity = self._default_complexity(style)
        roles = self._extract_roles(normalized)

        return GenerationSpec(
            prompt=prompt,
            style=style,
            substyle=self._substyle_from_form(form),
            tempo=tempo,
            key=key,
            meter=meter,
            form=form,
            ensemble=ensemble,
            duration_bars=duration_bars,
            density=density,
            mood=mood,
            complexity=complexity,
            instruments=instruments,
            constraints={
                "detected_instruments": explicit_instruments,
                "roles": roles,
                "compiler": "deterministic_v0",
                "fallbacks": self._fallbacks_used(
                    normalized=normalized,
                    style=style,
                    key=key,
                    form=form,
                    ensemble=ensemble,
                    tempo=tempo,
                    density=density,
                ),
            },
            seed=seed,
        )

    def _extract_style(self, normalized: str) -> str:
        return _first_pattern_match(normalized, STYLE_PATTERNS) or self.defaults.style

    def _extract_form(self, normalized: str, style: str) -> tuple[str, int | None]:
        match = _first_form_match(normalized)
        if match:
            return match
        if style == "jazz_ballad":
            return "ballad_aaba_32", 32
        if style == "modal_jazz":
            return "modal_vamp", 8
        if style == "jazz_waltz":
            return "jazz_waltz_32", 32
        return self.defaults.form, 12

    def _extract_ensemble(self, normalized: str) -> str:
        return _first_pattern_match(normalized, ENSEMBLE_PATTERNS) or self.defaults.ensemble

    def _extract_instruments(self, normalized: str) -> list[str]:
        detected = [
            instrument_id
            for instrument_id, patterns in INSTRUMENT_PATTERNS
            if _contains_any(normalized, patterns)
        ]
        return _ordered_unique_instruments(detected)

    def _resolve_quartet_variant(self, ensemble: str, instruments: list[str]) -> str:
        if ensemble != "jazz_quartet":
            return ensemble
        if "tenor_sax" in instruments:
            return "jazz_quartet_tenor"
        return "jazz_quartet_alto"

    def _resolve_instruments(self, ensemble: str, explicit_instruments: list[str]) -> list[str]:
        if explicit_instruments:
            if ensemble in self.instrument_catalog.ensembles:
                ensemble_instruments = self.instrument_catalog.get_ensemble(ensemble).instruments
                merged = [*ensemble_instruments, *explicit_instruments]
                return _ordered_unique_instruments(merged)
            return _ordered_unique_instruments(explicit_instruments)

        if ensemble in self.instrument_catalog.ensembles:
            return list(self.instrument_catalog.get_ensemble(ensemble).instruments)
        return list(self.instrument_catalog.get_ensemble(self.defaults.ensemble).instruments)

    def _extract_tempo(self, normalized: str) -> int | None:
        match = re.search(r"\b(?P<tempo>[4-9]\d|1\d{2}|2\d{2}|3[0-1]\d|320)\s*bpm\b", normalized)
        if match:
            return int(match.group("tempo"))
        match = re.search(
            r"\b(?P<tempo>[4-9]\d|1\d{2}|2\d{2}|3[0-1]\d|320)\s*"
            r"(?:beats|tempo)\b",
            normalized,
        )
        if match:
            return int(match.group("tempo"))
        return None

    def _extract_key(self, normalized: str) -> str | None:
        root_pattern = "|".join(
            re.escape(root)
            for root in sorted(KEY_ROOTS, key=len, reverse=True)
        )
        mode_pattern = r"minor|menor|major|mayor"
        patterns = [
            rf"\b(?:en|in|key of|tono de|tonalidad de)\s+"
            rf"(?P<root>{root_pattern})\s+(?P<mode>{mode_pattern})\b",
            rf"\b(?P<root>{root_pattern})\s+(?P<mode>{mode_pattern})\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, normalized)
            if match:
                root = KEY_ROOTS[match.group("root")]
                mode = "minor" if match.group("mode") in {"minor", "menor"} else "major"
                return f"{root} {mode}"
        return None

    def _extract_density(self, normalized: str) -> str | None:
        return _first_pattern_match(normalized, DENSITY_PATTERNS)

    def _extract_mood(self, normalized: str) -> str | None:
        return _first_pattern_match(normalized, MOOD_PATTERNS)

    def _extract_meter(self, normalized: str, style: str) -> str:
        meter_match = re.search(
            r"\b(?P<numerator>\d{1,2})\s*/\s*(?P<denominator>\d{1,2})\b",
            normalized,
        )
        if meter_match:
            return f"{meter_match.group('numerator')}/{meter_match.group('denominator')}"
        if style == "jazz_waltz" or _contains_any(normalized, ("waltz", "vals", "3 4")):
            return "3/4"
        return self.defaults.meter

    def _extract_roles(self, normalized: str) -> dict[str, str]:
        roles: dict[str, str] = {}
        for role, options in ROLE_PATTERNS.items():
            for role_value, patterns in options.items():
                if _contains_any(normalized, patterns):
                    roles[role] = role_value
                    break
        return roles

    def _default_tempo(self, style: str) -> int:
        try:
            profile = self.style_catalog.get(style)
        except KeyError:
            return self.defaults.tempo

        if not profile.tempo_ranges:
            return self.defaults.tempo

        preferred_band = "medium"
        if style == "jazz_ballad":
            preferred_band = "medium_slow" if "medium_slow" in profile.tempo_ranges else "slow"
        tempo_range = profile.tempo_ranges.get(preferred_band)
        if tempo_range is None:
            tempo_range = next(iter(profile.tempo_ranges.values()))
        return round((tempo_range[0] + tempo_range[1]) / 2)

    def _default_density(self, style: str) -> str:
        try:
            profile = self.style_catalog.get(style)
        except KeyError:
            return self.defaults.density
        return str(profile.rhythm.get("comping_density", self.defaults.density))

    def _default_complexity(self, style: str) -> float:
        try:
            profile = self.style_catalog.get(style)
        except KeyError:
            return self.defaults.complexity
        return float(profile.harmony.get("default_complexity", self.defaults.complexity))

    def _fallbacks_used(
        self,
        *,
        normalized: str,
        style: str,
        key: str,
        form: str,
        ensemble: str,
        tempo: int,
        density: str,
    ) -> dict[str, bool]:
        return {
            "style": (
                style == self.defaults.style
                and not _first_pattern_match(normalized, STYLE_PATTERNS)
            ),
            "key": key == self.defaults.key and self._extract_key(normalized) is None,
            "form": form == self.defaults.form and _first_form_match(normalized) is None,
            "ensemble": (
                ensemble == self.defaults.ensemble
                and _first_pattern_match(normalized, ENSEMBLE_PATTERNS) is None
            ),
            "tempo": (
                tempo == self._default_tempo(style)
                and self._extract_tempo(normalized) is None
            ),
            "density": (
                density == self._default_density(style)
                and self._extract_density(normalized) is None
            ),
        }

    @staticmethod
    def _substyle_from_form(form: str) -> str | None:
        if form == "minor_blues_12":
            return "minor_blues"
        if form == "jazz_blues_12":
            return "jazz_blues"
        return None


def compile_prompt(prompt: str, *, seed: int = 0) -> GenerationSpec:
    return PromptCompiler().compile(prompt, seed=seed)


def normalize_prompt(prompt: str) -> str:
    normalized = unicodedata.normalize("NFKD", prompt)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    lowered = ascii_text.lower()
    lowered = re.sub(r"[-_]", " ", lowered)
    lowered = re.sub(r"[^\w/#\s]", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip()


def _contains_any(normalized: str, patterns: tuple[str, ...]) -> bool:
    return any(_contains_phrase(normalized, pattern) for pattern in patterns)


def _contains_phrase(normalized: str, phrase: str) -> bool:
    phrase = normalize_prompt(phrase)
    return re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", normalized) is not None


def _first_pattern_match(
    normalized: str,
    pattern_groups: list[tuple[str, tuple[str, ...]]],
) -> str | None:
    for value, patterns in pattern_groups:
        if _contains_any(normalized, patterns):
            return value
    return None


def _first_form_match(normalized: str) -> tuple[str, int | None] | None:
    for form, duration_bars, patterns in FORM_PATTERNS:
        if _contains_any(normalized, patterns):
            return form, duration_bars
    return None


def _ordered_unique_instruments(instrument_ids: list[str]) -> list[str]:
    unique = set(instrument_ids)
    ordered = [
        instrument_id for instrument_id in ENSEMBLE_INSTRUMENT_ORDER if instrument_id in unique
    ]
    ordered.extend(
        instrument_id for instrument_id in instrument_ids if instrument_id not in ordered
    )
    return ordered
