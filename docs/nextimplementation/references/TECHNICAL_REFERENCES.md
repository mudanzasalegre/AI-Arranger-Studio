# Technical references for implementers

Este archivo existe para que Codex/devs sepan qué herramientas externas se contemplaron. Revisar siempre documentación oficial antes de activar un backend real.

## MIDI-GPT

Uso previsto: backend simbólico para multitrack infill/generate-track/continue-section.

Notas de integración:

- instalar opcionalmente;
- no importar en import-time;
- probar primero con mock;
- revisar licencia de código/pesos antes de uso comercial.

## Text2MIDI

Uso previsto: backend experimental `prompt -> MIDI sketch`.

Notas:

- no usar como export final directo;
- importar siempre a ArrangementProject sketch;
- marcar confianza.

## MidiTok

Uso previsto: tokenización futura para modelos propios por rol.

## MuseScore CLI

Uso previsto: export PDF/MusicXML/audio preview si ya está en stack.

## Política

Este plan no exige que ninguno de estos modelos esté instalado para que la app base funcione.
