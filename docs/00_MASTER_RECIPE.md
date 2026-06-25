# 00 — Receta maestra

## Producto

AI Arranger Studio es una aplicación para generar arreglos musicales editables desde texto, primero sin IA pesada y después con modelos simbólicos. El producto debe comportarse como un “Suno simbólico”: el usuario describe una canción y recibe un proyecto musical profesional, no un audio cerrado.

## Principios no negociables

1. **Formato maestro simbólico**: todo nace en `ArrangementProject`, no en WAV/MP3.
2. **Separación por roles**: bajo, batería, piano, melodía, vientos y forma no se generan como una única masa de notas.
3. **MusicXML como salida de partitura**: el MIDI sirve para DAW; MusicXML/PDF sirven para músicos.
4. **Rule-based primero**: antes de IA, el sistema debe generar jazz funcional mediante reglas, patrones, plantillas y validación.
5. **Dataset mode desde el inicio**: el usuario puede añadir librerías MIDI/MusicXML, etiquetarlas y extraer patrones. No se entrena un modelo todavía; se aprende una librería estadística reutilizable.
6. **IA enchufable después**: cada generador debe tener una interfaz estable para que luego se cambie por un modelo.
7. **Calidad mínima profesional**: ninguna exportación pasa si tiene compases incompletos, instrumentos fuera de rango, transposición incorrecta o pistas mezcladas indebidamente.
8. **Reproducibilidad**: toda generación tiene `seed`, configuración de estilo, versión de reglas y reporte de validación.
9. **Edición parcial**: el sistema debe permitir regenerar compases/pistas sin destruir el resto.
10. **Sin dependencia de material protegido**: no copiar Real Book ni progresiones completas de standards concretos. Se pueden usar gramáticas, clichés armónicos y plantillas originales inspiradas en prácticas comunes.

## MVP fuerte sin IA

El primer MVP debe hacer bien esto:

- Text prompt → `GenerationSpec`
- `GenerationSpec` → forma, tonalidad, tempo, plantilla de instrumentos
- forma → progresión jazz rica
- progresión → lead sheet
- lead sheet → arreglo combo/sexteto jazz
- arreglo → MIDI multipista + MusicXML + PDF + partes + preview
- análisis → reporte musical

## Estilos de jazz iniciales

- swing
- bebop
- hard bop
- jazz ballad
- modal jazz
- minor blues
- rhythm changes-like
- bossa nova / latin jazz básico
- jazz waltz
- funk jazz / straight-eighth jazz

## Ensembles iniciales

- jazz trio: piano, contrabajo, batería
- jazz quartet: + saxo alto o tenor
- jazz quintet: saxo, trompeta, piano, contrabajo, batería
- jazz sextet: saxo, trompeta, trombón, piano, contrabajo, batería
- concert-band-lite futuro: flauta, clarinetes, saxos, trompetas, trombones, bombardino, tuba, percusión

## Resultado esperado al acabar la receta

Una aplicación local/web donde se pueda escribir:

> “Jazz hard bop en Fa menor, 140 bpm, AABA, sexteto con saxo alto, trompeta y trombón, bajo caminante, piano rootless comping, batería swing y shout chorus.”

Y recibir:

- proyecto editable
- MIDI completo
- MIDI por pistas
- MusicXML
- full score PDF
- particellas transpuestas
- audio preview
- reporte de validación

## Cómo se consigue riqueza sin IA

La riqueza vendrá de combinar:

- librería de progresiones
- gramática de acordes extensible
- patrones rítmicos por estilo
- células de walking bass
- voicings jazz por familia de acorde
- motivos melódicos transformables
- respuestas de vientos
- humanización reproducible
- aprendizaje estadístico desde librerías MIDI/MusicXML autorizadas

## No hacer en la primera versión

- No entrenar redes neuronales todavía.
- No extraer MIDI desde audio.
- No generar voces cantadas.
- No copiar canciones existentes.
- No usar datasets sin manifiesto de licencia.
- No escribir todo como un único generador opaco.
