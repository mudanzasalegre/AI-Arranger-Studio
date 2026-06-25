# 03 — Especificación musical jazz

## Objetivo musical

Generar arreglos jazz funcionales y exportables, no simples secuencias aleatorias de notas. La salida debe ser tocable, legible y editable.

## Bloques musicales iniciales

### Forma

- 12-bar blues
- minor blues
- 16-bar tune
- 32-bar AABA
- 32-bar ABAC
- rhythm changes-like original
- modal vamp
- ballad AABA
- latin/bossa 32-bar
- jazz waltz 24/32-bar

### Secciones

- intro
- head in
- A1/A2/B/A3
- solo chorus
- shout chorus
- head out
- coda/tag ending

### Roles

- drums: groove/fills/setup hits
- double_bass: walking bass, two-feel, pedal, latin ostinato
- piano: comping, voicings, fills, optional guide melody
- lead horn: melody/head/solo
- trumpet: harmony, stabs, responses
- trombone: low harmony, counterline, pads/stabs
- optional sax section: soli/harmony

## Gramática de acordes

No crear una lista cerrada de “todos los acordes”. Crear parser por gramática:

```text
ROOT + QUALITY + EXTENSIONS + ALTERATIONS + SUSPENSIONS + ADDS + OMITS + BASS
```

### Roots

```text
C, C#, Db, D, D#, Eb, E, F, F#, Gb, G, G#, Ab, A, A#, Bb, B, Cb, B#
```

### Qualities

```text
maj, major, M
min, m, minor
7, dominant
dim, diminished
aug, augmented, +
sus, sus2, sus4
ø, half-diminished
```

### Core chord symbols

```text
C
C6
Cadd9
Cmaj7
Cmaj9
Cmaj13
Cmaj7#11
Cmaj9#11
Cm
Cm6
Cm7
Cm9
Cm11
Cm13
CmMaj7
CmMaj9
C7
C9
C11
C13
C7sus4
C9sus4
C13sus4
C7b9
C7#9
C7b5
C7#5
C7#11
C7b13
C7alt
C9#11
C13b9
Cm7b5
Cø7
Cdim
Cdim7
Caug
C+7
C7#5#9
C7b9b13
```

### Add/omit/slash

```text
Cadd9
Cadd#11
Cno3
Cno5
C/E
Cmaj7/G
D7alt/Ab
```

### Alteraciones permitidas

```text
b5, #5, b9, #9, #11, b13
```

### Tensiones jazz por familia

- maj7: 9, #11, 13
- m7: 9, 11, 13 según contexto
- dominant: b9, 9, #9, #11, b13, 13
- m7b5: 11, b13, b9 según función
- dim7: notas disminuidas y passing diminished
- sus: 9, 13

## Escalas asociadas

- major / ionian
- dorian
- phrygian
- lydian
- mixolydian
- aeolian
- locrian
- melodic minor
- lydian dominant
- altered scale
- diminished whole-half
- diminished half-whole
- whole tone
- bebop major
- bebop dominant
- blues scale major/minor
- minor pentatonic
- major pentatonic

## Progresiones iniciales

### Básicas

- ii-V-I major
- iiø-V7alt-i minor
- I-vi-ii-V turnaround
- iii-vi-ii-V
- I-VI7-ii-V
- backdoor: ivm7-bVII7-I
- tritone substitution: ii-bII7-I
- secondary dominants
- cycle of fifths
- diminished passing chords

### Blues

- basic jazz blues
- bebop blues
- minor blues
- Bird-like blues original variation
- blues with tritone subs
- blues with turnaround

### AABA

- rhythm changes-like original grammar
- ballad AABA
- minor AABA
- modal AABA

### Modal

- dorian vamp
- sus vamp
- pedal point
- minor modal with planing

### Latin/Bossa

- ii-V-I chains
- minor-major interchange
- montuno-compatible harmonic loops

## Melodía

Reglas mínimas:

- frases de 2 o 4 compases
- pregunta-respuesta
- motivo inicial transformado
- uso de 3ª y 7ª como notas guía
- aproximaciones cromáticas
- enclosure bebop
- silencios respirables
- clímax melódico planificado
- resolución en cadencias

## Walking bass

Reglas mínimas:

- tiempo 1: raíz, quinta o nota estructural
- tiempo 3: nota guía/quinta/raíz alternativa
- aproximación cromática al siguiente acorde
- movimiento conjunto preferente
- saltos compensados
- rango cómodo de contrabajo
- opción two-feel en baladas/intro/head suave

## Piano comping

Reglas mínimas:

- voicings rootless cuando hay bajo
- shell voicings para densidad baja
- drop-2/drop-3 simplificado para bloque
- evitar duplicar raíz grave
- tensiones según acorde
- patrones rítmicos sincopados
- comping debe dejar espacio a melodía

## Batería

Reglas mínimas:

- ride swing
- hi-hat 2 y 4
- snare comping
- kick feathering opcional
- fills antes de nuevas secciones
- setups para hits de viento
- latin grooves para bossa/latin
- straight-eighth para funk jazz/modal moderno

## Vientos

Reglas mínimas:

- call-and-response con melodía
- stabs en huecos
- voicings a 2/3 voces
- evitar cruce trompeta/saxo/trombón
- trombón menos ágil que saxo
- frases respirables
- shout chorus homofónico básico
- falls/doits opcionales como articulaciones aproximadas

## Humanización

- swing ratio configurable
- velocity por acento
- microtiming reproducible
- duración de notas natural
- articulaciones en MusicXML y MIDI CC cuando proceda

## Estándar de salida aceptable

Una salida jazz aceptable debe:

- tener forma clara
- tener cifrado armónico visible
- tener bajo coherente con acordes
- tener piano que no invada bajo
- tener batería con feel apropiado
- tener melodía respirable
- tener vientos dentro de rango
- abrir en MuseScore
- exportar MIDI separado por pistas
