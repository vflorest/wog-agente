# Chequeos de la revision visual

Un cuadro cada 6 pasos de fisica (0.10 s). Posiciones en px, centro de la bola.

## greedy (D1 e_max=∞)

- bolas: 5; altura final 613.3 px; causa: meta
- centro mas bajo de una bola dinamica en TODOS los cuadros: 143.75 px (radio 15; < 15 = penetra el suelo)
- cruces propios de uniones en la estructura final: [((0, 2), (1, 3))] (las diagonales del K4 base, por diseño)
- holgura minima bola-union (no extremo): 85.6 px
- pico de tension/umbral en toda la construccion: 0.459 (cuadro 147, paso 5)

| paso | accion | P predicha | posicion final | |final − P| | max desplazamiento de las previas |
|---|---|---|---|---|---|
| 1 | (2, 3, 200.0, 200.0, -1) | (75, 335) | (75, 333) | 1.2 | 0.5 |
| 2 | (3, 4, 200.0, 100.0, 1) | (-24, 347) | (-30, 344) | 6.5 | 5.1 |
| 3 | (4, 5, 200.0, 200.0, -1) | (42, 531) | (45, 529) | 3.8 | 1.5 |
| 4 | (5, 6, 200.0, 100.0, 1) | (-54, 542) | (-83, 531) | 31.1 | 28.2 |
| 5 | (6, 7, 100.0, 100.0, -1) | (-30, 616) | (-32, 613) | 3.3 | 2.7 |

## A* (5 bolas)

- bolas: 5; altura final 609.4 px; causa: meta
- centro mas bajo de una bola dinamica en TODOS los cuadros: 145.72 px (radio 15; < 15 = penetra el suelo)
- cruces propios de uniones en la estructura final: [((0, 2), (1, 3))] (las diagonales del K4 base, por diseño)
- holgura minima bola-union (no extremo): 66.7 px
- pico de tension/umbral en toda la construccion: 0.343 (cuadro 193, paso 5)

| paso | accion | P predicha | posicion final | |final − P| | max desplazamiento de las previas |
|---|---|---|---|---|---|
| 1 | (2, 3, 100.0, 200.0, -1) | (175, 246) | (179, 242) | 5.9 | 1.8 |
| 2 | (3, 4, 200.0, 200.0, 1) | (10, 350) | (4, 347) | 6.4 | 3.8 |
| 3 | (4, 5, 200.0, 200.0, -1) | (180, 444) | (200, 435) | 21.6 | 14.5 |
| 4 | (5, 6, 200.0, 200.0, 1) | (34, 548) | (12, 544) | 23.1 | 17.7 |
| 5 | (6, 7, 200.0, 100.0, -1) | (83, 614) | (109, 609) | 25.7 | 22.1 |

## D2 e_max=2.5

- bolas: 8; altura final 618.0 px; causa: meta
- centro mas bajo de una bola dinamica en TODOS los cuadros: 63.78 px (radio 15; < 15 = penetra el suelo)
- cruces propios de uniones en la estructura final: [((0, 2), (1, 3))] (las diagonales del K4 base, por diseño)
- holgura minima bola-union (no extremo): 68.7 px
- pico de tension/umbral en toda la construccion: 0.363 (cuadro 262, paso 8)

| paso | accion | P predicha | posicion final | |final − P| | max desplazamiento de las previas |
|---|---|---|---|---|---|
| 1 | (2, 3, 200.0, 200.0, -1) | (75, 335) | (75, 333) | 1.2 | 0.5 |
| 2 | (3, 4, 200.0, 100.0, 1) | (-24, 347) | (-30, 344) | 6.5 | 5.1 |
| 3 | (2, 4, 200.0, 100.0, -1) | (169, 348) | (177, 345) | 9.5 | 8.6 |
| 4 | (0, 3, 100.0, 100.0, 1) | (-67, 74) | (-69, 71) | 4.2 | 9.6 |
| 5 | (4, 6, 200.0, 200.0, 1) | (90, 531) | (98, 529) | 8.3 | 6.6 |
| 6 | (5, 8, 200.0, 100.0, 1) | (-1, 543) | (-14, 539) | 13.7 | 12.5 |
| 7 | (1, 2, 200.0, 200.0, -1) | (335, 76) | (338, 64) | 12.8 | 28.0 |
| 8 | (8, 9, 100.0, 100.0, -1) | (75, 620) | (51, 618) | 23.9 | 22.6 |

## Episodio con colapso (greedy, umbral ×0.75, p_carga 0.4)

- seed 0 (primera de 0..199 que colapsa por carga en el paso de llegada, paso 5)
- pruebas de carga: m = 6 (en fijas: 0), m = 6 (en fijas: 0), m = 5 (en fijas: 0), m = 1 (en fijas: 0), m = 4 (en fijas: 0)
- centro mas bajo de una bola dinamica (incluida la caida): 18.25 px
- rupturas:
  - fase carga, paso de fisica 152: F = 27.04 (1.001 × umbral); mayor tension en ese paso 27.04 -> es la mayor
  - fase carga, paso de fisica 206: F = 27.11 (1.004 × umbral); mayor tension en ese paso 27.11 -> es la mayor
  - fase carga, paso de fisica 300: F = 27.04 (1.001 × umbral); mayor tension en ese paso 27.04 -> es la mayor
  - fase carga, paso de fisica 380: F = 28.25 (1.046 × umbral); mayor tension en ese paso 28.25 -> es la mayor
  - fase carga, paso de fisica 383: F = 27.66 (1.024 × umbral); mayor tension en ese paso 27.66 -> es la mayor

