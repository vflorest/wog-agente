# Meta "boca" del nivel real (solo geometria)

A* con REGLAS_T4, k = 2, 14 bolas. Meta: algun nodo a <= 75 px de la boca E. Generado por `experiments/meta_real.py`.

## Referencia actual (meta altura >= 600)

- Minimo: 5 bolas; P final (78.65, 620.00)

## a) Base del modelo (K4 cuadrado 150 px)

- Base: 0 (0.00, 0.00), 1 (150.00, 0.00), 2 (150.00, 150.00), 3 (0.00, 150.00)
- E = (93.25, 414.00), radio 75
- **Minimo: 2 bolas** (sin cierre: 2; expandidos 13 / 13, 0.01 s / 0.01 s)

| paso | nodo | anclas (i, j) | Li, Lj | lado | P | dist a E |
|---|---|---|---|---|---|---|
| 1 | 4 | (0, 2) | 200, 200 | +1 | (-44.90, 194.90) | 259.02 |
| 2 | 5 | (2, 4) | 200, 200 | -1 | (91.43, 341.23) | 72.79 |

- Bola que entra al radio: nodo 5 en (91.43, 341.23) (dist 72.79)

## b) Base real de GoingUp (centros inferiores en y = 0)

- Base: 0 (-88.50, -1.50), 1 (49.50, 1.50), 2 (39.50, 105.50), 3 (-73.50, 89.50)
- E = (0.00, 414.00), radio 75
- **Minimo: 3 bolas** (sin cierre: 3; expandidos 100 / 122, 0.20 s / 0.22 s)

| paso | nodo | anclas (i, j) | Li, Lj | lado | P | dist a E |
|---|---|---|---|---|---|---|
| 1 | 4 | (0, 2) | 200, 200 | +1 | (-141.08, 191.46) | 263.49 |
| 2 | 5 | (2, 4) | 200, 200 | -1 | (23.66, 304.87) | 111.66 |
| 3 | 6 | (4, 5) | 200, 100 | +1 | (-51.84, 370.45) | 67.71 |

- Bola que entra al radio: nodo 6 en (-51.84, 370.45) (dist 67.71)

## Resumen

| meta | base | minimo |
|---|---|---|
| altura >= 600 | modelo | 5 |
| boca r = 75 | modelo | 2 |
| boca r = 75 | real | 3 |
