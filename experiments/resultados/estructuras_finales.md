# Estructuras finales (trayectoria nominal)

| politica | bolas usadas | restantes | tension max / umbral | P(rotura) m=1 | m=2 | m=3 | m=4 | E[m] al llegar |
|---|---|---|---|---|---|---|---|---|
| greedy: D1 e_max=∞ | 5 | 9 | 0.457 | 0.0% | 2.3% | 25.0% | 43.0% | 1.8 |
| D2 e_max=2.5 | 8 | 6 | 0.340 | 0.0% | 0.0% | 0.0% | 0.7% | 1.2 |

P(rotura | m): 300 sorteos de m bolas de carga repartidas por grado sobre la estructura final. E[m] = p_carga x restantes (p_carga = 0.2).

## greedy: D1 e_max=∞

| paso | accion (i, j, Li, Lj, lado) | regla | posicion final (px) |
|---|---|---|---|
| 1 | (2, 3, 200.0, 200.0, -1) | avanzar | (60, 331) |
| 2 | (3, 4, 200.0, 100.0, 1) | avanzar | (-41, 336) |
| 3 | (4, 5, 200.0, 200.0, -1) | avanzar | (16, 526) |
| 4 | (5, 6, 200.0, 100.0, 1) | avanzar | (-85, 529) |
| 5 | (6, 7, 100.0, 100.0, -1) | llegar | (-32, 613) |

## D2 e_max=2.5

| paso | accion (i, j, Li, Lj, lado) | regla | posicion final (px) |
|---|---|---|---|
| 1 | (2, 3, 200.0, 200.0, -1) | avanzar | (72, 330) |
| 2 | (3, 4, 200.0, 100.0, 1) | avanzar | (-28, 341) |
| 3 | (2, 4, 200.0, 100.0, -1) | avanzar | (171, 344) |
| 4 | (0, 3, 100.0, 100.0, 1) | avanzar | (-69, 71) |
| 5 | (4, 6, 200.0, 200.0, 1) | avanzar | (92, 527) |
| 6 | (5, 8, 200.0, 100.0, 1) | avanzar | (-8, 538) |
| 7 | (1, 2, 200.0, 200.0, -1) | asegurar | (337, 68) |
| 8 | (8, 9, 100.0, 100.0, -1) | llegar | (51, 618) |
