# Percentiles de e (50 episodios por fuente, 8 s)

## Definicion 'apoyo'

| fuente | muestra | N | p5 | p25 | p50 | p75 | p95 |
|---|---|---|---|---|---|---|---|
| aleatoria | e(s) | 549 | 0.99 | 1.57 | 2.16 | 2.57 | 3.00 |
| aleatoria | e(s,a) | 10086 | 1.29 | 2.00 | 2.32 | 2.78 | 3.54 |
| aleatoria | e(s,a), a en L(s) | 61 | 4.01 | 4.06 | 4.09 | 4.12 | 4.25 |
| t4_apoyo_inf | e(s) | 247 | 0.99 | 2.22 | 2.29 | 3.53 | 3.54 |
| t4_apoyo_inf | e(s,a) | 4325 | 1.17 | 2.22 | 3.53 | 3.54 | 3.54 |
| t4_apoyo_inf | e(s,a), a en L(s) | 94 | 4.11 | 4.11 | 4.46 | 4.82 | 4.82 |
| t4_tercio_inf | e(s) | 247 | 0.99 | 2.22 | 2.29 | 3.53 | 3.54 |
| t4_tercio_inf | e(s,a) | 4325 | 1.17 | 2.22 | 3.53 | 3.54 | 3.54 |
| t4_tercio_inf | e(s,a), a en L(s) | 94 | 4.11 | 4.11 | 4.46 | 4.82 | 4.82 |

## Definicion 'tercio'

| fuente | muestra | N | p5 | p25 | p50 | p75 | p95 |
|---|---|---|---|---|---|---|---|
| aleatoria | e(s) | 549 | 0.57 | 0.74 | 0.99 | 1.28 | 2.20 |
| aleatoria | e(s,a) | 10086 | 0.55 | 0.72 | 0.98 | 1.29 | 2.22 |
| aleatoria | e(s,a), a en L(s) | 61 | 0.99 | 1.16 | 1.26 | 1.54 | 2.58 |
| t4_apoyo_inf | e(s) | 247 | 0.99 | 2.22 | 2.29 | 3.45 | 3.49 |
| t4_apoyo_inf | e(s,a) | 4325 | 0.99 | 1.64 | 2.29 | 3.45 | 3.49 |
| t4_apoyo_inf | e(s,a), a en L(s) | 94 | 4.00 | 4.00 | 4.35 | 4.70 | 4.70 |
| t4_tercio_inf | e(s) | 247 | 0.99 | 2.22 | 2.29 | 3.45 | 3.49 |
| t4_tercio_inf | e(s,a) | 4325 | 0.99 | 1.64 | 2.29 | 3.45 | 3.49 |
| t4_tercio_inf | e(s,a), a en L(s) | 94 | 4.00 | 4.00 | 4.35 | 4.70 | 4.70 |

## Episodios

| fuente | exito | causas |
|---|---|---|
| aleatoria | 0/50 | {'colapso_carga': 13, 'colapso_propio': 18, 'sin_bolas': 19} |
| t4_apoyo_inf | 43/50 | {'meta': 43, 'colapso_carga': 7} |
| t4_tercio_inf | 43/50 | {'meta': 43, 'colapso_carga': 7} |
