# Taller 5: agente base en dos condiciones

Agente base: PoliticaT4 e_max = ∞ (greedy), D1, k = 2, REGLAS_T4. Meta: altura. Semillas 0–199 por condición.
Condiciones: referencia (θ por defecto, p_carga 0,2) y adversa (umbral ×0,75, p_carga 0,4).
A*: plan geométrico reproducido en las mismas condiciones y semillas; es una referencia, no un agente.
Bolas: el fracaso cuenta 0; IC 95% bootstrap de la media (5000 remuestreos). Éxito: IC 95% de Wilson.

| condición | política | bolas μ ± σ | IC95 bolas | éxito [IC95] | colapso (propio / carga) | pasos μ ± σ |
|---|---|---|---|---|---|---|
| referencia | greedy (agente base) | 7.96 ± 2.88 | [7.56, 8.37] | 88.5% [83.3, 92.2] | 11.5% (0.0 / 11.5) | 4.99 ± 0.12 |
| referencia | A* (referencia, no agente) | 7.92 ± 2.93 | [7.51, 8.28] | 88.0% [82.8, 91.8] | 12.0% (0.0 / 12.0) | 4.92 ± 0.35 |
| adversa | greedy (agente base) | 1.94 ± 3.71 | [1.44, 2.43] | 21.5% [16.4, 27.7] | 78.5% (0.0 / 78.5) | 4.47 ± 0.74 |
| adversa | A* (referencia, no agente) | 3.38 ± 4.37 | [2.79, 3.96] | 37.5% [31.1, 44.4] | 62.5% (0.0 / 62.5) | 4.19 ± 1.11 |
