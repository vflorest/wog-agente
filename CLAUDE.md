# CLAUDE.md

Proyecto individual del curso Agentes Inteligentes (UTFSM). Agente que juega World of Goo nivel 1.

## Nivel
- Base de 4 bolas (K4), 14 bolas disponibles, meta a 600 px.
- Para pasar deben quedar al menos 4 bolas sin usar.

## Acción
- Colocar una bola anclada a 2 existentes (extensión Henneberg tipo I, preserva rigidez por Laman).
- Largos parametrizados por k.
- Acciones discretas y siempre modifican la estructura (sin no-op).

## Fases
1. Geometría con NetworkX, enumeración/A*.
2. Física Pymunk con ruptura por tensión.
3. MaskablePPO + GNN.
4. Domain randomization.

## MDP (Taller 4)
- Estado: s = (G, pos, n).
- Un paso = colocar bola + estabilizar.
- Carga móvil W sorteada cada turno, fuera del estado.
- Parámetros físicos θ fijos y desconocidos para el agente (MDP con modelo parcialmente desconocido, no POMDP).
- Recompensa: n si llega con n ≥ 4, si no 0.
- Shaping: F = Φ(s') − Φ(s) con Φ = altura y Φ = 0 en terminales.
- γ = 1, horizonte 14.

## Física
- Parámetros derivados de balls.xml en `wog/fisica.py` (escala 1 u = 10 px, umbral de ruptura = 12 pesos de bola).
- La construcción SIEMPRE es incremental: construir todo de una vez produce picos de tensión falsos.

## Stack
uv, Python 3.12, networkx, pymunk, gymnasium. macOS Apple Silicon.

## Código
- `wog/geometria.py`: primitivas, `Estado` y un único `generar_acciones` con filtros según un objeto `Reglas`. Presets: `REGLAS_FASE1`, `REGLAS_NOTEBOOK`, `REGLAS_GYM_T2` (reproducen Taller 2) y `REGLAS_T4` (entorno futuro: suelo, alcance, cruce permisivo, holgura, envolvente; k largos con `interseccion`; sin dedup ni tope). Con k=2 da 12 acciones en s₀ (K4), lo que confirma el valor del Taller 4 (fijado en `test_t4_k4`). `REGLAS_T4` tiene `tol_alcance` = 1e-6 px (con k=2 los largos son los bordes de [MINLEN, MAXLEN]); los presets de Taller 2 la dejan en 0.
  - `esbeltez(pos, definicion)` = altura / `ancho_inferior`. `"apoyo"` (D1, oficial): extensión en x de los nodos con y ≤ Y_SUELO + R_BOLA + 1 px. `"tercio"` (D2): extensión en x de los nodos con y ≤ altura/3; queda en el código pero ningún experimento nuevo la usa. Con D1 el ancho casi siempre es 150 (nodos 0 y 1), así que toda acción en L(s) tiene e ≥ 4.
- `wog/fisica.py`: parámetros, espacio, bolas, uniones, tensión, `estabilizar` con ruptura.
- `wog/env.py`: `GooPhysicsEnv` (MDP del Taller 4). Acción con índice estable `((par(i,j)·k+iℓi)·k+iℓj)·2+iσ` (1224 con k=2); W = m ~ Binomial(n, p_carga=0.2, provisional) bolas en nodos dinámicos (por grado o uniforme), probada sobre una copia del espacio; altura = centro de la bola más alta; `clone_state`/`restore_state` con un solo deepcopy de (espacio, cuerpos, uniones). `info["esbeltez"]` usa D1. `predicciones()` devuelve una copia de {acción válida: P predicha}.
  - Como W nunca cambia el estado, una política determinista construye la MISMA estructura en toda seed; la seed solo decide si colapsa y cuándo.
  - La estructura de un plan tampoco depende del régimen de sensibilidad, pero SOLO porque los regímenes varían el umbral y p_carga: sin rupturas, `estabilizar` con umbral finito y con umbral = ∞ es idéntico bit a bit, así que el colapso propio en el paso t ⇔ pico sin carga en t > umbral, y con una carga dada se rompe ⇔ pico de la corrida sin ruptura > umbral. Si varían k, la masa, la rigidez o el damping, la estructura cambia y hay que buscar candidatos de nuevo (`experiments/planes.py candidatos`).
- `wog/politicas.py`: `PoliticaT4(e_max, definicion)` (reglas llegar / asegurar / avanzar / ensanchar; usa solo (G, pos, n) y las P predichas, sin física; devuelve `(a, regla)`) y `PoliticaAleatoria(seed)` sobre la máscara.
- `wog/busqueda.py`: `a_estrella` geométrico sin física (g = bolas, h = (META_Y − altura)/MAXLEN). Cierre opcional con clave = posiciones y aristas redondeadas (no fusiona grafos distintos); `cerrar=False` es exacto y da el mismo mínimo.
- `experiments/baseline.py`: subcomandos `percentiles`, `astar`, `barrido`, `sensibilidad`, `estructuras`, `todo`. Salidas en `experiments/resultados/`.
- `experiments/visual.py`: GIF de construcción (greedy, A* y los planes de `--plan`) y de un colapso del greedy (umbral ×0,75, p_carga 0,4), comparación de estructuras finales y `chequeos.md`, en `experiments/resultados/visual/`. Instrumenta con una copia exacta del bucle de `estabilizar` (verificado bit a bit contra el entorno).
- `experiments/en_vivo.py`: Thompson sampling (Beta por plan, juega argmax q̃·R) entre planes fijos (greedy, A* y los candidatos de `--plan`, JSON) en un régimen que el agente no conoce; ventana en vivo (estructura, posteriores, recompensa acumulada vs oráculo). El oráculo estima q con seeds 10000+, separadas de las del agente.
- Scripts de Taller 2 importan de `wog`. El notebook queda intacto como entregable.
- `uv run pytest`: los scripts deben reproducir `evidencia/*.txt` byte a byte y el notebook sus salidas guardadas.

## Resultados (baseline Taller 4, entorno por defecto, 200 seeds)
- A* sin física desde el K4 ideal: mínimo 5 bolas, así que la recompensa no puede pasar de 9.
- Greedy (`PoliticaT4`, e_max = ∞): 88,5% de éxito y 7,96 bolas de media.

### Nomenclatura de planes
- Referencias con nombre: solo greedy y A*. Todo otro plan se identifica por su largo y un índice (`L6-#03`), sin trato especial. Ningún reporte nuevo lleva columna Δ contra un plan que no sea greedy o A*.

### Exploratorio (no se reporta)
- D2 ("tercio") con e_max = 2,5: 100% de éxito y 6,00 bolas (8 colocadas). No se reporta porque D2 es una definición de comparación y no la oficial, y porque 2,5 se eligió como el mejor de una grilla de 10 valores muy irregular (1 → 97,5%, 1,25 → 0%, 3 → 77,5%) evaluada en las mismas 200 seeds, así que su desempeño tiene sesgo de selección. Su secuencia de 8 acciones entra en `planes.py` como un candidato L8 más, sin nombre propio. Las salidas antiguas de `baseline.py` (`tabla.md`, `sensibilidad.md`, `estructuras_finales.md`) la siguen mostrando por reproducibilidad.

## Bugs conocidos (no corregidos, a propósito)
- Sin tolerancia de alcance (`tol_alcance=0`), los presets del Taller 2 pierden acciones válidas por redondeo: `interseccion` deja `dist(P, ancla)` en 99,999… o 200,000…1 y el filtro `[MINLEN, MAXLEN]` las descarta. En la tabla k del notebook (greedy de altura, 4 pasos, k=2) el estado final tiene 22 acciones y con tolerancia serían 37 en ese mismo estado (39 si el greedy también usa tolerancia, porque cambia el camino). Se mantienen así por reproducibilidad; solo `REGLAS_T4` usa 1e-6 px.
- `fase1_paso_a_paso.py` (`REGLAS_FASE1`) no filtra el suelo: acepta la bola en (75.0, −129.9), anclada en (0,1) lado −1.
- `apice(A,B,L,σ)` e `interseccion(A,B,L,L,σ)` son el mismo punto matemático pero no bit a bit: difieren hasta ~1e-13 (≈9% de pares al azar). Ej.: A=(0,0), B=(150,150), σ=+1 → `apice` (0.0, 150.0), `interseccion` (−1.4e-14, 150.0). `REGLAS_FASE1` usa `apice` para conservar la salida.
- `se_cruzan` tiene dos modos (`estricto=False` en fase1/notebook, `True` en gym):
  - Segmentos colineales superpuestos (parcial, contenido, idénticos): **ninguno de los dos modos detecta el cruce**, ambos devuelven `False`. Con todas las orientaciones en 0, `o1 != o2` es falso.
  - Solo difieren en el contacto en "T" (extremo de un segmento en el interior del otro): permisivo `True`, estricto `False`.
  - La tolerancia de colinealidad es absoluta (1e-9 sobre el producto cruz, en px²), así que en la práctica solo atrapa colinealidad exacta.
  - **Resuelto en `REGLAS_T4`** con el filtro `holgura` (distancia ≥ R_BOLA de la bola nueva a toda unión existente y de cada unión nueva a toda bola existente que no sea su extremo). Descarta el contacto en T, los colineales superpuestos y los casi colineales. Los presets `REGLAS_FASE1`, `REGLAS_GYM_T2` y `REGLAS_NOTEBOOK` lo mantienen desactivado por compatibilidad con la evidencia del Taller 2.

## Reglas
- No cambies decisiones de diseño sin preguntar.
- El código no es fuente de verdad sobre el diseño; el MDP de arriba sí.
- Textos y comentarios en español.
