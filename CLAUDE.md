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
- Parámetros derivados de balls.xml en `prueba_pymunk.py` (escala 1 u = 10 px, umbral de ruptura = 12 pesos de bola).
- La construcción SIEMPRE es incremental: construir todo de una vez produce picos de tensión falsos.

## Stack
uv, Python 3.12, networkx, pymunk, gymnasium. macOS Apple Silicon.

## Código
- `wog/geometria.py`: primitivas, `Estado` y un único `generar_acciones` con filtros según un objeto `Reglas`. Presets: `REGLAS_FASE1`, `REGLAS_NOTEBOOK`, `REGLAS_GYM_T2` (reproducen Taller 2) y `REGLAS_T4` (entorno futuro: suelo, alcance, cruce permisivo, holgura, envolvente; k largos con `interseccion`; sin dedup ni tope).
- `wog/fisica.py`: parámetros, espacio, bolas, uniones, tensión, `estabilizar` con ruptura.
- `wog/env.py`: pendiente (MDP del Taller 4).
- Scripts de Taller 2 importan de `wog`. El notebook queda intacto como entregable.
- `uv run pytest`: los scripts deben reproducir `evidencia/*.txt` byte a byte y el notebook sus salidas guardadas.

## Bugs conocidos (no corregidos, a propósito)
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
