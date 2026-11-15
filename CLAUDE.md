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

## Reglas
- No cambies decisiones de diseño sin preguntar.
- El código no es fuente de verdad sobre el diseño; el MDP de arriba sí.
- Textos y comentarios en español.
