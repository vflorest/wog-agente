# wog-agente

Agente que juega el nivel 1 de World of Goo (proyecto individual, Agentes Inteligentes, UTFSM).
Base K4, 14 bolas disponibles, meta a 600 px; para pasar deben quedar al menos 4 bolas sin usar.
Física con Pymunk y ruptura por tensión; el MDP está descrito en `CLAUDE.md`.

## Instalación

Requiere [uv](https://docs.astral.sh/uv/) (Python 3.12).

```bash
uv sync
```

## Tests

```bash
uv run pytest
```

Los tests verifican, entre otras cosas, que los scripts del Taller 2 reproduzcan `evidencia/*.txt` byte a byte y que el notebook reproduzca sus salidas guardadas.

## Reproducir el Taller 5

```bash
uv run python experiments/t5_base.py
```

Evalúa el agente base (`PoliticaT4` con e_max = ∞, greedy, D1, k = 2, `REGLAS_T4`) en dos condiciones, referencia (θ por defecto, p_carga 0,2) y adversa (umbral ×0,75, p_carga 0,4), con las semillas 0–199 en cada una. Como referencia adicional (no es un agente) reproduce el plan de A* en las mismas condiciones y semillas. Escribe:

- `experiments/resultados/t5_base.csv`: una fila por episodio.
- `experiments/resultados/t5_base.md`: tabla resumen (bolas μ ± σ con IC95 bootstrap, éxito con IC de Wilson, colapsos y pasos).

`--meta {altura,boca}` queda reservado; por ahora solo se admite `altura`.

## Otros experimentos

- `experiments/baseline.py`: baseline del Taller 4 (`percentiles`, `astar`, `barrido`, `sensibilidad`, `estructuras`, `todo`).
- `experiments/visual.py`: GIF de construcción y colapso, y comparación de estructuras.
- `experiments/en_vivo.py`: Thompson sampling entre planes fijos con una ventana en vivo.
- `experiments/meta_real.py`: chequeo geométrico de la meta "boca" del nivel real (GoingUp).

## Estructura

- `wog/`: geometría, física, entorno Gymnasium (`GooPhysicsEnv`), políticas y A*.
- `experiments/`: scripts de experimentos; salidas en `experiments/resultados/`.
- `tests/`: pruebas con pytest.
- `Fase1_World_of_Goo.ipynb`, `evidencia/`: entregables del Taller 2.
