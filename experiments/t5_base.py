"""
Taller 5: evaluacion del agente base en dos condiciones.

Uso (un solo comando reproduce todo):
    uv run python experiments/t5_base.py

Agente base: PoliticaT4 con e_max = inf (greedy), definicion D1 ("apoyo").
Parametros fijos: k = 2, REGLAS_T4 (las del entorno), theta del regimen.

Condiciones:
  - referencia: theta por defecto, p_carga 0,2;
  - adversa:    umbral de ruptura x0,75, p_carga 0,4.
Semillas 0..199 por condicion (la seed del entorno, registrada por episodio).

Referencia adicional (NO es un agente): el plan de A* geometrico
(experiments/resultados/astar_optimo.json) reproducido accion por accion en
las mismas condiciones y semillas, para mostrar si existe un plan con mayor
supervivencia.

Metricas:
  - principal: bolas recolectadas por episodio (recompensa; el fracaso
    cuenta 0): media +- desviacion e IC 95% bootstrap de la media;
  - secundarias: tasa de exito (IC de Wilson), tasa de colapso (propio y
    por carga), pasos por episodio.

--meta {altura,boca}: reservado. Por ahora solo "altura" (la meta es que la
bola mas alta alcance META_Y).

Salidas en experiments/resultados/: t5_base.csv (una fila por episodio) y
t5_base.md (tabla resumen).
"""

import argparse
import csv
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baseline import wilson                               # noqa: E402
from visual import plan_astar                             # noqa: E402
from wog.env import GooPhysicsEnv, Theta                  # noqa: E402
from wog.fisica import MAXFORCE                           # noqa: E402
from wog.politicas import PoliticaT4                      # noqa: E402

RESULTADOS = Path(__file__).resolve().parent / "resultados"
SEEDS = range(200)
K = 2
CONDICIONES = {"referencia": (1.0, 0.2),      # (factor del umbral, p_carga)
               "adversa": (0.75, 0.4)}
AGENTES = {"greedy": "greedy (agente base)",
           "astar": "A* (referencia, no agente)"}
CAMPOS = ["agente", "condicion", "umbral_factor", "p_carga", "meta", "seed",
          "bolas", "exito", "colapso", "colapso_por", "causa", "pasos",
          "altura_max"]


# --------------------------------------------------------------------------
# Episodios
# --------------------------------------------------------------------------
def episodio(args):
    """Un episodio de 'agente' en 'condicion' con la seed dada."""
    agente, condicion, seed, plan, meta = args
    f_umbral, p_carga = CONDICIONES[condicion]
    env = GooPhysicsEnv(k=K, theta=Theta(umbral=f_umbral * MAXFORCE),
                        p_carga=p_carga)
    env.reset(seed=seed)
    pi = PoliticaT4(math.inf, definicion="apoyo")
    bolas, pasos, altura_max = 0.0, 0, 0.0
    while True:
        if agente == "greedy":
            a, _ = pi(env)
        else:
            if pasos >= len(plan):
                raise RuntimeError("el plan de A* no termina el episodio")
            a = env.codificar(*plan[pasos])
        _, r, term, trunc, info = env.step(a)
        bolas += r
        pasos += 1
        altura_max = max(altura_max, info["altura"])
        if term or trunc:
            break
    colapso = info["causa"] == "colapso"
    return {"agente": agente, "condicion": condicion,
            "umbral_factor": f_umbral, "p_carga": p_carga, "meta": meta,
            "seed": seed, "bolas": bolas, "exito": int(bolas > 0),
            "colapso": int(colapso),
            "colapso_por": info["colapso_por"] if colapso else "",
            "causa": info["causa"], "pasos": pasos,
            "altura_max": round(altura_max, 4)}


def correr(meta):
    plan = plan_astar()
    tareas = [(ag, cond, s, plan, meta) for ag in AGENTES
              for cond in CONDICIONES for s in SEEDS]
    with ProcessPoolExecutor() as ex:
        return list(ex.map(episodio, tareas, chunksize=4))


# --------------------------------------------------------------------------
# Resumen
# --------------------------------------------------------------------------
def resumir(filas, semilla, n_boot):
    n = len(filas)
    b = np.array([f["bolas"] for f in filas])
    p = np.array([f["pasos"] for f in filas])
    rng = np.random.default_rng(semilla)
    medias = b[rng.integers(0, n, (n_boot, n))].mean(axis=1)
    exitos = sum(f["exito"] for f in filas)
    return {"n": n, "b_media": b.mean(), "b_desv": b.std(ddof=1),
            "b_ic": tuple(np.percentile(medias, [2.5, 97.5])),
            "exitos": exitos, "exito": exitos / n,
            "exito_ic": wilson(exitos, n),
            "colapso": sum(f["colapso"] for f in filas) / n,
            "col_propio": sum(f["colapso_por"] == "propio" for f in filas) / n,
            "col_carga": sum(f["colapso_por"] == "carga" for f in filas) / n,
            "p_media": p.mean(), "p_desv": p.std(ddof=1)}


def tabla(filas, meta, n_boot):
    md = ["# Taller 5: agente base en dos condiciones", "",
          f"Agente base: PoliticaT4 e_max = ∞ (greedy), D1, k = {K}, "
          f"REGLAS_T4. Meta: {meta}. Semillas {SEEDS.start}–{SEEDS.stop - 1} "
          "por condición.",
          "Condiciones: referencia (θ por defecto, p_carga 0,2) y adversa "
          "(umbral ×0,75, p_carga 0,4).",
          "A*: plan geométrico reproducido en las mismas condiciones y "
          "semillas; es una referencia, no un agente.",
          f"Bolas: el fracaso cuenta 0; IC 95% bootstrap de la media "
          f"({n_boot} remuestreos). Éxito: IC 95% de Wilson.", "",
          "| condición | política | bolas μ ± σ | IC95 bolas | éxito "
          "[IC95] | colapso (propio / carga) | pasos μ ± σ |",
          "|---|---|---|---|---|---|---|"]
    resumenes = {}
    for ic, cond in enumerate(CONDICIONES):
        for ia, ag in enumerate(AGENTES):
            sel = [f for f in filas
                   if f["agente"] == ag and f["condicion"] == cond]
            s = resumir(sel, [5, ic, ia], n_boot)
            resumenes[(cond, ag)] = s
            md.append(
                f"| {cond} | {AGENTES[ag]} | {s['b_media']:.2f} ± "
                f"{s['b_desv']:.2f} | [{s['b_ic'][0]:.2f}, {s['b_ic'][1]:.2f}]"
                f" | {100 * s['exito']:.1f}% [{100 * s['exito_ic'][0]:.1f}, "
                f"{100 * s['exito_ic'][1]:.1f}] | {100 * s['colapso']:.1f}% "
                f"({100 * s['col_propio']:.1f} / {100 * s['col_carga']:.1f}) "
                f"| {s['p_media']:.2f} ± {s['p_desv']:.2f} |")
    md.append("")
    return "\n".join(md), resumenes


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--meta", choices=("altura", "boca"), default="altura",
                    help="criterio de meta (reservado; por ahora solo altura)")
    ap.add_argument("--n-boot", type=int, default=5000)
    args = ap.parse_args()
    if args.meta != "altura":
        ap.error(f"--meta {args.meta}: reservado, aún no implementado")

    filas = correr(args.meta)
    RESULTADOS.mkdir(exist_ok=True)
    with open(RESULTADOS / "t5_base.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CAMPOS)
        w.writeheader()
        w.writerows(filas)
    md, _ = tabla(filas, args.meta, args.n_boot)
    (RESULTADOS / "t5_base.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
