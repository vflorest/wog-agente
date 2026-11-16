"""
Baseline del Taller 4: politica determinista PoliticaT4 vs aleatoria
enmascarada en GooPhysicsEnv, mas la referencia geometrica de A*.

Uso:
    uv run python experiments/baseline.py percentiles   # e en episodios
    uv run python experiments/baseline.py astar         # cota geometrica
    uv run python experiments/baseline.py barrido       # grilla x 200 seeds
    uv run python experiments/baseline.py sensibilidad  # theta y carga
    uv run python experiments/baseline.py estructuras   # figuras finales
    uv run python experiments/baseline.py todo

Salidas en experiments/resultados/.

Convenciones:
  - Seeds 0..199; la aleatoria usa la misma seed que el entorno.
  - "bolas recolectadas" = recompensa del episodio (n si llega con n >= 4).
  - Colapso "al llegar": la accion del paso en que colapsa estaba en L(s),
    es decir, su P predicha alcanzaba META_Y.
  - Regla del colapso: la regla que eligio la accion del paso del colapso.
"""

import argparse
import copy
import csv
import json
import math
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt                           # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wog.busqueda import a_estrella                       # noqa: E402
from wog.env import NODOS_FIJOS, GooPhysicsEnv, Theta     # noqa: E402
from wog.fisica import MAXFORCE, estabilizar, tension     # noqa: E402
from wog.geometria import (                               # noqa: E402
    BOLAS_DISPONIBLES, DEFINICIONES_ESBELTEZ, META_Y, R_BOLA, Estado,
    esbeltez,
)
from wog.politicas import REGLAS, PoliticaAleatoria, PoliticaT4  # noqa: E402

RESULTADOS = Path(__file__).resolve().parent / "resultados"
SEEDS = range(200)
CAUSAS = ("meta", "meta_sin_bolas", "colapso_propio", "colapso_carga",
          "sin_bolas", "sin_acciones", "horizonte")

# Grilla de e_max por definicion (aprobada tras 'percentiles').
GRILLA = {
    "apoyo": [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, math.inf],
    "tercio": [0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 4.5, math.inf],
}
SENS_UMBRAL = (0.75, 1.0, 1.25)          # factores sobre MAXFORCE
SENS_P_CARGA = (0.2, 0.4)
SENS_POLITICAS = ["aleatoria"] + [("t4", "apoyo", e) for e in
                                   (3.0, 4.0, 4.5, math.inf)] \
    + [("t4", "tercio", e) for e in (1.0, 2.5)]
SENS_REF = "D1 e_max=∞"                    # referencia de los Δ pareados

# Paleta categorica de referencia (modo claro), en orden fijo.
C = {"azul": "#2a78d6", "naranja": "#eb6834", "aqua": "#1baf7a",
     "amarillo": "#eda100", "magenta": "#e87ba4", "verde": "#008300",
     "violeta": "#6250d6", "rojo": "#e34948", "gris": "#8a8984",
     "tinta": "#0b0b0b", "tinta2": "#52514e", "rejilla": "#e4e3df",
     "fondo": "#fcfcfb"}
COLOR_DEF = {"apoyo": C["azul"], "tercio": C["naranja"]}
NOMBRE_DEF = {"apoyo": "D1 apoyo", "tercio": "D2 tercio"}

plt.rcParams.update({
    "figure.facecolor": C["fondo"], "axes.facecolor": C["fondo"],
    "savefig.facecolor": C["fondo"], "axes.edgecolor": C["rejilla"],
    "axes.labelcolor": C["tinta2"], "xtick.color": C["tinta2"],
    "ytick.color": C["tinta2"], "text.color": C["tinta"],
    "axes.grid": True, "grid.color": C["rejilla"], "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 11, "axes.titleweight": "bold", "font.size": 9.5,
    "legend.frameon": False,
})


def fmt_e(e):
    return "∞" if math.isinf(e) else f"{e:g}"


def wilson(k, n, z=1.96):
    """Intervalo de Wilson al 95% para una proporcion k/n."""
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    den = 1 + z * z / n
    centro = (p + z * z / (2 * n)) / den
    medio = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (centro - medio, centro + medio)


# --------------------------------------------------------------------------
# Episodios
# --------------------------------------------------------------------------
def crear_politica(nombre, seed):
    """nombre: 'aleatoria' o ('t4', definicion, e_max)."""
    if nombre == "aleatoria":
        return PoliticaAleatoria(seed)
    _, definicion, e_max = nombre
    return PoliticaT4(e_max, definicion=definicion)


def episodio(args):
    """Corre un episodio y devuelve un resumen serializable."""
    nombre, seed, registrar_e, f_umbral, p_carga = args
    env = GooPhysicsEnv(theta=Theta(umbral=f_umbral * MAXFORCE),
                        p_carga=p_carga)
    env.reset(seed=seed)
    pi = crear_politica(nombre, seed)
    reglas, llegaba, alturas = [], [], []
    muestras = {d: [] for d in DEFINICIONES_ESBELTEZ}
    r_total = 0.0
    while True:
        pred = env.predicciones()
        if registrar_e:
            pts = list(env.pos.values())
            for d in DEFINICIONES_ESBELTEZ:
                muestras[d].append(
                    (esbeltez(pts, d),
                     [(esbeltez(pts + [P], d), P[1] >= META_Y)
                      for P in pred.values()]))
        a, regla = pi(env)
        _, r, term, trunc, info = env.step(a)
        r_total += r
        reglas.append(regla)
        llegaba.append(bool(pred[a][1] >= META_Y))
        alturas.append(info["altura"])
        if term or trunc:
            break
    causa = info["causa"]
    colapso = causa == "colapso"
    if colapso:
        causa = f"colapso_{info['colapso_por']}"
    return {"seed": seed, "recompensa": r_total, "causa": causa,
            "pasos": len(reglas), "reglas": reglas,
            "altura_max": max(alturas),
            "colapso_paso": len(reglas) if colapso else None,
            "colapso_al_llegar": llegaba[-1] if colapso else None,
            "colapso_regla": reglas[-1] if colapso else None,
            "muestras": muestras if registrar_e else None}


def correr(nombre, seeds=SEEDS, registrar_e=False, f_umbral=1.0,
           p_carga=0.2):
    with ProcessPoolExecutor() as ex:
        return list(ex.map(episodio,
                           [(nombre, s, registrar_e, f_umbral, p_carga)
                            for s in seeds], chunksize=4))


def resumir(eps):
    """Metricas agregadas de una lista de episodios."""
    n = len(eps)
    r = np.array([ep["recompensa"] for ep in eps])
    exitos = int((r > 0).sum())
    causas = Counter(ep["causa"] for ep in eps)
    reglas = Counter(x for ep in eps for x in ep["reglas"])
    total_pasos = sum(reglas.values())
    col = [ep for ep in eps if ep["colapso_paso"] is not None]
    return {"n": n, "exitos": exitos, "exito": exitos / n,
            "ic": wilson(exitos, n), "r_media": float(r.mean()),
            "r_desv": float(r.std(ddof=1)),
            "pasos_media": float(np.mean([ep["pasos"] for ep in eps])),
            "causas": {c: causas.get(c, 0) / n for c in CAUSAS},
            "reglas": {k: v / total_pasos for k, v in reglas.items()},
            "colapsos": len(col),
            "colapso_al_llegar": (sum(ep["colapso_al_llegar"] for ep in col)
                                  / len(col) if col else math.nan),
            "colapso_regla": Counter(ep["colapso_regla"] for ep in col),
            "colapso_paso": Counter(ep["colapso_paso"] for ep in col),
            "colapso_paso_llegar": Counter(ep["colapso_paso"] for ep in col
                                           if ep["colapso_al_llegar"])}


# --------------------------------------------------------------------------
# 1. Percentiles de e
# --------------------------------------------------------------------------
def percentiles(n_seeds=50):
    qs = [5, 25, 50, 75, 95]
    t0 = time.perf_counter()
    fuentes = {"aleatoria": correr("aleatoria", range(n_seeds), True)}
    for d in DEFINICIONES_ESBELTEZ:
        fuentes[f"t4_{d}_inf"] = correr(("t4", d, math.inf),
                                        range(n_seeds), True)
    lineas = [f"# Percentiles de e ({n_seeds} episodios por fuente, "
              f"{time.perf_counter() - t0:.0f} s)", ""]
    for d in DEFINICIONES_ESBELTEZ:
        lineas += [f"## Definicion {d!r}", "",
                   "| fuente | muestra | N | " +
                   " | ".join(f"p{q}" for q in qs) + " |",
                   "|---|---|---|" + "---|" * len(qs)]
        for f, eps in fuentes.items():
            ms = [m for ep in eps for m in ep["muestras"][d]]
            e_s = [m[0] for m in ms]
            e_a = [x for m in ms for x, _ in m[1]]
            e_l = [x for m in ms for x, llega in m[1] if llega]
            for etiqueta, v in (("e(s)", e_s), ("e(s,a)", e_a),
                                ("e(s,a), a en L(s)", e_l)):
                p = np.percentile(v, qs) if v else [math.nan] * len(qs)
                lineas.append(f"| {f} | {etiqueta} | {len(v)} | " +
                              " | ".join(f"{x:.2f}" for x in p) + " |")
        lineas.append("")
    lineas += ["## Episodios", "",
               "| fuente | exito | causas |", "|---|---|---|"]
    for f, eps in fuentes.items():
        c = Counter(ep["causa"] for ep in eps)
        lineas.append(f"| {f} | {sum(ep['recompensa'] > 0 for ep in eps)}"
                      f"/{len(eps)} | {dict(c)} |")
    escribir("percentiles_e.md", "\n".join(lineas) + "\n")


# --------------------------------------------------------------------------
# 2. Referencia geometrica (A*)
# --------------------------------------------------------------------------
def astar():
    con = a_estrella(cerrar=True)
    sin = a_estrella(cerrar=False)
    assert con.minimo == sin.minimo, "el cierre cambio el minimo"
    datos = {
        "descripcion": "A* geometrico sin fisica desde el K4 ideal "
                       "(Estado.inicial), REGLAS_T4, k=2. g = bolas, "
                       "h = (META_Y - altura) / MAXLEN.",
        "meta_y": META_Y, "k": 2,
        "minimo_bolas": con.minimo,
        "cota_recompensa": BOLAS_DISPONIBLES - con.minimo,
        "secuencia": [{"paso": t + 1, "i": i, "j": j, "Li": Li, "Lj": Lj,
                       "lado": lado, "P": list(P), "altura": h}
                      for t, ((i, j, Li, Lj, lado, P), h)
                      in enumerate(zip(con.acciones, con.alturas))],
        "con_cierre": {"expandidos": con.expandidos,
                       "generados": con.generados,
                       "segundos": round(con.segundos, 3)},
        "sin_cierre": {"minimo_bolas": sin.minimo,
                       "misma_secuencia": con.acciones == sin.acciones,
                       "expandidos": sin.expandidos,
                       "generados": sin.generados,
                       "segundos": round(sin.segundos, 3)},
    }
    RESULTADOS.mkdir(parents=True, exist_ok=True)
    (RESULTADOS / "astar_optimo.json").write_text(
        json.dumps(datos, indent=2, ensure_ascii=False) + "\n")
    figura_astar(con)
    print(json.dumps({k: v for k, v in datos.items() if k != "secuencia"},
                     indent=2, ensure_ascii=False))
    return datos


def figura_astar(res):
    e = Estado.inicial()
    estados = [e]
    for acc in res.acciones:
        e = e.aplicar(acc)
        estados.append(e)
    final, n_base = estados[-1], 4
    fig, ax = plt.subplots(figsize=(5.2, 7.2))
    ax.axhline(META_Y, color=C["verde"], lw=1.5, ls=(0, (5, 3)))
    ax.text(-130, META_Y + 10, f"meta {META_Y:.0f} px", color=C["tinta2"])
    ax.axhline(0, color=C["tinta2"], lw=2)
    for u, v in final.G.edges:
        nueva = max(u, v) >= n_base
        (x1, y1), (x2, y2) = final.pos[u], final.pos[v]
        ax.plot([x1, x2], [y1, y2], lw=2,
                color=C["azul"] if nueva else C["gris"], zorder=1)
    for n, (x, y) in final.pos.items():
        nueva = n >= n_base
        ax.add_patch(plt.Circle((x, y), R_BOLA, zorder=2,
                                color=C["azul"] if nueva else C["gris"],
                                ec=C["fondo"], lw=2))
        if nueva:
            ax.text(x, y, str(n - n_base + 1), ha="center", va="center",
                    color="white", fontsize=9, fontweight="bold", zorder=3)
            ax.text(x + 22, y, f"{y:.0f}", va="center", color=C["tinta2"],
                    fontsize=8)
    ax.set_aspect("equal")
    ax.set_xlim(-160, 330)
    ax.set_ylim(-20, 680)
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.set_title(f"A* sin fisica: {res.minimo} bolas "
                 f"(cota de recompensa {BOLAS_DISPONIBLES - res.minimo})",
                 loc="left")
    ax.text(0.0, -0.11, "gris: K4 ideal; azul: bolas en orden de "
            "colocacion (y del centro a la derecha)", transform=ax.transAxes,
            color=C["tinta2"], fontsize=8)
    guardar(fig, "astar_optimo.png")


# --------------------------------------------------------------------------
# 3. Barrido principal
# --------------------------------------------------------------------------
def barrido():
    t0 = time.perf_counter()
    filas = [("aleatoria", "-", math.nan, resumir(correr("aleatoria")))]
    for d, grilla in GRILLA.items():
        for e_max in grilla:
            filas.append(("t4", d, e_max, resumir(correr(("t4", d, e_max)))))
    print(f"barrido: {time.perf_counter() - t0:.0f} s")

    # tabla CSV completa
    cab = (["politica", "definicion", "e_max", "n", "exitos", "exito",
            "ic_inf", "ic_sup", "r_media", "r_desv", "pasos_media",
            "colapsos", "colapso_al_llegar"] + [f"causa_{c}" for c in CAUSAS]
           + [f"regla_{r}" for r in REGLAS])
    RESULTADOS.mkdir(parents=True, exist_ok=True)
    with open(RESULTADOS / "tabla.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cab)
        for pol, d, e_max, s in filas:
            w.writerow([pol, d, fmt_e(e_max) if pol == "t4" else "-",
                        s["n"], s["exitos"], f"{s['exito']:.4f}",
                        f"{s['ic'][0]:.4f}", f"{s['ic'][1]:.4f}",
                        f"{s['r_media']:.4f}", f"{s['r_desv']:.4f}",
                        f"{s['pasos_media']:.3f}", s["colapsos"],
                        f"{s['colapso_al_llegar']:.4f}"]
                       + [f"{s['causas'][c]:.4f}" for c in CAUSAS]
                       + [f"{s['reglas'].get(r, 0):.4f}" for r in REGLAS])

    # mejor configuracion por definicion: mayor recompensa media,
    # desempate por exito y luego por menor e_max
    mejores = {}
    for d in GRILLA:
        cand = [f for f in filas if f[1] == d]
        mejores[d] = max(cand, key=lambda f: (f[3]["r_media"],
                                              f[3]["exito"], -f[2]))

    md = ["# Baseline Taller 4: tabla principal", "",
          f"200 seeds por configuracion; GooPhysicsEnv por defecto "
          f"(k=2, p_carga=0.2, umbral={MAXFORCE:g}). Cota geometrica de "
          f"la recompensa (A*): ver astar_optimo.json.", "",
          "| politica | def | e_max | exito [IC95 Wilson] | bolas μ ± σ "
          "| pasos | meta | meta s/bolas | col. propio | col. carga "
          "| sin bolas | sin acc. | horiz. | % col. al llegar "
          "| llegar | asegurar | avanzar | ensanchar |",
          "|---|---|---|---|---|---|" + "---|" * 7 + "---|" + "---|" * 4]
    for pol, d, e_max, s in filas:
        md.append(
            f"| {pol} | {NOMBRE_DEF.get(d, '-')} | "
            f"{fmt_e(e_max) if pol == 't4' else '-'} | "
            f"{100 * s['exito']:.1f}% [{100 * s['ic'][0]:.1f}, "
            f"{100 * s['ic'][1]:.1f}] | {s['r_media']:.2f} ± "
            f"{s['r_desv']:.2f} | {s['pasos_media']:.1f} | "
            + " | ".join(f"{100 * s['causas'][c]:.0f}%" for c in CAUSAS)
            + f" | {pct(s['colapso_al_llegar'])} | "
            + " | ".join(f"{100 * s['reglas'].get(r, 0):.0f}%"
                         for r in REGLAS) + " |")
    md += ["", "Causas en % de episodios; reglas en % de pasos.", ""]

    md += ["## Mejor configuracion por definicion "
           "(mayor recompensa media)", "",
           "| def | e_max | bolas μ | exito | colapsos | "
           "en llegar | en asegurar | en avanzar | en ensanchar | "
           "al llegar a la meta |", "|---|---|---|---|---|---|---|---|---|---|"]
    for d, (_, _, e_max, s) in mejores.items():
        cr = s["colapso_regla"]
        nc = s["colapsos"]
        md.append(f"| {NOMBRE_DEF[d]} | {fmt_e(e_max)} | {s['r_media']:.2f} "
                  f"| {100 * s['exito']:.1f}% | {nc} | "
                  + " | ".join(pct(cr.get(r, 0) / nc if nc else math.nan)
                               for r in REGLAS)
                  + f" | {pct(s['colapso_al_llegar'])} |")
    md.append("")

    # distribucion de colapsos por paso
    sel = [filas[0]] + [f for f in filas if f[1] == "apoyo"
                        and math.isinf(f[2])] + list(mejores.values())
    sel = list({(f[0], f[1], f[2]): f for f in sel}.values())
    md += ["## Colapsos por paso", "",
           "Cada celda: colapsos en ese paso (de ellos, al llegar a la "
           "meta).", "",
           "| config | " + " | ".join(str(p) for p in range(1, 15))
           + " | total |", "|---|" + "---|" * 15]
    for f in sel:
        s = f[3]
        celdas = []
        for p in range(1, 15):
            c, cl = s["colapso_paso"].get(p, 0), s["colapso_paso_llegar"].get(p, 0)
            celdas.append("·" if c == 0 else (f"{c} ({cl})" if cl else str(c)))
        md.append(f"| {etiqueta(f)} | " + " | ".join(celdas)
                  + f" | {s['colapsos']} |")
    md.append("")
    escribir("tabla.md", "\n".join(md) + "\n")

    figura_barrido(filas)
    figura_causas_reglas(filas)
    figura_colapsos(sel)
    return filas, mejores


def pct(x):
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) \
        else f"{100 * x:.0f}%"


def etiqueta(f):
    pol, d, e_max, _ = f
    return "aleatoria" if pol == "aleatoria" else \
        f"{NOMBRE_DEF[d]}, e_max={fmt_e(e_max)}"


def figura_barrido(filas):
    alea = filas[0][3]
    cota = BOLAS_DISPONIBLES - a_estrella_minimo()
    fig, axs = plt.subplots(2, 2, figsize=(11, 7), sharey="row")
    for col, d in enumerate(GRILLA):
        fs = [f for f in filas if f[1] == d]
        xs = list(range(len(fs)))            # eje categorico
        ticks = [fmt_e(f[2]) for f in fs]
        color = COLOR_DEF[d]
        ex = np.array([f[3]["exito"] for f in fs]) * 100
        lo = ex - np.array([f[3]["ic"][0] for f in fs]) * 100
        hi = np.array([f[3]["ic"][1] for f in fs]) * 100 - ex
        ax = axs[0, col]
        ax.errorbar(xs, ex, yerr=[lo, hi], color=color, lw=2, marker="o",
                    ms=6, capsize=3, mec=C["fondo"], mew=1.5)
        ax.axhline(alea["exito"] * 100, color=C["gris"], ls=(0, (4, 3)))
        ax.text(xs[0], alea["exito"] * 100 + 2, "aleatoria",
                color=C["tinta2"], fontsize=8)
        ax.set_title(f"{NOMBRE_DEF[d]}: tasa de exito (IC 95% Wilson)",
                     loc="left")
        ax.set_ylabel("exito (%)")
        ax.set_ylim(-3, 103)
        ax = axs[1, col]
        r = np.array([f[3]["r_media"] for f in fs])
        sem = np.array([f[3]["r_desv"] / math.sqrt(f[3]["n"]) for f in fs])
        ax.errorbar(xs, r, yerr=1.96 * sem, color=color, lw=2, marker="o",
                    ms=6, capsize=3, mec=C["fondo"], mew=1.5)
        ax.axhline(cota, color=C["verde"], ls=(0, (5, 3)))
        ax.text(xs[0], cota + 0.2, f"cota A* = {cota}", color=C["tinta2"],
                fontsize=8)
        ax.axhline(alea["r_media"], color=C["gris"], ls=(0, (4, 3)))
        ax.set_title(f"{NOMBRE_DEF[d]}: bolas recolectadas (media, IC 95%)",
                     loc="left")
        ax.set_ylabel("bolas")
        ax.set_ylim(-0.3, cota + 1)
        for a in axs[:, col]:
            a.set_xticks(xs, ticks)
            a.set_xlabel("e_max")
    fig.tight_layout()
    guardar(fig, "exito_vs_emax.png")


_MINIMO = {}


def a_estrella_minimo():
    if "m" not in _MINIMO:
        p = RESULTADOS / "astar_optimo.json"
        _MINIMO["m"] = (json.loads(p.read_text())["minimo_bolas"]
                        if p.exists() else a_estrella().minimo)
    return _MINIMO["m"]


def barras_apiladas(ax, filas, claves, valores, colores, titulo):
    y = np.arange(len(filas))[::-1]
    izq = np.zeros(len(filas))
    for c, col in zip(claves, colores):
        v = np.array([valores(f)[c] for f in filas]) * 100
        ax.barh(y, v, left=izq, color=col, edgecolor=C["fondo"],
                linewidth=1.5, height=0.72, label=c)
        izq += v
    ax.set_yticks(y, [etiqueta(f) for f in filas], fontsize=8)
    ax.set_xlim(0, 100)
    ax.set_xlabel("%")
    ax.grid(axis="y", visible=False)
    ax.set_title(titulo, loc="left")
    ax.legend(ncol=4, loc="upper left", bbox_to_anchor=(0, -0.08),
              fontsize=8)


def figura_causas_reglas(filas):
    causas = [c for c in CAUSAS if any(f[3]["causas"][c] for f in filas)]
    col_causa = {"meta": C["aqua"], "meta_sin_bolas": C["amarillo"],
                 "colapso_propio": C["rojo"], "colapso_carga": C["naranja"],
                 "sin_bolas": C["violeta"], "sin_acciones": C["magenta"],
                 "horizonte": C["gris"]}
    fig, ax = plt.subplots(figsize=(9, 8.5))
    barras_apiladas(ax, filas, causas, lambda f: f[3]["causas"],
                    [col_causa[c] for c in causas],
                    "Causa de termino (% de episodios)")
    fig.tight_layout()
    guardar(fig, "causas.png")

    t4 = [f for f in filas if f[0] == "t4"]
    col_regla = [C["azul"], C["naranja"], C["aqua"], C["amarillo"]]
    fig, ax = plt.subplots(figsize=(9, 8))
    barras_apiladas(ax, t4, list(REGLAS),
                    lambda f: {r: f[3]["reglas"].get(r, 0) for r in REGLAS},
                    col_regla, "Regla disparada (% de pasos)")
    fig.tight_layout()
    guardar(fig, "reglas.png")


def figura_colapsos(sel):
    fig, axs = plt.subplots(1, len(sel), figsize=(3.3 * len(sel), 3.4),
                            sharey=True)
    pasos = np.arange(1, 15)
    for ax, f in zip(np.atleast_1d(axs), sel):
        s = f[3]
        llegar = np.array([s["colapso_paso_llegar"].get(p, 0) for p in pasos])
        total = np.array([s["colapso_paso"].get(p, 0) for p in pasos])
        ax.bar(pasos, total - llegar, color=C["naranja"], width=0.8,
               edgecolor=C["fondo"], label="antes de llegar")
        ax.bar(pasos, llegar, bottom=total - llegar, color=C["azul"],
               width=0.8, edgecolor=C["fondo"], label="al llegar a la meta")
        ax.set_title(etiqueta(f), loc="left", fontsize=9)
        ax.set_xticks(pasos[::2])
        ax.set_xlabel("paso del colapso")
        ax.grid(axis="x", visible=False)
    np.atleast_1d(axs)[0].set_ylabel("colapsos (de 200 episodios)")
    np.atleast_1d(axs)[0].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    guardar(fig, "colapsos_por_paso.png")


# --------------------------------------------------------------------------
# 4. Sensibilidad a theta y a la carga
# --------------------------------------------------------------------------
def sensibilidad(n_boot=5000):
    t0 = time.perf_counter()
    filas = []
    for f_u in SENS_UMBRAL:
        for p in SENS_P_CARGA:
            eps = {nombre_sens(pol): correr(pol, f_umbral=f_u, p_carga=p)
                   for pol in SENS_POLITICAS}
            r_ref = np.array([ep["recompensa"] for ep in eps[SENS_REF]])
            for k, (clave, lista) in enumerate(eps.items()):
                s = resumir(lista)
                # diferencia pareada por seed contra D1 e_max = inf, con IC
                # bootstrap; generador propio por celda (reproducible)
                rng = np.random.default_rng([int(100 * f_u), int(10 * p), k])
                d = np.array([ep["recompensa"] for ep in lista]) - r_ref
                bs = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(axis=1)
                filas.append((f_u, p, clave, s, float(d.mean()),
                              tuple(np.percentile(bs, [2.5, 97.5]))))
    print(f"sensibilidad: {time.perf_counter() - t0:.0f} s")

    md = ["# Sensibilidad a theta (umbral) y a la carga (p_carga)", "",
          f"200 seeds por celda. umbral = factor x {MAXFORCE:g}. "
          f"Δ bolas = diferencia pareada por seed contra {SENS_REF} "
          "(IC 95% bootstrap). ★: IC de Δ entero sobre 0; ▼: entero "
          "bajo 0.", "",
          "| umbral | p_carga | politica | exito [IC95 Wilson] | bolas μ "
          f"| Δ bolas vs {SENS_REF} [IC95] | col. propio | col. carga "
          "| meta s/bolas |",
          "|---|---|---|---|---|---|---|---|---|"]
    for f_u, p, clave, s, dm, (lo, hi) in filas:
        marca = " ★" if lo > 0 else (" ▼" if hi < 0 else "")
        delta = "-" if clave == SENS_REF else \
            f"{dm:+.2f} [{lo:+.2f}, {hi:+.2f}]{marca}"
        md.append(f"| ×{f_u:g} | {p:g} | {clave} | {100 * s['exito']:.1f}% "
                  f"[{100 * s['ic'][0]:.1f}, {100 * s['ic'][1]:.1f}] | "
                  f"{s['r_media']:.2f} | {delta} | "
                  f"{100 * s['causas']['colapso_propio']:.0f}% | "
                  f"{100 * s['causas']['colapso_carga']:.0f}% | "
                  f"{100 * s['causas']['meta_sin_bolas']:.0f}% |")

    def respuesta(pregunta, filtro):
        md.extend(["", f"## {pregunta}", ""])
        sel = [f for f in filas if filtro(f[2])]
        for f_u, p, clave, s, dm, (lo, hi) in sel:
            veredicto = ("supera" if lo > 0 else "peor" if hi < 0
                         else "sin diferencia significativa")
            md.append(f"- umbral ×{f_u:g}, p_carga {p:g}, {clave}: "
                      f"Δ = {dm:+.2f} [{lo:+.2f}, {hi:+.2f}] -> {veredicto}")

    respuesta(f"¿Alguna e_max finita de D1 supera a {SENS_REF}?",
              lambda c: c.startswith("D1") and c != SENS_REF)
    respuesta(f"¿En que regimenes supera D2 e_max=2.5 a {SENS_REF}?",
              lambda c: c == "D2 e_max=2.5")
    md.append("")
    escribir("sensibilidad.md", "\n".join(md) + "\n")
    with open(RESULTADOS / "sensibilidad.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["umbral_factor", "p_carga", "politica", "n", "exitos",
                    "exito", "ic_inf", "ic_sup", "r_media", "r_desv",
                    "delta_vs_ref", "delta_ic_inf", "delta_ic_sup"]
                   + [f"causa_{c}" for c in CAUSAS])
        for f_u, p, clave, s, dm, (lo, hi) in filas:
            w.writerow([f_u, p, clave, s["n"], s["exitos"],
                        f"{s['exito']:.4f}", f"{s['ic'][0]:.4f}",
                        f"{s['ic'][1]:.4f}", f"{s['r_media']:.4f}",
                        f"{s['r_desv']:.4f}", f"{dm:.4f}", f"{lo:.4f}",
                        f"{hi:.4f}"]
                       + [f"{s['causas'][c]:.4f}" for c in CAUSAS])
    figura_sensibilidad(filas)
    return filas


def nombre_sens(pol):
    if pol == "aleatoria":
        return "aleatoria"
    _, d, e = pol
    return f"{'D1' if d == 'apoyo' else 'D2'} e_max={fmt_e(e)}"


def figura_sensibilidad(filas):
    cats = [nombre_sens(p) for p in SENS_POLITICAS]
    xs = np.arange(len(cats))
    colores = [C["azul"], C["naranja"], C["aqua"]]
    fig, axs = plt.subplots(2, len(SENS_P_CARGA), figsize=(12, 7.5),
                            sharey="row", sharex=True)
    for col, p in enumerate(SENS_P_CARGA):
        for k, f_u in enumerate(SENS_UMBRAL):
            fs = {f[2]: f[3] for f in filas if f[0] == f_u and f[1] == p}
            off = (k - 1) * 0.14
            ex = np.array([fs[c]["exito"] for c in cats]) * 100
            lo = ex - np.array([fs[c]["ic"][0] for c in cats]) * 100
            hi = np.array([fs[c]["ic"][1] for c in cats]) * 100 - ex
            axs[0, col].errorbar(xs + off, ex, yerr=[lo, hi], lw=0,
                                 elinewidth=2, marker="o", ms=7, capsize=3,
                                 color=colores[k], mec=C["fondo"], mew=1.5,
                                 label=f"umbral ×{f_u:g}")
            r = [fs[c]["r_media"] for c in cats]
            sem = [1.96 * fs[c]["r_desv"] / math.sqrt(fs[c]["n"])
                   for c in cats]
            axs[1, col].errorbar(xs + off, r, yerr=sem, lw=0, elinewidth=2,
                                 marker="o", ms=7, capsize=3,
                                 color=colores[k], mec=C["fondo"], mew=1.5)
        for fila in (0, 1):
            # separa la familia D2 de la D1
            axs[fila, col].axvline(len(cats) - 2.5, color=C["tinta2"],
                                   lw=0.8, ls=(0, (2, 3)))
        axs[0, col].set_title(f"p_carga = {p:g}: exito (IC 95% Wilson)",
                              loc="left")
        axs[1, col].set_title(f"p_carga = {p:g}: bolas (media, IC 95%)",
                              loc="left")
        axs[1, col].set_xticks(xs, [c.replace(" e_max=", "\n")
                                    for c in cats], fontsize=8)
        axs[1, col].set_xlabel("politica (definicion y e_max)")
        axs[0, col].grid(axis="x", visible=False)
        axs[1, col].grid(axis="x", visible=False)
    axs[0, 0].set_ylabel("exito (%)")
    axs[1, 0].set_ylabel("bolas")
    axs[0, 0].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    guardar(fig, "sensibilidad.png")


# --------------------------------------------------------------------------
# 5. Estructuras finales (la trayectoria no depende de la seed)
# --------------------------------------------------------------------------
def trayectoria_nominal(pol):
    """
    Corre la politica sin carga. La carga W se prueba sobre una copia del
    espacio y nunca cambia el estado, asi que una politica determinista
    construye la MISMA estructura en toda seed; la seed solo decide si (y
    cuando) colapsa. Devuelve (env final, [(accion, regla), ...]).
    """
    env = GooPhysicsEnv(p_carga=0.0)
    env.reset(seed=0)
    pi = crear_politica(pol, 0)
    pasos = []
    while True:
        a, regla = pi(env)
        _, _, term, _, _ = env.step(a)
        pasos.append((env.tabla[a], regla))
        if term:
            return env, pasos


def fragilidad(env, m, n_sorteos=300, seed=1):
    """P(rotura) de la estructura con m bolas de carga repartidas por grado."""
    rng = np.random.default_rng(seed)
    dyn = [n for n in env.G.nodes if n not in NODOS_FIJOS]
    w = np.array([env.G.degree(n) for n in dyn], float)
    rotas = 0
    for _ in range(n_sorteos):
        sp, cu, un = copy.deepcopy((env.space, env.cuerpos, env.uniones))
        for n in rng.choice(dyn, size=m, p=w / w.sum()):
            cu[int(n)].mass += env.theta.masa
        rotas += bool(estabilizar(sp, un, env.theta.umbral).rotas)
    return rotas / n_sorteos


def estructuras():
    casos = [(("t4", "apoyo", math.inf), "greedy: D1 e_max=∞"),
             (("t4", "tercio", 2.5), "D2 e_max=2.5")]
    datos, lineas = [], ["# Estructuras finales (trayectoria nominal)", "",
                         "| politica | bolas usadas | restantes | tension "
                         "max / umbral | P(rotura) m=1 | m=2 | m=3 | m=4 "
                         "| E[m] al llegar |", "|---|---|---|---|---|---|"
                         "---|---|---|"]
    for pol, titulo in casos:
        env, pasos = trayectoria_nominal(pol)
        cuerpo_a_nodo = {id(c): n for n, c in env.cuerpos.items()}
        tens = {frozenset((cuerpo_a_nodo[id(j.a)], cuerpo_a_nodo[id(j.b)])):
                tension(j) / env.theta.umbral for j in env.uniones}
        frag = [fragilidad(env, m) for m in (1, 2, 3, 4)]
        datos.append((env, pasos, tens, titulo))
        lineas.append(f"| {titulo} | {len(pasos)} | {env.restantes} | "
                      f"{max(tens.values()):.3f} | "
                      + " | ".join(f"{100 * f:.1f}%" for f in frag)
                      + f" | {0.2 * env.restantes:.1f} |")
    lineas += ["", "P(rotura | m): 300 sorteos de m bolas de carga "
               "repartidas por grado sobre la estructura final. "
               "E[m] = p_carga x restantes (p_carga = 0.2).", ""]
    for env, pasos, _, titulo in datos:
        lineas += [f"## {titulo}", "", "| paso | accion (i, j, Li, Lj, lado) "
                   "| regla | posicion final (px) |", "|---|---|---|---|"]
        for t, (acc, regla) in enumerate(pasos):
            x, y = env.pos[4 + t]
            lineas.append(f"| {t + 1} | {acc} | {regla} | "
                          f"({x:.0f}, {y:.0f}) |")
        lineas.append("")
    escribir("estructuras_finales.md", "\n".join(lineas))
    figura_estructuras(datos)


def figura_estructuras(datos):
    vmax = max(max(t.values()) for _, _, t, _ in datos)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "tension", ["#cfe0f6", "#2a78d6", "#0d2f63"])
    norm = matplotlib.colors.Normalize(0, vmax)
    fig, axs = plt.subplots(1, 2, figsize=(11, 6.0), sharey=True)
    for ax, (env, pasos, tens, titulo) in zip(axs, datos):
        ax.axhline(META_Y, color=C["verde"], lw=1.5, ls=(0, (5, 3)))
        ax.axhline(0, color=C["tinta2"], lw=2)
        for (u, v), d in env.G.edges.items():
            (x1, y1), (x2, y2) = env.pos[u], env.pos[v]
            t = tens.get(frozenset((u, v)))
            ax.plot([x1, x2], [y1, y2], lw=2.5, zorder=1,
                    color=C["gris"] if t is None else cmap(norm(t)))
        for n, (x, y) in env.pos.items():
            base = n < 4
            regla = None if base else pasos[n - 4][1]
            color = (C["gris"] if base else
                     C["naranja"] if regla != "avanzar" and regla != "llegar"
                     else C["tinta"])
            ax.add_patch(plt.Circle((x, y), R_BOLA, zorder=2, color=color,
                                    ec=C["fondo"], lw=2))
            if not base:
                ax.text(x, y, str(n - 3), ha="center", va="center",
                        color="white", fontsize=7, fontweight="bold",
                        zorder=3)
        ax.set_aspect("equal")
        ax.set_xlim(-260, 420)
        ax.set_ylim(-20, 680)
        ax.set_xlabel("x (px)")
        ax.set_title(f"{titulo}: {len(pasos)} bolas, "
                     f"{env.restantes} restantes", loc="left")
        ax.grid(False)
    axs[0].set_ylabel("y (px)")
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=axs, shrink=0.6, pad=0.02)
    cb.set_label("tension de la union / umbral (sin carga)")
    cb.outline.set_visible(False)
    axs[0].text(0.0, -0.2, "numeros: orden de colocacion; negro: avanzar o "
             "llegar; naranja: asegurar; gris: K4 asentado (union entre "
             "fijos sin tension fisica)", color=C["tinta2"], fontsize=8.5,
                transform=axs[0].transAxes)
    guardar(fig, "estructuras_finales.png")


# --------------------------------------------------------------------------
# Utilidades
# --------------------------------------------------------------------------
def escribir(nombre, texto):
    RESULTADOS.mkdir(parents=True, exist_ok=True)
    (RESULTADOS / nombre).write_text(texto)
    print(texto)


def guardar(fig, nombre):
    RESULTADOS.mkdir(parents=True, exist_ok=True)
    fig.savefig(RESULTADOS / nombre, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("tarea", choices=["percentiles", "astar", "barrido",
                                      "sensibilidad", "estructuras", "todo"])
    args = ap.parse_args()
    t = args.tarea
    if t in ("percentiles", "todo"):
        percentiles()
    if t in ("astar", "todo"):
        astar()
    if t in ("barrido", "todo"):
        barrido()
    if t in ("sensibilidad", "todo"):
        sensibilidad()
    if t in ("estructuras", "todo"):
        estructuras()


if __name__ == "__main__":
    main()
