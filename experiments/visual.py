"""
Revision visual de la fisica: animaciones GIF y una figura comparativa.

Uso:
    uv run python experiments/visual.py              # todo
    uv run python experiments/visual.py --cada 6     # un cuadro cada 6 pasos

Salidas en experiments/resultados/visual/:
  - construccion_{greedy,astar,d2}.gif : construccion sin carga, con cuadros
    DURANTE cada estabilizacion (no solo el estado final de cada paso).
  - colapso_greedy.gif : episodio del greedy con umbral x0.75 y p_carga 0.4
    en una seed donde colapsa al llegar. La prueba de carga W es una fase
    aparte (se aplica sobre una copia y se descarta); en el colapso se marca
    la union rota y se sigue simulando unos segundos sin ella.
  - comparacion.png : las tres estructuras finales, misma escala y colores.
  - chequeos.md : chequeos automaticos (suelo, cruces, asentamiento, carga en
    nodos fijos, ruptura de la union de mayor tension).
  - cuadros/*.png : cuadros clave para inspeccion.

Instrumentacion: se reemplaza wog.env.estabilizar por una copia EXACTA del
bucle de wog.fisica.estabilizar que ademas lee posiciones cada N pasos. Leer
posiciones no altera la simulacion, asi que el episodio es el mismo bit a bit
(se verifica contra el entorno sin instrumentar).
"""

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from unittest import mock

import numpy as np

import matplotlib.pyplot as plt                           # noqa: E402
from matplotlib import animation                          # noqa: E402
from matplotlib.collections import LineCollection         # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pymunk                                             # noqa: E402

from wog import fisica                                    # noqa: E402
from wog.env import GooPhysicsEnv, Theta                  # noqa: E402
from wog.fisica import (                                  # noqa: E402
    DT, K_ESCALA, MAXFORCE, PASO_MIN, PASOS_ESTABILIZAR, V_REPOSO, dinamicos,
    tension,
)
from wog.geometria import META_Y, R_BOLA                  # noqa: E402
from wog.politicas import PoliticaT4                      # noqa: E402

RESULTADOS = Path(__file__).resolve().parent / "resultados"
SALIDA = RESULTADOS / "visual"
SEGUIR_COLAPSO = 180          # pasos (3 s) tras la ruptura, sin la union
PAUSA = 6                     # cuadros repetidos al final de cada fase
LIM_X, LIM_Y = (-320, 470), (-40, 700)

CMAP = LinearSegmentedColormap.from_list(
    "tension", ["#1baf7a", "#eda100", "#e34948"])
NORM = Normalize(0.0, 1.0)
C = {"fija": "#0b0b0b", "dinamica": "#8a8984", "nueva": "#2a78d6",
     "carga": "#6250d6", "rota": "#e34948", "meta": "#2a78d6",
     "suelo": "#52514e", "fondo": "#fcfcfb", "tinta": "#0b0b0b"}


# --------------------------------------------------------------------------
# Grabacion
# --------------------------------------------------------------------------
class Grabador:
    """Recibe cuadros desde estabilizar_grabando. 'ctx' lo fija el que
    conduce el episodio (paso, n, fase); 'llamada' cuenta las
    estabilizaciones dentro de un env.step (0 = construccion, 1 = carga)."""

    def __init__(self, cada, umbral, seguir=False):
        self.cada = cada
        self.umbral = umbral
        self.seguir = seguir
        self.cuadros = []
        self.ctx = {}
        self.llamada = 0
        self.rupturas = []        # chequeo: rota vs mayor tension
        self.cargas = []          # chequeo: carga solo en nodos dinamicos
        self.min_y = math.inf     # chequeo: suelo (centro, px)

    def capturar(self, space, uniones, fase, rotas=(), pico=0.0, extra=""):
        bolas = []
        for b in space.bodies:
            x, y = b.position.x / K_ESCALA, b.position.y / K_ESCALA
            fija = b.body_type != pymunk.Body.DYNAMIC
            k = 0 if fija else int(round(b.mass / fisica.MASA)) - 1
            bolas.append((x, y, fija, k))
            if not fija:
                self.min_y = min(self.min_y, y)
        resortes = [(j.a.position.x / K_ESCALA, j.a.position.y / K_ESCALA,
                     j.b.position.x / K_ESCALA, j.b.position.y / K_ESCALA,
                     tension(j) / self.umbral) for j in uniones]
        rot = [(a.position.x / K_ESCALA, a.position.y / K_ESCALA,
                b.position.x / K_ESCALA, b.position.y / K_ESCALA)
               for a, b in rotas]
        ratio = max((r[4] for r in resortes), default=0.0)
        self.cuadros.append({
            "bolas": bolas, "resortes": resortes, "rotas": rot,
            "fase": fase, "ctx": dict(self.ctx), "ratio": ratio,
            "pico": pico / self.umbral, "extra": extra,
            "altura": max(y for _, y, f, _ in bolas)})

    def pausa(self, n=PAUSA):
        if self.cuadros:
            self.cuadros.extend([self.cuadros[-1]] * n)


REC = None


def estabilizar_grabando(space, uniones, umbral=None, pasos=PASOS_ESTABILIZAR,
                         paso_min=PASO_MIN):
    """Copia exacta de wog.fisica.estabilizar + captura de cuadros."""
    rec = REC
    fase = "construccion" if rec.llamada == 0 else "carga"
    rec.llamada += 1
    if fase == "carga":
        m = sum(int(round(b.mass / fisica.MASA)) - 1 for b in space.bodies
                if b.body_type == pymunk.Body.DYNAMIC)
        rec.cargas.append({
            "m": m, "en_fijas": sum(1 for b in space.bodies
                                    if b.body_type != pymunk.Body.DYNAMIC
                                    and b.mass != math.inf)})
        rec.ctx["m"] = m
    pico, rotas, rotas_cuerpos = 0.0, [], []
    rec.capturar(space, uniones, fase)
    resultado = None
    for p in range(pasos):
        space.step(DT)
        tens = [tension(j) for j in uniones]
        for j, f in zip(list(uniones), tens):
            pico = max(pico, f)
            if umbral is not None and f > umbral:
                rec.rupturas.append({"f": f, "max": max(tens),
                                     "es_max": f == max(tens),
                                     "fase": fase, "paso": p})
                rotas_cuerpos.append((j.a, j.b))
                space.remove(j)
                uniones.remove(j)
                rotas.append((p, f))
        if p % rec.cada == 0:
            rec.capturar(space, uniones, fase, rotas_cuerpos, pico)
        v = max((b.velocity.length for b in dinamicos(space)), default=0)
        if p > paso_min and v < V_REPOSO:
            resultado = fisica.Estabilizacion(p, True, pico, rotas)
            break
    if resultado is None:
        resultado = fisica.Estabilizacion(pasos - 1, False, pico, rotas)
    rec.capturar(space, uniones, fase, rotas_cuerpos, pico)
    rec.pausa()

    if rotas and rec.seguir:
        # Solo para ver la caida: se sigue sobre una copia, sin tocar el
        # resultado ni el estado que usa el entorno.
        sp, un, rc = copy.deepcopy((space, uniones, rotas_cuerpos))
        for p in range(SEGUIR_COLAPSO):
            sp.step(DT)
            for j in list(un):
                if umbral is not None and tension(j) > umbral:
                    rc.append((j.a, j.b))
                    sp.remove(j)
                    un.remove(j)
            if p % rec.cada == 0:
                rec.capturar(sp, un, "caida", rc, pico)
        rec.pausa(2 * PAUSA)
    return resultado


# --------------------------------------------------------------------------
# Planes
# --------------------------------------------------------------------------
def plan_politica(pol):
    env = GooPhysicsEnv(p_carga=0.0)
    env.reset(seed=0)
    plan = []
    while True:
        a, _ = pol(env)
        plan.append(env.tabla[a])
        _, _, term, _, _ = env.step(a)
        if term:
            return plan


def plan_astar():
    sec = json.loads((RESULTADOS / "astar_optimo.json").read_text())
    return [(s["i"], s["j"], s["Li"], s["Lj"], s["lado"])
            for s in sec["secuencia"]]


def planes():
    return {"greedy": ("greedy (D1 e_max=∞)",
                       plan_politica(PoliticaT4(math.inf, "apoyo"))),
            "astar": ("A* (5 bolas)", plan_astar()),
            "d2": ("D2 e_max=2.5",
                   plan_politica(PoliticaT4(2.5, "tercio")))}


def correr(plan, cada, theta=None, p_carga=0.0, seed=0, seguir=False):
    """Episodio instrumentado. Devuelve (grabador, env, infos, Ps)."""
    global REC
    theta = theta or Theta()
    REC = Grabador(cada, theta.umbral, seguir)
    infos, Ps, previas = [], [], []
    with mock.patch("wog.env.estabilizar", estabilizar_grabando):
        env = GooPhysicsEnv(theta=theta, p_carga=p_carga)
        REC.ctx = {"paso": 0, "n": env.presupuesto}
        REC.llamada = 0
        env.reset(seed=seed)
        for t, acc in enumerate(plan, 1):
            a = env.codificar(*acc)
            Ps.append(env.predicciones()[a])
            previas.append(dict(env.pos))
            REC.ctx = {"paso": t, "n": env.restantes - 1, "accion": acc}
            REC.llamada = 0
            _, r, term, _, info = env.step(a)
            infos.append(info)
            if info.get("m_carga") and not info["rotas"]:
                # la copia se descarta: se vuelve a mostrar el estado real
                REC.ctx = {"paso": t, "n": env.restantes}
                REC.capturar(env.space, env.uniones, "descartada")
                REC.pausa(PAUSA // 2)
            if term:
                REC.ctx["causa"] = info["causa"]
                extra = f"fin: {info['causa']}  r = {r:g}"
                if info["causa"] == "colapso" and seguir:
                    # el ultimo cuadro es el de la caida, no el estado real
                    # (con colapso por carga, la copia rota se descarto)
                    REC.cuadros.append(dict(REC.cuadros[-1], fase="final",
                                            extra=extra))
                else:
                    REC.capturar(env.space, env.uniones, "final", extra=extra)
                REC.pausa(3 * PAUSA)
                break
    return REC, env, infos, Ps, previas


# --------------------------------------------------------------------------
# Dibujo
# --------------------------------------------------------------------------
def limites(cuadros, margen=40):
    """Encuadre que contiene todos los cuadros (y al menos LIM_X, LIM_Y)."""
    xs = [x for c in cuadros for x, *_ in c["bolas"]]
    ys = [y for c in cuadros for _, y, *_ in c["bolas"]]
    return ((min(LIM_X[0], min(xs) - margen), max(LIM_X[1], max(xs) + margen)),
            (min(LIM_Y[0], min(ys) - margen), max(LIM_Y[1], max(ys) + margen)))


def preparar_eje(ax, lims):
    (x0, x1), (y0, y1) = lims
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_facecolor(C["fondo"])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])


def dibujar(ax, cu, titulo="", lims=(LIM_X, LIM_Y)):
    preparar_eje(ax, lims)
    (x0, x1), (y0, y1) = lims
    ax.fill_between((x0, x1), y0, 0, color="#e4e3df", zorder=0)
    ax.axhline(0, color=C["suelo"], lw=1.2, zorder=1)
    ax.axhline(META_Y, color=C["meta"], lw=1, ls="--", zorder=1)
    ax.text(x1 - 8, META_Y + 8, "meta 600 px", ha="right",
            va="bottom", fontsize=8, color=C["meta"])
    segs = [((x1, y1), (x2, y2)) for x1, y1, x2, y2, _ in cu["resortes"]]
    cols = [CMAP(NORM(min(r, 1.0))) for *_, r in cu["resortes"]]
    ax.add_collection(LineCollection(segs, colors=cols, linewidths=3,
                                     zorder=2))
    for x1, y1, x2, y2 in cu["rotas"]:
        ax.plot([x1, x2], [y1, y2], color=C["rota"], lw=2, ls=":", zorder=3)
        ax.plot((x1 + x2) / 2, (y1 + y2) / 2, marker="X", ms=11,
                color=C["rota"], mec="white", zorder=6)
    for x, y, fija, k in cu["bolas"]:
        if fija:
            ax.add_patch(plt.Circle((x, y), R_BOLA, color=C["fija"],
                                    zorder=4))
            ax.plot([x - 14, x + 14], [y - 20, y - 20], color=C["fija"],
                    lw=2, zorder=4)
        else:
            ax.add_patch(plt.Circle((x, y), R_BOLA, color=C["dinamica"],
                                    zorder=4))
        if k:
            ax.add_patch(plt.Circle((x, y), R_BOLA * 1.25, fill=False,
                                    ec=C["carga"], lw=2.5, zorder=5))
            ax.text(x + 18, y + 10, f"+{k}", color=C["carga"], fontsize=9,
                    fontweight="bold", zorder=6, clip_on=True)
    ctx = cu["ctx"]
    rot = {"construccion": "estabilizando", "carga": "PRUEBA DE CARGA "
           f"(copia), m = {ctx.get('m', 0)}", "descartada":
           "carga descartada: estado real", "caida": "COLAPSO: union rota, "
           "sigue la simulacion", "final": cu["extra"]}[cu["fase"]]
    pico = (f"   (pico de la fase {cu['pico']:.2f})"
            if cu["fase"] in ("construccion", "carga", "caida") else "")
    lin = [titulo, f"paso {ctx.get('paso', 0)}   n = {ctx.get('n', '')}   "
           f"altura = {cu['altura']:.0f} px",
           f"tension max/umbral = {cu['ratio']:.2f}{pico}", rot]
    ax.set_title("\n".join(lin), loc="left", fontsize=8.5,
                 fontweight="normal", color=C["tinta"], family="monospace")
    if cu["fase"] in ("carga", "caida"):
        for sp in ax.spines.values():          # marco de la fase
            sp.set_visible(True)
            sp.set_color(C["carga"] if cu["fase"] == "carga" else C["rota"])
            sp.set_linewidth(5)


def barra(fig, ax):
    sm = plt.cm.ScalarMappable(norm=NORM, cmap=CMAP)
    cb = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("tension / umbral (≥ 1 se rompe)")


def gif(rec, ruta, titulo, fps=20):
    lims = limites(rec.cuadros)
    ancho = (lims[0][1] - lims[0][0]) / (lims[1][1] - lims[1][0])
    fig, ax = plt.subplots(figsize=(1.2 + 5.2 * ancho, 6.2), dpi=80)
    fig.patch.set_facecolor(C["fondo"])
    barra(fig, ax)

    def cuadro(i):
        ax.clear()
        dibujar(ax, rec.cuadros[i], titulo, lims)
    anim = animation.FuncAnimation(fig, cuadro, frames=len(rec.cuadros))
    anim.save(ruta, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)


def png(cu, ruta, titulo, lims=(LIM_X, LIM_Y)):
    fig, ax = plt.subplots(figsize=(5.6, 6.2), dpi=100)
    fig.patch.set_facecolor(C["fondo"])
    barra(fig, ax)
    dibujar(ax, cu, titulo, lims)
    fig.savefig(ruta, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# Chequeos
# --------------------------------------------------------------------------
def se_cortan(p1, p2, p3, p4):
    """Interseccion propia de segmentos (sin compartir extremos)."""
    def o(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    return (o(p1, p2, p3) * o(p1, p2, p4) < 0
            and o(p3, p4, p1) * o(p3, p4, p2) < 0)


def dist_seg(p, a, b):
    ax_, ay = a
    dx, dy = b[0] - ax_, b[1] - ay
    t = max(0.0, min(1.0, ((p[0] - ax_) * dx + (p[1] - ay) * dy)
                     / (dx * dx + dy * dy)))
    return math.dist(p, (ax_ + t * dx, ay + t * dy))


def chequear_estructura(env):
    pos = env.pos
    aristas = list(env.G.edges)
    cruces, holgura = [], math.inf
    for i, (a, b) in enumerate(aristas):
        for c, d in aristas[i + 1:]:
            if {a, b} & {c, d}:
                continue
            if se_cortan(pos[a], pos[b], pos[c], pos[d]):
                cruces.append(((a, b), (c, d)))
    for n in pos:
        for a, b in aristas:
            if n not in (a, b):
                holgura = min(holgura, dist_seg(pos[n], pos[a], pos[b]))
    return cruces, holgura


def asentamiento(Ps, previas, env_pos_por_paso):
    """Por paso: |pos final de la bola nueva - P| y max desplazamiento de
    las bolas ya existentes."""
    filas = []
    for t, (P, prev, pos) in enumerate(zip(Ps, previas, env_pos_por_paso), 1):
        nueva = pos[3 + t]
        desp = max(math.dist(prev[n], pos[n]) for n in prev)
        filas.append((t, P, nueva, math.dist(P, nueva), desp))
    return filas


def posiciones_por_paso(plan, theta=None):
    env = GooPhysicsEnv(theta=theta or Theta(), p_carga=0.0)
    env.reset(seed=0)
    out = []
    for acc in plan:
        env.step(env.codificar(*acc))
        out.append(dict(env.pos))
    return out


def buscar_seed(plan, f=0.75, p=0.4, seeds=range(200)):
    """Primera seed en que el plan colapsa por carga en el paso de llegada."""
    for s in seeds:
        env = GooPhysicsEnv(theta=Theta(umbral=f * MAXFORCE), p_carga=p)
        env.reset(seed=s)
        for t, acc in enumerate(plan, 1):
            _, _, term, _, info = env.step(env.codificar(*acc))
            if term:
                break
        if (info["causa"] == "colapso" and info["colapso_por"] == "carga"
                and t == len(plan)):
            return s, info
    raise RuntimeError("ninguna seed colapsa al llegar")


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cada", type=int, default=6,
                    help="un cuadro cada N pasos de fisica (DT = 1/60 s)")
    args = ap.parse_args()
    plt.switch_backend("Agg")       # solo archivos; en_vivo.py usa ventana
    SALIDA.mkdir(parents=True, exist_ok=True)
    (SALIDA / "cuadros").mkdir(exist_ok=True)
    md = ["# Chequeos de la revision visual", "",
          f"Un cuadro cada {args.cada} pasos de fisica "
          f"({args.cada / 60:.2f} s). Posiciones en px, centro de la bola.",
          ""]

    # 1. Construccion sin carga
    finales = {}
    for clave, (titulo, plan) in planes().items():
        rec, env, infos, Ps, previas = correr(plan, args.cada)
        # la instrumentacion no altera la simulacion
        ref = posiciones_por_paso(plan)[-1]
        assert ref == env.pos, f"{clave}: la instrumentacion cambio la fisica"
        gif(rec, SALIDA / f"construccion_{clave}.gif", titulo)
        finales[clave] = (titulo, rec.cuadros[-1], env)
        # cuadros clave: fin de cada paso
        idx_fin = [i for i, c in enumerate(rec.cuadros[:-1])
                   if c["ctx"].get("paso") != rec.cuadros[i + 1]["ctx"]
                   .get("paso")] + [len(rec.cuadros) - 1]
        for i in idx_fin:
            png(rec.cuadros[i], SALIDA / "cuadros" /
                f"{clave}_paso{rec.cuadros[i]['ctx'].get('paso', 0)}.png",
                titulo)
        # pico transitorio del paso (cuadro de maxima tension)
        imax = max(range(len(rec.cuadros)),
                   key=lambda i: rec.cuadros[i]["ratio"])
        png(rec.cuadros[imax], SALIDA / "cuadros" / f"{clave}_pico.png",
            titulo)

        cruces, holgura = chequear_estructura(env)
        filas = asentamiento(Ps, previas, posiciones_por_paso(plan))
        md += [f"## {titulo}", "",
               f"- bolas: {len(plan)}; altura final {env.altura():.1f} px; "
               f"causa: {infos[-1]['causa']}",
               f"- centro mas bajo de una bola dinamica en TODOS los cuadros:"
               f" {rec.min_y:.2f} px (radio {R_BOLA:g}; < {R_BOLA:g} = "
               "penetra el suelo)",
               f"- cruces propios de uniones en la estructura final: "
               f"{cruces or 'ninguno'}"
               + (" (las diagonales del K4 base, por diseño)"
                  if cruces == [((0, 2), (1, 3))] else ""),
               f"- holgura minima bola-union (no extremo): {holgura:.1f} px",
               f"- pico de tension/umbral en toda la construccion: "
               f"{max(c['ratio'] for c in rec.cuadros):.3f} (cuadro "
               f"{imax}, paso {rec.cuadros[imax]['ctx'].get('paso')})",
               "", "| paso | accion | P predicha | posicion final | "
               "|final − P| | max desplazamiento de las previas |",
               "|---|---|---|---|---|---|"]
        for (t, P, nueva, d, desp), acc in zip(filas, plan):
            md.append(f"| {t} | {acc} | ({P[0]:.0f}, {P[1]:.0f}) | "
                      f"({nueva[0]:.0f}, {nueva[1]:.0f}) | {d:.1f} | "
                      f"{desp:.1f} |")
        md.append("")

    # 2. Episodio con colapso
    titulo, plan = planes()["greedy"]
    theta = Theta(umbral=0.75 * MAXFORCE)
    seed, info = buscar_seed(plan)
    rec, env, infos, _, _ = correr(plan, args.cada, theta=theta, p_carga=0.4,
                                   seed=seed, seguir=True)
    assert infos[-1]["causa"] == "colapso" and len(infos) == len(plan)
    gif(rec, SALIDA / "colapso_greedy.gif",
        f"greedy, umbral ×0.75, p 0.4, seed {seed}")
    claves = {"carga": None, "caida": None}
    for i, c in enumerate(rec.cuadros):
        if c["fase"] == "carga" and c["rotas"] and claves["carga"] is None:
            claves["carga"] = i
        if c["fase"] == "caida":
            claves["caida"] = i
    for nombre, i in claves.items():
        if i is not None:
            png(rec.cuadros[i], SALIDA / "cuadros" / f"colapso_{nombre}.png",
                f"greedy, umbral ×0.75, p 0.4, seed {seed}",
                limites(rec.cuadros))
    md += ["## Episodio con colapso (greedy, umbral ×0.75, p_carga 0.4)", "",
           f"- seed {seed} (primera de 0..199 que colapsa por carga en el "
           f"paso de llegada, paso {len(plan)})",
           "- pruebas de carga: " + ", ".join(
               f"m = {c['m']} (en fijas: {c['en_fijas']})"
               for c in rec.cargas),
           f"- centro mas bajo de una bola dinamica (incluida la caida): "
           f"{rec.min_y:.2f} px", "- rupturas:"]
    for r in rec.rupturas:
        md.append(f"  - fase {r['fase']}, paso de fisica {r['paso']}: "
                  f"F = {r['f']:.2f} ({r['f'] / theta.umbral:.3f} × umbral);"
                  f" mayor tension en ese paso {r['max']:.2f} -> "
                  f"{'es la mayor' if r['es_max'] else 'NO es la mayor'}")
    md.append("")

    # 3. Comparacion de las estructuras finales
    fig, axs = plt.subplots(1, 3, figsize=(15, 6.0), dpi=100)
    fig.patch.set_facecolor(C["fondo"])
    for ax, (titulo, cu, env) in zip(axs, finales.values()):
        dibujar(ax, dict(cu, fase="final", extra=f"{env.restantes} bolas "
                         "restantes"), titulo)
    barra(fig, axs.tolist())
    fig.savefig(SALIDA / "comparacion.png", bbox_inches="tight")
    plt.close(fig)

    (SALIDA / "chequeos.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
