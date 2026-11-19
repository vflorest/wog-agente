"""
Agente Thompson sampling que elige, episodio a episodio, entre planes fijos
en un regimen theta que NO conoce, con una ventana de matplotlib en vivo.

Uso:
    uv run python experiments/en_vivo.py --umbral 0.75 --p-carga 0.4 \
        --episodios 100 --rapido
    uv run python experiments/en_vivo.py --umbral 1 --p-carga 0.2 \
        --plan L6-03.json --plan L7-01.json

Por que un bandido: W nunca cambia el estado, asi que cada plan determinista
construye siempre la misma estructura y su recompensa por episodio es
R_k = bolas restantes del plan (conocida) con probabilidad q_k (desconocida,
depende de theta) y 0 si no. Elegir plan es un bandido de Bernoulli con
pagos escalados:
  - posterior de q_k: Beta(1 + exitos_k, 1 + fracasos_k) (prior uniforme);
  - Thompson: se sortea q~_k de cada posterior y se juega argmax q~_k * R_k.
El regimen solo entra en el entorno: el agente ve exitos y fracasos.

Oraculo (conociendo theta): antes de empezar se estima q_k de cada plan con
--seeds-oraculo seeds propias (10000, 10001, ...), separadas de las del
agente. El oraculo juega siempre k* = argmax q_k * R_k. En el panel de
recompensa se compara el acumulado del agente con el del oraculo jugado en
la MISMA seed de cada episodio (numeros aleatorios comunes) y con la
esperanza t * max_k q_k R_k. Las q estimadas se marcan en el panel de los
posteriores solo para quien mira: el agente no las usa.

Paneles (se actualizan en cada episodio):
  1. la estructura construyendose (cuadros de visual.py: tension/umbral,
     prueba de carga, ruptura y caida); se omite con --rapido;
  2. el posterior Beta de exito de cada plan;
  3. la recompensa acumulada del agente contra la del oraculo.

Planes: las dos referencias con nombre, greedy y A*, mas los candidatos de
--plan (se puede repetir): un JSON {"nombre": "L6-#03", "acciones": [[i, j,
Li, Lj, lado], ...]} o la lista de acciones sola. Los candidatos salen de
experiments/planes.py y se nombran por largo e indice. Si se eligen segun
la evaluacion de un regimen, conviene fijarlos de antemano (no segun el
theta de la corrida) para no filtrarle theta al agente.
"""

import argparse
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import visual                                             # noqa: E402
from wog.env import GooPhysicsEnv, Theta                  # noqa: E402
from wog.fisica import MAXFORCE                           # noqa: E402

SEED_ORACULO = 10_000
LIMS = ((-760, 480), (-40, 700))           # caben la torre y su caida
COLORES = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#eda100"]
TINTA, REJILLA, FONDO = "#0b0b0b", "#e4e3df", "#fcfcfb"


def beta_pdf(x, a, b):
    """Densidad Beta(a, b) en x (0 < x < 1), sin scipy."""
    log_b = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    return np.exp((a - 1) * np.log(x) + (b - 1) * np.log1p(-x) - log_b)


# --------------------------------------------------------------------------
# Planes y episodios
# --------------------------------------------------------------------------
def cargar_planes(rutas):
    planes = {titulo: plan for titulo, plan in visual.planes(rutas).values()}
    if not rutas:
        print("aviso: sin --plan; se juegan solo greedy y A* (los candidatos "
              "salen de experiments/planes.py)")
    return planes


def episodio(args):
    """Juega un plan en (umbral, p_carga, seed). Devuelve la recompensa."""
    plan, f_umbral, p_carga, seed = args
    env = GooPhysicsEnv(theta=Theta(umbral=f_umbral * MAXFORCE),
                        p_carga=p_carga)
    env.reset(seed=seed)
    for acc in plan:
        _, r, term, _, _ = env.step(env.codificar(*acc))
        if term:
            return r
    raise RuntimeError("el plan no termina el episodio")


def bolas_al_llegar(plan):
    """R_k: bolas restantes si el plan llega (no depende de theta)."""
    return 14 - len(plan)


def oraculo(planes, f_umbral, p_carga, n_seeds):
    """q_k estimada por plan con seeds propias del oraculo."""
    tareas = [(plan, f_umbral, p_carga, SEED_ORACULO + s)
              for plan in planes.values() for s in range(n_seeds)]
    with ProcessPoolExecutor() as ex:
        r = list(ex.map(episodio, tareas, chunksize=8))
    r = np.array(r).reshape(len(planes), n_seeds)
    return {k: float((r[i] > 0).mean()) for i, k in enumerate(planes)}


# --------------------------------------------------------------------------
# Agente
# --------------------------------------------------------------------------
class Thompson:
    def __init__(self, nombres, R, seed):
        self.nombres = list(nombres)
        self.R = np.array([R[k] for k in self.nombres], float)
        self.a = np.ones(len(self.nombres))      # 1 + exitos
        self.b = np.ones(len(self.nombres))      # 1 + fracasos
        self.rng = np.random.default_rng(seed)
        self.ultima_muestra = None

    def elegir(self):
        q = self.rng.beta(self.a, self.b)
        self.ultima_muestra = q
        return self.nombres[int(np.argmax(q * self.R))]

    def actualizar(self, nombre, exito):
        i = self.nombres.index(nombre)
        if exito:
            self.a[i] += 1
        else:
            self.b[i] += 1


# --------------------------------------------------------------------------
# Ventana
# --------------------------------------------------------------------------
class Ventana:
    def __init__(self, nombres, R, q_real, k_oraculo, regimen, rapido, fps):
        self.nombres, self.R, self.q_real = nombres, R, q_real
        self.k_oraculo, self.regimen = k_oraculo, regimen
        self.rapido, self.fps = rapido, fps
        self.color = dict(zip(nombres, COLORES))
        self.interactiva = plt.get_backend().lower() not in (
            "agg", "pdf", "svg", "ps", "cairo", "template")
        plt.ion()
        if rapido:
            self.fig, (self.ax_beta, self.ax_r) = plt.subplots(
                1, 2, figsize=(12, 5.4), layout="constrained")
            self.ax_est = None
        else:
            self.fig, (self.ax_est, self.ax_beta, self.ax_r) = plt.subplots(
                1, 3, figsize=(18, 6.2), layout="constrained",
                gridspec_kw={"width_ratios": [1.3, 1, 1]})
            visual.barra(self.fig, self.ax_est)
        self.fig.patch.set_facecolor(FONDO)
        self.fig.suptitle(f"Thompson sampling sobre planes fijos — regimen "
                          f"{regimen} (desconocido para el agente)",
                          fontsize=11, fontweight="bold")
        self.x = np.linspace(0.0005, 0.9995, 400)

    def pausa(self, s):
        if self.interactiva:
            plt.pause(s)

    def construccion(self, rec, titulo):
        if self.ax_est is None:
            return
        for cu in rec.cuadros:
            self.ax_est.clear()
            visual.dibujar(self.ax_est, cu, titulo, LIMS)
            self.fig.canvas.draw_idle()
            self.pausa(1 / self.fps)

    def aprendizaje(self, ag, t, eleccion, r, acum, acum_or, acum_esp):
        ax = self.ax_beta
        ax.clear()
        for i, k in enumerate(self.nombres):
            c = self.color[k]
            y = beta_pdf(self.x, ag.a[i], ag.b[i])
            n = int(ag.a[i] + ag.b[i] - 2)
            ax.plot(self.x, y, color=c, lw=2.2 if k == eleccion else 1.4,
                    label=f"{k} (R={self.R[k]}, n={n}, "
                          f"media {ag.a[i] / (ag.a[i] + ag.b[i]):.2f})")
            ax.fill_between(self.x, y, color=c, alpha=0.12)
            ax.axvline(self.q_real[k], color=c, ls=":", lw=1.2)
            if ag.ultima_muestra is not None:
                ax.plot(ag.ultima_muestra[i], 0, marker="v", ms=9, color=c,
                        clip_on=False)
        ax.set_xlim(0, 1)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("q = P(exito del plan)")
        ax.set_ylabel("densidad posterior")
        ax.set_title(f"Posterior Beta, episodio {t}\n▼ muestra de Thompson"
                     "   ··· q del oraculo", fontsize=9.5)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(color=REJILLA)

        ax = self.ax_r
        ax.clear()
        ts = np.arange(1, len(acum) + 1)
        ax.plot(ts, acum, color=COLORES[0], lw=2, label="agente")
        ax.plot(ts, acum_or, color=TINTA, lw=1.4,
                label=f"oraculo ({self.k_oraculo}), misma seed")
        ax.plot(ts, acum_esp, color=TINTA, lw=1, ls="--",
                label="oraculo, esperanza t·max q·R")
        ax.set_xlabel("episodio")
        ax.set_ylabel("bolas acumuladas")
        ax.set_title(f"Recompensa acumulada\nultimo: {eleccion} -> r = "
                     f"{r:g}; regret {acum_esp[-1] - acum[-1]:+.1f}",
                     fontsize=9.5)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(color=REJILLA)
        self.fig.canvas.draw_idle()
        self.pausa(0.001)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--umbral", type=float, default=1.0,
                    help="factor sobre MAXFORCE (regimen: 0.75, 1, 1.25)")
    ap.add_argument("--p-carga", type=float, default=0.2)
    ap.add_argument("--episodios", type=int, default=100)
    ap.add_argument("--rapido", action="store_true",
                    help="sin animacion de construccion")
    ap.add_argument("--plan", action="append",
                    help="JSON de un plan candidato (se puede repetir)")
    ap.add_argument("--seed", type=int, default=0,
                    help="seed del primer episodio (luego +1 por episodio)")
    ap.add_argument("--seed-agente", type=int, default=0,
                    help="seed del muestreo de Thompson")
    ap.add_argument("--seeds-oraculo", type=int, default=200)
    ap.add_argument("--cada", type=int, default=12,
                    help="un cuadro cada N pasos de fisica en la animacion")
    ap.add_argument("--fps", type=float, default=60)
    ap.add_argument("--guardar", help="PNG con el estado final de la ventana")
    args = ap.parse_args()

    planes = cargar_planes(args.plan)
    R = {k: bolas_al_llegar(p) for k, p in planes.items()}
    regimen = f"umbral ×{args.umbral:g}, p_carga {args.p_carga:g}"
    print(f"oraculo: estimando q con {args.seeds_oraculo} seeds por plan...")
    q_real = oraculo(planes, args.umbral, args.p_carga, args.seeds_oraculo)
    k_or = max(planes, key=lambda k: q_real[k] * R[k])
    for k in planes:
        print(f"  {k:10s} {len(planes[k])} bolas, R = {R[k]}, q = "
              f"{q_real[k]:.3f}, q·R = {q_real[k] * R[k]:.2f}"
              + ("   <- oraculo" if k == k_or else ""))
    v_or = q_real[k_or] * R[k_or]

    ag = Thompson(planes, R, args.seed_agente)
    win = Ventana(list(planes), R, q_real, k_or, regimen, args.rapido,
                  args.fps)
    theta = Theta(umbral=args.umbral * MAXFORCE)
    acum, acum_or, acum_esp, elecciones = [], [], [], []
    for t in range(1, args.episodios + 1):
        seed = args.seed + t - 1
        k = ag.elegir()
        if args.rapido:
            r = episodio((planes[k], args.umbral, args.p_carga, seed))
        else:
            rec, env, infos, _, _ = visual.correr(
                planes[k], args.cada, theta=theta, p_carga=args.p_carga,
                seed=seed, seguir=True)
            r = float(env.restantes) if infos[-1]["causa"] == "meta" else 0.0
            win.construccion(rec, f"ep. {t}: {k} (seed {seed})")
        r_or = r if k == k_or else episodio(
            (planes[k_or], args.umbral, args.p_carga, seed))
        ag.actualizar(k, r > 0)
        elecciones.append(k)
        acum.append((acum[-1] if acum else 0) + r)
        acum_or.append((acum_or[-1] if acum_or else 0) + r_or)
        acum_esp.append(t * v_or)
        win.aprendizaje(ag, t, k, r, acum, acum_or, acum_esp)

    print(f"\n{args.episodios} episodios en {regimen}:")
    for k in planes:
        print(f"  {k:10s} jugado {elecciones.count(k):4d} veces")
    print(f"  agente {acum[-1]:.0f} bolas, oraculo (misma seed) "
          f"{acum_or[-1]:.0f}, esperanza oraculo {acum_esp[-1]:.1f}")
    if args.guardar:
        win.fig.savefig(args.guardar, bbox_inches="tight")
        print(f"  figura en {args.guardar}")
    if win.interactiva:
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
