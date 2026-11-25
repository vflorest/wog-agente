"""
Chequeo geometrico (sin fisica) de la meta "boca" del nivel real GoingUp.

En el juego la meta no es una altura: una bola entra cuando su centro queda
a <= R_SALIDA = 75 px de la boca E (levelexit). Medido en GoingUp.level, la
boca esta en dx = +18,25 px desde el centroide de la base y dy = 414 px sobre
el promedio de los centros inferiores (ver CLAUDE.md, "Nivel real").

Variantes (REGLAS_T4, k = 2, 14 bolas):
  a) base del modelo: K4 cuadrado de 150 px, E = (75 + 18,25, 414);
  b) base real de GoingUp trasladada en y para que el promedio de los
     centros inferiores quede en y = 0 (suelo plano y = 0 como
     simplificacion), E en la misma posicion relativa = (0, 414).
Referencia: a_estrella() actual (meta altura >= META_Y = 600).

No toca wog/ ni el entorno: META_Y sigue en 600 para el Taller 5.

Uso:
    uv run python experiments/meta_real.py
Salida en experiments/resultados/meta_real.md.
"""

import heapq
import itertools
import math
import sys
import time
from pathlib import Path

import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wog.busqueda import a_estrella, clave                # noqa: E402
from wog.geometria import (                               # noqa: E402
    BOLAS_DISPONIBLES, MAXLEN, META_Y, REGLAS_T4, Estado, dist,
    generar_acciones,
)

R_SALIDA = 75.0                     # radius del levelexit
DX_BOCA = 18.25                     # desde el centroide de la base
DY_BOCA = 414.0                     # desde el promedio de centros inferiores

# GoingUp.level: BallInstance 0..3 y sus Strand (K4)
BASE_REAL = {0: (-88.5, 98.5), 1: (49.5, 101.5),
             2: (39.5, 205.5), 3: (-73.5, 189.5)}
STRANDS_REAL = [(0, 2), (3, 2), (2, 1), (1, 0), (0, 3), (3, 1)]
SALIDA = Path(__file__).resolve().parent / "resultados" / "meta_real.md"


def boca(pos, inferiores):
    cx = sum(p[0] for p in pos.values()) / len(pos)
    y0 = sum(pos[i][1] for i in inferiores) / len(inferiores)
    return (cx + DX_BOCA, y0 + DY_BOCA)


def estado_real():
    y0 = (BASE_REAL[0][1] + BASE_REAL[1][1]) / 2          # 100
    pos = {n: (x, y - y0) for n, (x, y) in BASE_REAL.items()}
    G = nx.Graph()
    G.add_nodes_from(pos)
    G.add_edges_from(STRANDS_REAL)
    return Estado(G, pos, BOLAS_DISPONIBLES)


def a_estrella_boca(inicial, E, k=2, reglas=REGLAS_T4, cerrar=True,
                    decimales=6):
    """
    A* con meta: existe un nodo con dist(nodo, E) <= R_SALIDA.
    h = max(0, (min dist(n, E) - R_SALIDA) / MAXLEN): cada bola nueva queda
    a <= MAXLEN de un nodo existente, asi que la distancia minima a E baja a
    lo mas MAXLEN por bola (admisible y consistente). Prueba de meta al
    extraer el nodo.
    """
    def dmin(e):
        return min(dist(p, E) for p in e.pos.values())

    def h(e):
        return max(0.0, (dmin(e) - R_SALIDA) / MAXLEN)

    t0 = time.perf_counter()
    orden = itertools.count()
    abierta = [(h(inicial), 0, next(orden), inicial, ())]
    mejor_g = {clave(inicial, decimales): 0} if cerrar else None
    expandidos = 0
    while abierta:
        _, g, _, e, camino = heapq.heappop(abierta)
        if cerrar and mejor_g.get(clave(e, decimales), math.inf) < g:
            continue
        if dmin(e) <= R_SALIDA:
            return g, list(camino), e, expandidos, time.perf_counter() - t0
        expandidos += 1
        for acc in generar_acciones(e, k, reglas)[0]:
            hijo = e.aplicar(acc)
            if cerrar:
                c = clave(hijo, decimales)
                if mejor_g.get(c, math.inf) <= g + 1:
                    continue
                mejor_g[c] = g + 1
            heapq.heappush(abierta, (g + 1 + h(hijo), g + 1, next(orden),
                                     hijo, camino + (acc,)))
    return None, [], None, expandidos, time.perf_counter() - t0


def fmt(p):
    return f"({p[0]:.2f}, {p[1]:.2f})"


def reporte(nombre, inicial, E, lineas):
    res = {}
    for cerrar in (True, False):
        res[cerrar] = a_estrella_boca(inicial, E, cerrar=cerrar)
    g, camino, final, expandidos, seg = res[True]
    g2, _, _, exp2, seg2 = res[False]
    lineas.append(f"## {nombre}\n")
    lineas.append(f"- Base: " + ", ".join(
        f"{n} {fmt(p)}" for n, p in inicial.pos.items()))
    lineas.append(f"- E = {fmt(E)}, radio {R_SALIDA:.0f}")
    if g is None:
        lineas.append("- Sin solucion con 14 bolas.\n")
        return None
    lineas.append(f"- **Minimo: {g} bolas** (sin cierre: {g2}; "
                  f"expandidos {expandidos} / {exp2}, "
                  f"{seg:.2f} s / {seg2:.2f} s)")
    lineas.append("")
    lineas.append("| paso | nodo | anclas (i, j) | Li, Lj | lado | P | "
                  "dist a E |")
    lineas.append("|---|---|---|---|---|---|---|")
    n0 = max(inicial.G.nodes)
    for t, (i, j, Li, Lj, lado, P) in enumerate(camino, 1):
        lineas.append(f"| {t} | {n0 + t} | ({i}, {j}) | {Li:.0f}, {Lj:.0f} "
                      f"| {lado:+d} | {fmt(P)} | {dist(P, E):.2f} |")
    dentro = [(n, p) for n, p in final.pos.items() if dist(p, E) <= R_SALIDA]
    lineas.append("")
    lineas.append("- Bola que entra al radio: " + "; ".join(
        f"nodo {n} en {fmt(p)} (dist {dist(p, E):.2f})" for n, p in dentro))
    lineas.append("")
    return g


def main():
    lineas = ["# Meta \"boca\" del nivel real (solo geometria)", "",
              "A* con REGLAS_T4, k = 2, 14 bolas. Meta: algun nodo a "
              f"<= {R_SALIDA:.0f} px de la boca E. Generado por "
              "`experiments/meta_real.py`.", ""]

    ref = a_estrella()
    lineas.append(f"## Referencia actual (meta altura >= {META_Y:.0f})\n")
    lineas.append(f"- Minimo: {ref.minimo} bolas; P final "
                  f"{fmt(ref.acciones[-1][-1])}\n")

    a = Estado.inicial(BOLAS_DISPONIBLES)
    ga = reporte("a) Base del modelo (K4 cuadrado 150 px)", a,
                 boca(a.pos, (0, 1)), lineas)
    b = estado_real()
    gb = reporte("b) Base real de GoingUp (centros inferiores en y = 0)", b,
                 boca(b.pos, (0, 1)), lineas)

    lineas.append("## Resumen\n")
    lineas.append("| meta | base | minimo |")
    lineas.append("|---|---|---|")
    lineas.append(f"| altura >= {META_Y:.0f} | modelo | {ref.minimo} |")
    lineas.append(f"| boca r = 75 | modelo | {ga} |")
    lineas.append(f"| boca r = 75 | real | {gb} |")
    texto = "\n".join(lineas) + "\n"
    SALIDA.write_text(texto)
    print(texto)


if __name__ == "__main__":
    main()
