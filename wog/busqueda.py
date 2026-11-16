"""
Busqueda geometrica (sin fisica) del minimo de bolas para llegar a la meta.

A* sobre Estado (Fase 1) con:
  - transiciones: generar_acciones(estado, k, reglas) y Estado.aplicar;
  - costo g: bolas usadas (1 por accion);
  - heuristica h: Estado.heuristica() = max(0, (META_Y - altura) / MAXLEN).
    Admisible y consistente: P queda a <= MAXLEN de sus anclas, que estan a
    altura <= altura(s), asi que una bola sube la altura a lo mas MAXLEN;
  - meta: altura_max >= META_Y (centro de la bola, como el entorno);
  - prueba de meta al EXTRAER el nodo (requisito de optimalidad de A*).

Cierre de estados repetidos (opcional): la clave es la GEOMETRIA COMPLETA del
estado, no solo las posiciones:
    (frozenset de posiciones redondeadas,
     frozenset de aristas como pares no ordenados de posiciones redondeadas)
Dos estados con las mismas posiciones pero distintas uniones tienen
sucesores distintos (los filtros de cruce y holgura dependen de las
aristas), por eso las aristas van en la clave. La clave no depende de las
etiquetas de los nodos: dos ordenes de construccion que llevan a la misma
estructura se fusionan, lo que es correcto porque generar_acciones solo ve
geometria. El unico riesgo residual es el redondeo: dos estados que difieren
en menos de 10^-decimales px se tratan como uno y un filtro justo en su borde
podria aceptar en uno y rechazar en otro. Con decimales=6 la holgura de los
filtros (tol_alcance = 1e-6 px) es del mismo orden; el minimo se valida
ademas corriendo sin cierre (cerrar=False), que es exacto por construccion.
"""

import heapq
import itertools
import math
import time
from dataclasses import dataclass

from wog.geometria import (
    BOLAS_DISPONIBLES, MAXLEN, META_Y, REGLAS_T4, Estado, generar_acciones,
)


@dataclass
class ResultadoBusqueda:
    minimo: int | None               # bolas usadas; None si no hay solucion
    acciones: list                   # [(i, j, Li, Lj, lado, P), ...]
    alturas: list                    # altura tras cada accion
    expandidos: int
    generados: int
    segundos: float
    cerrar: bool


def clave(estado, decimales=6):
    """Clave geometrica del estado: posiciones y aristas redondeadas."""
    r = {n: (round(x, decimales) + 0.0, round(y, decimales) + 0.0)
         for n, (x, y) in estado.pos.items()}
    return (frozenset(r.values()),
            frozenset(frozenset((r[u], r[v])) for u, v in estado.G.edges))


def a_estrella(inicial=None, k=2, reglas=REGLAS_T4, meta=META_Y,
               cerrar=True, decimales=6, max_expandidos=None):
    """
    A* desde 'inicial' (por defecto el K4 ideal de Estado.inicial()).
    Devuelve ResultadoBusqueda con la secuencia optima de acciones.
    """
    if inicial is None:
        inicial = Estado.inicial(BOLAS_DISPONIBLES)

    def h(e):                     # = Estado.heuristica() con meta variable
        return max(0.0, (meta - e.altura_max()) / MAXLEN)

    t0 = time.perf_counter()
    orden = itertools.count()               # desempate FIFO estable
    abierta = [(h(inicial), 0, next(orden), inicial, ())]
    mejor_g = {clave(inicial, decimales): 0} if cerrar else None
    expandidos = generados = 0
    while abierta:
        f, g, _, e, camino = heapq.heappop(abierta)
        if cerrar and mejor_g.get(clave(e, decimales), math.inf) < g:
            continue                         # entrada obsoleta
        if e.altura_max() >= meta:
            alturas, x = [], inicial
            for acc in camino:
                x = x.aplicar(acc)
                alturas.append(x.altura_max())
            return ResultadoBusqueda(g, list(camino), alturas, expandidos,
                                     generados, time.perf_counter() - t0,
                                     cerrar)
        expandidos += 1
        if max_expandidos is not None and expandidos > max_expandidos:
            break
        for acc in generar_acciones(e, k, reglas)[0]:
            hijo = e.aplicar(acc)
            generados += 1
            if cerrar:
                c = clave(hijo, decimales)
                if mejor_g.get(c, math.inf) <= g + 1:
                    continue
                mejor_g[c] = g + 1
            heapq.heappush(abierta, (g + 1 + h(hijo), g + 1, next(orden),
                                     hijo, camino + (acc,)))
    return ResultadoBusqueda(None, [], [], expandidos, generados,
                             time.perf_counter() - t0, cerrar)
