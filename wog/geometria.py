"""
Geometria del nivel 1: primitivas, estado combinatorio y generacion de
acciones (extension de Henneberg tipo I).

Los tres generadores de Taller 2 (fase1_paso_a_paso.py, el notebook y
prueba_gymnasium.py) difieren en sus filtros. Aqui hay UN generador cuyos
filtros se activan con un objeto Reglas; cada variante es un preset.
"""

import itertools
import math
from dataclasses import dataclass

import networkx as nx

# --------------------------------------------------------------------------
# Parametros tomados de balls.xml de OpenGOO (assets originales de WoG), en px
# --------------------------------------------------------------------------
R_BOLA = 15.0      # shape="circle,30,0.25" -> diametro 30, radio 15
MINLEN = 100.0     # largo minimo de una union
MAXLEN = 200.0     # largo maximo de una union = alcance maximo por bola
L_REPOSO = (MINLEN + MAXLEN) / 2       # 150 px
BOLAS_DISPONIBLES = 14
META_Y = 600.0     # altura de la caneria (a medir del juego real)
Y_SUELO = 0.0      # altura del terreno (a medir del juego real)


# --------------------------------------------------------------------------
# Primitivas
# --------------------------------------------------------------------------
def dist(p, q):
    return math.dist(p, q)


def apice(A, B, L, lado):
    """
    Tercer vertice del triangulo isosceles apoyado en A-B, con lados de largo L.
    'lado' = +1 o -1 elige a que lado de la recta A-B queda la bola nueva.
    Devuelve None si A y B estan demasiado lejos para alcanzarlos con largo L.

    Matematicamente igual a interseccion(A, B, L, L, lado), pero NO bit a bit
    (difiere hasta ~1e-13). Se conserva para reproducir fase1_paso_a_paso.py.
    """
    d = dist(A, B)
    if d > 2 * L or d == 0:
        return None
    mx, my = (A[0] + B[0]) / 2, (A[1] + B[1]) / 2      # punto medio
    h = math.sqrt(L**2 - (d / 2) ** 2)                  # altura del triangulo
    ux, uy = (B[0] - A[0]) / d, (B[1] - A[1]) / d       # direccion A->B
    # perpendicular a (ux, uy) es (-uy, ux)
    return (mx - uy * h * lado, my + ux * h * lado)


def interseccion(A, B, La, Lb, lado):
    """
    Punto P tal que |P-A| = La y |P-B| = Lb.
    'lado' (+1/-1) elige cual de las dos soluciones (una a cada lado de A-B).
    Devuelve None si los circulos no se cortan.
    """
    d = dist(A, B)
    if d == 0 or d > La + Lb or d < abs(La - Lb):
        return None
    a = (La**2 - Lb**2 + d**2) / (2 * d)      # proyeccion sobre A-B
    r = La**2 - a**2
    if r < 0:
        return None
    h = math.sqrt(r)                           # distancia perpendicular
    ux, uy = (B[0] - A[0]) / d, (B[1] - A[1]) / d
    return (A[0] + a * ux - uy * h * lado, A[1] + a * uy + ux * h * lado)


def orient(a, b, c):
    """Signo del producto cruz (b-a) x (c-a): +1, -1 o 0 (colineales)."""
    v = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    return 0 if abs(v) < 1e-9 else (1 if v > 0 else -1)


def se_cruzan(p1, p2, p3, p4, estricto=False):
    """
    Interseccion entre los segmentos p1-p2 y p3-p4. Si comparten un extremo
    no cuenta como cruce.

    estricto=False (fase1 y notebook): compara signos de orientacion tal cual;
        una orientacion 0 (colinealidad) puede contar como cruce.
    estricto=True (prueba_gymnasium): exige las cuatro orientaciones no nulas;
        todo caso con colinealidad se declara "no cruza".

    Las dos versiones NO estan unificadas a proposito (ver CLAUDE.md,
    Bugs conocidos).
    """
    if p1 in (p3, p4) or p2 in (p3, p4):   # comparten extremo: no es cruce
        return False
    o1, o2 = orient(p1, p2, p3), orient(p1, p2, p4)
    o3, o4 = orient(p3, p4, p1), orient(p3, p4, p2)
    if estricto:
        return 0 not in (o1, o2, o3, o4) and o1 != o2 and o3 != o4
    return o1 != o2 and o3 != o4


def dist_segmento(P, A, B):
    """Distancia del punto P al segmento A-B."""
    dx, dy = B[0] - A[0], B[1] - A[1]
    L2 = dx * dx + dy * dy
    if L2 == 0:
        return dist(P, A)
    t = max(0.0, min(1.0, ((P[0] - A[0]) * dx + (P[1] - A[1]) * dy) / L2))
    return dist(P, (A[0] + t * dx, A[1] + t * dy))


def holgura_suficiente(P, i, j, pos, aristas, minimo=R_BOLA):
    """
    Holgura entre la bola nueva P (anclada a i y j) y la estructura:
      - P queda a >= minimo de toda union existente;
      - cada union nueva (P-i, P-j) queda a >= minimo de toda bola
        existente distinta de sus dos extremos.
    Cubre los casos que se_cruzan no detecta: contacto en T, segmentos
    colineales superpuestos y casi colineales.
    """
    if any(dist_segmento(P, pos[u], pos[v]) < minimo for u, v in aristas):
        return False
    for ancla in (i, j):
        if any(dist_segmento(pos[m], P, pos[ancla]) < minimo
               for m in pos if m != ancla):
            return False
    return True


def largos(k):
    """k valores de largo de union, equiespaciados en [MINLEN, MAXLEN]."""
    if k <= 1:
        return [L_REPOSO]
    paso = (MAXLEN - MINLEN) / (k - 1)
    return [MINLEN + i * paso for i in range(k)]


def envolvente(puntos):
    """Envolvente convexa (cadena monotona de Andrew), sentido antihorario."""
    pts = sorted(set(puntos))
    if len(pts) < 3:
        return pts

    def media(sec):
        out = []
        for p in sec:
            while len(out) >= 2 and orient(out[-2], out[-1], p) <= 0:
                out.pop()
            out.append(p)
        return out[:-1]

    return media(pts) + media(reversed(pts))


def dentro(P, poly):
    """Punto en poligono por paridad de cruces (ray casting)."""
    if len(poly) < 3:
        return False
    adentro = False
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        if ((y1 > P[1]) != (y2 > P[1])) and \
           (P[0] < (x2 - x1) * (P[1] - y1) / (y2 - y1 + 1e-15) + x1):
            adentro = not adentro
    return adentro


# --------------------------------------------------------------------------
# Estado combinatorio-geometrico
# --------------------------------------------------------------------------
class Estado:
    def __init__(self, G, pos, restantes):
        self.G = G
        self.pos = pos
        self.restantes = restantes

    @staticmethod
    def inicial(restantes=BOLAS_DISPONIBLES, base="K4"):
        """
        base="K4"        -> cuadrado con diagonales (juego original, 6 uniones)
        base="triangulo" -> triangulo (nivel de prueba del clon OpenGOO)
        """
        lado = L_REPOSO
        if base == "K4":
            pos = {0: (0.0, 0.0), 1: (lado, 0.0),
                   2: (lado, lado), 3: (0.0, lado)}
            G = nx.complete_graph(4)
        else:
            h = lado * math.sqrt(3) / 2
            pos = {0: (0.0, 0.0), 1: (lado, 0.0), 2: (lado / 2, h)}
            G = nx.complete_graph(3)
        return Estado(G, pos, restantes)

    def altura_max(self):
        return max(p[1] for p in self.pos.values())

    def laman(self):
        """Devuelve (V, E, 2V-3, exceso). Exceso >= 0 => rigido."""
        V, E = self.G.number_of_nodes(), self.G.number_of_edges()
        return V, E, 2 * V - 3, E - (2 * V - 3)

    def es_planar(self):
        return nx.check_planarity(self.G)[0]

    def heuristica(self):
        """h(n) en unidades de BOLAS, no de pixeles. Admisible."""
        return max(0.0, (META_Y - self.altura_max()) / MAXLEN)

    def aplicar(self, accion):
        """
        Extension de Henneberg tipo I: +1 nodo, +2 aristas.
        Acepta (i, j, lado, P) o (i, j, Li, Lj, lado, P).
        """
        i, j, P = accion[0], accion[1], accion[-1]
        n = max(self.G.nodes) + 1
        G = self.G.copy()
        G.add_node(n)
        G.add_edge(n, i)
        G.add_edge(n, j)
        pos = dict(self.pos)
        pos[n] = P
        return Estado(G, pos, self.restantes - 1)


# --------------------------------------------------------------------------
# Generacion de acciones
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Reglas:
    """
    Filtros del generador de acciones. Orden de aplicacion: alcance
    geometrico (P existe), suelo, alcance [MINLEN, MAXLEN], solapamiento,
    cruce, holgura, envolvente, duplicado y, al final, tope.
    """
    suelo: bool
    alcance: bool
    cruce_estricto: bool
    envolvente: bool
    tol_duplicado: float | None   # None = sin descarte de duplicados
    max_acciones: int | None      # None = sin tope
    usar_apice: bool              # isosceles con apice() en vez de interseccion()
    holgura: bool = False         # distancia >= R_BOLA entre bolas y uniones


# fase1_paso_a_paso.py: isosceles, sin suelo (ver Bugs conocidos)
REGLAS_FASE1 = Reglas(suelo=False, alcance=True, cruce_estricto=False,
                      envolvente=False, tol_duplicado=None,
                      max_acciones=None, usar_apice=True)

# Fase1_World_of_Goo.ipynb (el notebook conserva su propio codigo; este
# preset solo documenta sus reglas)
REGLAS_NOTEBOOK = Reglas(suelo=True, alcance=True, cruce_estricto=False,
                         envolvente=False, tol_duplicado=2.0,
                         max_acciones=None, usar_apice=False)

# prueba_gymnasium.py
REGLAS_GYM_T2 = Reglas(suelo=True, alcance=False, cruce_estricto=True,
                       envolvente=True, tol_duplicado=2.0,
                       max_acciones=256, usar_apice=False)

# Entorno del Taller 4. La accion es (i, j, Li, Lj, lado): dos acciones con
# la misma posicion pero distintos anclajes dan grafos (y tensiones)
# distintos, por eso NO hay descarte de duplicados. Sin tope: la
# codificacion de acciones sera estable. Cruce permisivo + holgura: la
# holgura descarta los solapamientos colineales que se_cruzan no ve.
REGLAS_T4 = Reglas(suelo=True, alcance=True, cruce_estricto=False,
                   envolvente=True, tol_duplicado=None,
                   max_acciones=None, usar_apice=False, holgura=True)


def generar_acciones(estado, k=1, reglas=REGLAS_FASE1):
    """
    Toda accion es anclar una bola nueva a un par (i, j) existente con
    largos (Li, Lj) a un lado (+1/-1) de la recta i-j.

    Devuelve (acciones, descartes):
      acciones  : lista de (i, j, Li, Lj, lado, P)
      descartes : conteo por filtro
    """
    descartes = {"alcance": 0, "suelo": 0, "solapa": 0, "cruce": 0,
                 "holgura": 0, "envolvente": 0, "duplicado": 0}
    if estado.restantes == 0:
        return [], descartes

    pos = estado.pos
    LS = largos(k)
    borde = envolvente(pos.values()) if reglas.envolvente else None
    validas, vistos = [], set()

    for (i, j) in itertools.combinations(estado.G.nodes, 2):
        for Li, Lj, lado in itertools.product(LS, LS, (+1, -1)):
            if reglas.usar_apice and Li == Lj:
                P = apice(pos[i], pos[j], Li, lado)
            else:
                P = interseccion(pos[i], pos[j], Li, Lj, lado)
            if P is None:
                descartes["alcance"] += 1
                continue

            if reglas.suelo and P[1] < Y_SUELO + R_BOLA:
                descartes["suelo"] += 1
                continue

            if reglas.alcance and not all(
                    MINLEN <= dist(P, pos[m]) <= MAXLEN for m in (i, j)):
                descartes["alcance"] += 1
                continue

            if any(dist(P, q) < 2 * R_BOLA for q in pos.values()):
                descartes["solapa"] += 1
                continue

            nuevas = [(P, pos[i]), (P, pos[j])]
            if any(se_cruzan(a, b, pos[u], pos[v], reglas.cruce_estricto)
                   for a, b in nuevas for u, v in estado.G.edges):
                descartes["cruce"] += 1
                continue

            if reglas.holgura and not holgura_suficiente(
                    P, i, j, pos, estado.G.edges):
                descartes["holgura"] += 1
                continue

            if borde is not None and dentro(P, borde):
                descartes["envolvente"] += 1
                continue

            if reglas.tol_duplicado is not None:
                tol = reglas.tol_duplicado
                clave = (i, j, round(P[0] / tol), round(P[1] / tol))
                if clave in vistos:
                    descartes["duplicado"] += 1
                    continue
                vistos.add(clave)

            validas.append((i, j, Li, Lj, lado, P))

    if reglas.max_acciones is not None:
        validas = validas[:reglas.max_acciones]
    return validas, descartes
