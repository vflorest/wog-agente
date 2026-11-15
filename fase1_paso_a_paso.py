"""
World of Goo - Nivel 1
Modelo combinatorio-geometrico (Fase 1: sin fisica)

Version narrada paso a paso.
Ejecutar con:  python3 fase1_paso_a_paso.py
Para ir mas rapido:  python3 fase1_paso_a_paso.py --rapido
"""

import math
import itertools
import sys
import time

import networkx as nx

# --------------------------------------------------------------------------
# Parametros tomados de balls.xml de OpenGOO (assets originales de WoG)
# --------------------------------------------------------------------------
R_BOLA = 15.0      # shape="circle,30,0.25" -> diametro 30, radio 15
MINLEN = 100.0     # largo minimo de una union
MAXLEN = 200.0     # largo maximo de una union = alcance maximo por bola
L_REPOSO = (MINLEN + MAXLEN) / 2       # 150 px
BOLAS_DISPONIBLES = 14
META_Y = 600.0     # altura de la caneria (a medir del juego real)

PAUSA = 0.0 if "--rapido" in sys.argv else 1.4


def narrar(texto="", pausa=None):
    """Imprime y espera, para poder seguir la ejecucion."""
    print(texto)
    sys.stdout.flush()
    time.sleep(PAUSA if pausa is None else pausa)


def titulo(texto):
    narrar("\n" + "=" * 68, 0)
    narrar("  " + texto, 0)
    narrar("=" * 68)


# --------------------------------------------------------------------------
# Geometria
# --------------------------------------------------------------------------
def dist(p, q):
    return math.dist(p, q)


def apice(A, B, L, lado):
    """
    Tercer vertice del triangulo isosceles apoyado en A-B, con lados de largo L.
    'lado' = +1 o -1 elige a que lado de la recta A-B queda la bola nueva.
    Devuelve None si A y B estan demasiado lejos para alcanzarlos con largo L.
    """
    d = dist(A, B)
    if d > 2 * L or d == 0:
        return None
    mx, my = (A[0] + B[0]) / 2, (A[1] + B[1]) / 2      # punto medio
    h = math.sqrt(L**2 - (d / 2) ** 2)                  # altura del triangulo
    ux, uy = (B[0] - A[0]) / d, (B[1] - A[1]) / d       # direccion A->B
    # perpendicular a (ux, uy) es (-uy, ux)
    return (mx - uy * h * lado, my + ux * h * lado)


def se_cruzan(p1, p2, p3, p4):
    """Interseccion propia entre los segmentos p1-p2 y p3-p4."""
    def orient(a, b, c):
        v = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        return 0 if abs(v) < 1e-9 else (1 if v > 0 else -1)

    if p1 in (p3, p4) or p2 in (p3, p4):   # comparten extremo: no es cruce
        return False
    return (orient(p1, p2, p3) != orient(p1, p2, p4) and
            orient(p3, p4, p1) != orient(p3, p4, p2))


# --------------------------------------------------------------------------
# Estado
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

    # ---- consultas -------------------------------------------------------
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

    # ---- acciones --------------------------------------------------------
    def acciones(self, explicar=False):
        """
        Toda accion es: anclar una bola nueva a un par (i, j) existente.
        Se aplican tres filtros de validez, en orden de costo creciente.
        """
        if self.restantes == 0:
            return []

        validas = []
        descartes = {"alcance": 0, "solapa": 0, "cruce": 0}

        candidatos = itertools.product(
            itertools.combinations(self.G.nodes, 2), (+1, -1))

        for (i, j), lado in candidatos:
            P = apice(self.pos[i], self.pos[j], L_REPOSO, lado)
            if P is None:
                descartes["alcance"] += 1
                continue

            # Filtro 1: las dos uniones deben caer dentro de [MINLEN, MAXLEN]
            if not all(MINLEN <= dist(P, self.pos[k]) <= MAXLEN
                       for k in (i, j)):
                descartes["alcance"] += 1
                continue

            # Filtro 2: la bola nueva no puede solaparse con una existente
            if any(dist(P, q) < 2 * R_BOLA for q in self.pos.values()):
                descartes["solapa"] += 1
                continue

            # Filtro 3: las uniones nuevas no pueden cruzar uniones existentes
            #           (planaridad geometrica del embedding)
            nuevas = [(P, self.pos[i]), (P, self.pos[j])]
            if any(se_cruzan(a, b, self.pos[u], self.pos[v])
                   for a, b in nuevas for u, v in self.G.edges):
                descartes["cruce"] += 1
                continue

            validas.append((i, j, lado, P))

        if explicar:
            total = len(validas) + sum(descartes.values())
            narrar(f"    candidatos brutos (pares x 2 lados) : {total}")
            narrar(f"    descartados por alcance             : {descartes['alcance']}")
            narrar(f"    descartados por solapamiento        : {descartes['solapa']}")
            narrar(f"    descartados por cruce de uniones    : {descartes['cruce']}")
            narrar(f"    ACCIONES VALIDAS                    : {len(validas)}")

        return validas

    def aplicar(self, accion):
        """Extension de Henneberg tipo I: +1 nodo, +2 aristas."""
        i, j, lado, P = accion
        n = max(self.G.nodes) + 1
        G = self.G.copy()
        G.add_node(n)
        G.add_edge(n, i)
        G.add_edge(n, j)
        pos = dict(self.pos)
        pos[n] = P
        return Estado(G, pos, self.restantes - 1)

    def describir(self):
        V, E, laman, exc = self.laman()
        narrar(f"    bolas colocadas : {V}      uniones: {E}", 0)
        narrar(f"    bolas restantes : {self.restantes}", 0)
        narrar(f"    altura maxima   : {self.altura_max():.1f} px "
               f"(meta {META_Y:.0f} px)", 0)
        narrar(f"    conteo de Laman : E={E}, 2V-3={laman}, exceso={exc} "
               f"-> {'rigida' if exc >= 0 else 'MECANISMO'}", 0)
        narrar(f"    grafo planar    : {self.es_planar()}", 0)
        narrar(f"    h(n)            : {self.heuristica():.2f} bolas")


# ==========================================================================
# DEMO
# ==========================================================================
def main():
    titulo("PASO 0 - PARAMETROS DEL MODELO")
    narrar("Tomados de balls.xml (assets originales de World of Goo):", 0)
    narrar(f"    radio de bola        : {R_BOLA:.0f} px", 0)
    narrar(f"    union: largo minimo  : {MINLEN:.0f} px", 0)
    narrar(f"    union: largo maximo  : {MAXLEN:.0f} px  <- alcance por bola", 0)
    narrar(f"    largo de reposo      : {L_REPOSO:.0f} px", 0)
    narrar(f"    bolas disponibles    : {BOLAS_DISPONIBLES}")

    titulo("PASO 1 - ESTADO INICIAL")
    narrar("El nivel 1 arranca con 4 bolas, todas conectadas entre si.", 0)
    narrar("Eso es un grafo completo K4: C(4,2) = 6 uniones.", 0)
    narrar("Los 4 lados del cuadrado, mas las 2 diagonales.\n", 0)
    e = Estado.inicial()
    e.describir()
    narrar("\n  El exceso de Laman vale 1: el cuadrado sin diagonales", 0)
    narrar("  seria un mecanismo (se deforma en rombo). Con las dos", 0)
    narrar("  diagonales queda rigido y con una union de sobra.")

    titulo("PASO 2 - GENERACION DE ACCIONES")
    narrar("Una accion es: elegir un par de bolas ya colocadas y anclar", 0)
    narrar("una bola nueva formando triangulo. El 'lado' (+1/-1) elige a", 0)
    narrar("que lado de ese par queda la bola nueva.\n", 0)
    narrar("Se filtra en tres etapas, de mas barata a mas cara:\n", 0)
    acc = e.acciones(explicar=True)

    titulo("PASO 3 - INSPECCION DE LAS ACCIONES VALIDAS")
    for k, (i, j, lado, P) in enumerate(acc):
        narrar(f"    accion {k}: anclar en ({i},{j}) lado {lado:+d} "
               f"-> posicion ({P[0]:7.1f}, {P[1]:7.1f})", PAUSA / 2)

    titulo("PASO 4 - APLICAR UNA ACCION")
    elegida = max(acc, key=lambda a: a[3][1])     # la que mas altura gana
    i, j, lado, P = elegida
    narrar(f"Elegimos la que mas altura gana: ({i},{j}) lado {lado:+d}\n", 0)
    e = e.aplicar(elegida)
    e.describir()
    narrar("\n  Se agrego 1 nodo y 2 aristas: extension de Henneberg tipo I.", 0)
    narrar("  El exceso de Laman NO cambio. La rigidez se preserva.")

    titulo("PASO 5 - CONSTRUCCION GREEDY (heuristica: ganar altura)")
    narrar("Repetimos eligiendo siempre la accion que sube mas.\n", 0)
    narrar("paso  accion         altura  ramif.  V   E   2V-3 exc  h(n)", 0)
    narrar("-" * 62, 0)
    for paso in range(2, 9):
        accs = e.acciones()
        if not accs:
            narrar("  sin acciones validas")
            break
        ramif = len(accs)
        mejor = max(accs, key=lambda a: a[3][1])
        e = e.aplicar(mejor)
        i, j, lado, _ = mejor
        V, E, lam, exc = e.laman()
        narrar(f"{paso:^4}  ({i:>2},{j:>2}) lado{lado:+d}  {e.altura_max():6.1f}  "
               f"{ramif:^6}  {V:<3} {E:<3} {lam:<4} {exc:^3}  "
               f"{e.heuristica():.2f}", PAUSA / 2)

    titulo("PASO 6 - LECTURA DE LOS RESULTADOS")
    usadas = BOLAS_DISPONIBLES - e.restantes
    narrar(f"bolas usadas: {usadas}   sobrantes: {e.restantes}\n", 0)
    narrar("Tres cosas que se ven en la tabla:\n", 0)
    narrar("  1. La ramificacion CRECE con la profundidad. Cada bola nueva", 0)
    narrar("     habilita nuevos pares de anclaje. Por eso el arbol de", 0)
    narrar("     busqueda se ensancha en vez de angostarse.\n", 0)
    narrar("  2. El exceso de Laman se mantiene constante. La estructura", 0)
    narrar("     es siempre rigida por construccion, asi que el colapso", 0)
    narrar("     (con fisica) sera falla de resistencia, no de rigidez.\n", 0)
    narrar("  3. h(n) baja mas lento de lo que sugiere el alcance maximo:", 0)
    narrar("     el greedy gana altura cada DOS bolas, no cada una. La", 0)
    narrar("     heuristica subestima el costo real, o sea es ADMISIBLE,", 0)
    narrar("     que es lo que A* necesita para garantizar optimalidad.\n", 0)
    narrar("  Y notar que el greedy construye un zigzag fino y vertical:", 0)
    narrar("  justo el tipo de estructura que colapsaria con fisica activa.", 0)
    narrar("  La heuristica de altura es enganosa, como anticipamos.")

    titulo("PASO 7 - LA MISMA BASE, PERO TRIANGULAR")
    narrar("El nivel de prueba del clon OpenGOO arranca con un triangulo.", 0)
    narrar("El modelo no esta cableado a un caso particular:\n", 0)
    t = Estado.inicial(base="triangulo")
    t.describir()
    narrar("\n  Exceso 0: el triangulo es MINIMAMENTE rigido (sin redundancia).", 0)
    narrar("  El K4 tiene exceso 1: tolera perder una union.", 0)
    narrar("  En ambos casos Henneberg preserva la rigidez.")

    narrar("\n" + "=" * 68, 0)
    narrar("  FIN", 0)
    narrar("=" * 68, 0)


if __name__ == "__main__":
    main()
