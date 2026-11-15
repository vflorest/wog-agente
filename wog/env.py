"""
Entorno Gymnasium del Taller 4: GooPhysicsEnv.

MDP (CLAUDE.md):
  s = (G, pos, n); un paso = colocar bola + estabilizar; carga movil W
  sorteada cada turno, fuera del estado; recompensa n si llega con n >= 4,
  si no 0; shaping opcional F = Phi(s') - Phi(s) con Phi = altura / META_Y
  y Phi = 0 en terminales; gamma = 1, horizonte 14.

Reutiliza wog.geometria (generar_acciones con REGLAS_T4) y wog.fisica
(colocar_bola, unir, estabilizar con ruptura). La construccion es
incremental: cada step agrega una bola al espacio ya simulado.

Codificacion de acciones ESTABLE: la tupla (i, j, Li, Lj, lado) con i < j
tiene un indice fijo durante todo el episodio,
    a = ((par(i, j) * k + iLi) * k + iLj) * 2 + ilado
con par(i, j) el indice lexicografico de (i, j) en combinations(range(18), 2),
iL el indice del largo en largos(k) e ilado = 0 si lado = +1, 1 si -1.
Con k = 2 son C(18, 2) * 2 * 2 * 2 = 1224 indices. La mascara marca los
validos segun REGLAS_T4; nunca se usa la posicion en la lista de validas.

Unidades: G, pos, largos y alturas en px; el espacio de pymunk en u
(px = u / K_ESCALA), mismo origen, suelo en y = 0. La altura es la del
CENTRO de la bola mas alta (misma convencion que Estado.altura_max).
"""

import copy
import itertools
import math
from dataclasses import dataclass

import gymnasium as gym
import networkx as nx
import numpy as np
from gymnasium import spaces

from wog import fisica
from wog.fisica import K_ESCALA, colocar_bola, crear_espacio, estabilizar, unir
from wog.geometria import (
    BOLAS_DISPONIBLES, META_Y, REGLAS_T4, Estado, generar_acciones, largos,
)

MAX_NODOS = 4 + BOLAS_DISPONIBLES                     # 18
PARES = list(itertools.combinations(range(MAX_NODOS), 2))
INDICE_PAR = {par: n for n, par in enumerate(PARES)}
NODOS_FIJOS = (0, 1)                                  # base K4 anclada al suelo

CAUSAS = (None, "colapso", "meta", "meta_sin_bolas", "sin_bolas",
          "sin_acciones", "horizonte")


@dataclass(frozen=True)
class Theta:
    """Parametros fisicos theta: fijos durante el episodio y desconocidos
    para el agente. Por defecto, los de wog/fisica.py."""
    rigidez: float = fisica.K_RESORTE
    damping: float = fisica.C_DAMP
    masa: float = fisica.MASA
    umbral: float = fisica.MAXFORCE


def esbeltez(G, pos):
    """Stub: la definicion de esbeltez se decide despues."""
    return float("nan")


class GooPhysicsEnv(gym.Env):
    """
    Nivel 1 de World of Goo con fisica Pymunk y ruptura por tension.

    Parametros
    ----------
    k : cantidad de largos de union (largos(k) en [MINLEN, MAXLEN]).
    theta : Theta con rigidez, damping, masa y umbral de ruptura.
    p_carga : probabilidad de que cada bola no usada camine sobre la
        estructura en un turno: m ~ Binomial(n, p_carga).
        VALOR PROVISIONAL (0.2): pendiente de calibrar.
    modelo_carga : "grado" (por defecto) reparte las m bolas en nodos
        dinamicos con probabilidad proporcional al grado (distribucion
        estacionaria de una caminata aleatoria); "uniforme" con la misma
        probabilidad para todos.
    shaping : si True suma F = Phi(s') - Phi(s), Phi = altura / META_Y,
        Phi = 0 en terminales.
    """

    metadata = {"render_modes": []}

    def __init__(self, k=2, theta=None, p_carga=0.2, modelo_carga="grado",
                 shaping=False, presupuesto=BOLAS_DISPONIBLES):
        super().__init__()
        if modelo_carga not in ("grado", "uniforme"):
            raise ValueError(f"modelo_carga desconocido: {modelo_carga!r}")
        self.k = k
        self.theta = theta if theta is not None else Theta()
        self.p_carga = p_carga
        self.modelo_carga = modelo_carga
        self.shaping = shaping
        self.presupuesto = presupuesto
        self.largos = largos(k)

        # tabla fija indice -> (i, j, Li, Lj, lado)
        self.tabla = [(i, j, Li, Lj, lado)
                      for (i, j) in PARES
                      for Li, Lj, lado in itertools.product(
                          self.largos, self.largos, (+1, -1))]
        self.action_space = spaces.Discrete(len(self.tabla))
        self.observation_space = spaces.Dict({
            "posiciones": spaces.Box(-1e4, 1e4, (MAX_NODOS, 2), np.float32),
            "nodos":      spaces.MultiBinary(MAX_NODOS),
            "adyacencia": spaces.MultiBinary(MAX_NODOS * MAX_NODOS),
            "largos":     spaces.Box(0, 1e4, (MAX_NODOS * MAX_NODOS,),
                                     np.float32),
            "restantes":  spaces.Discrete(presupuesto + 1),
        })

    # ------------------------------------------------------------------
    # Codificacion de acciones
    # ------------------------------------------------------------------
    def codificar(self, i, j, Li, Lj, lado):
        iLi, iLj = self.largos.index(Li), self.largos.index(Lj)
        return ((INDICE_PAR[(i, j)] * self.k + iLi) * self.k + iLj) * 2 \
            + (0 if lado == +1 else 1)

    def decodificar(self, a):
        return self.tabla[a]

    def action_masks(self):
        """Vector booleano de acciones validas (lo consume MaskablePPO)."""
        return self._mascara.copy()

    # ------------------------------------------------------------------
    # Ciclo estandar
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        e0 = Estado.inicial(self.presupuesto)
        self.space = crear_espacio()
        self.cuerpos = {n: colocar_bola(self.space, self._a_u(p),
                                        fija=n in NODOS_FIJOS,
                                        masa=self.theta.masa)
                        for n, p in e0.pos.items()}
        self.uniones = []
        self.G = nx.Graph()
        self.G.add_nodes_from(e0.G.nodes)
        for u, v in e0.G.edges:
            self._unir(u, v, math.dist(e0.pos[u], e0.pos[v]))
        r = estabilizar(self.space, self.uniones, self.theta.umbral)
        if r.rotas:
            raise RuntimeError("la base K4 se rompe sin carga con este theta")
        self.restantes = self.presupuesto
        self.t = 0
        self.terminado = False
        self._leer_pos()
        self._generar()
        return self._obs(), {"altura": self.altura(),
                             "tension_max": r.pico,
                             "esbeltez": esbeltez(self.G, self.pos),
                             "acciones_validas": int(self._mascara.sum()),
                             "causa": None}

    def step(self, a):
        if self.terminado:
            raise RuntimeError("step() tras un estado terminal; llame a reset()")
        a = int(a)
        if not self._mascara[a]:
            raise ValueError(f"accion {a} {self.tabla[a]} no valida en este estado")
        phi = self._phi()
        i, j, Li, Lj, lado = self.tabla[a]

        # 1. colocar la bola en el espacio ya simulado
        nuevo = self.G.number_of_nodes()
        self.cuerpos[nuevo] = colocar_bola(self.space, self._a_u(self._P[a]),
                                           masa=self.theta.masa)
        self.G.add_node(nuevo)
        self._unir(nuevo, i, Li)
        self._unir(nuevo, j, Lj)
        self.restantes -= 1
        self.t += 1

        # 2. estabilizar (una ruptura aqui ya es colapso)
        est = estabilizar(self.space, self.uniones, self.theta.umbral)
        self._leer_pos()

        # 3-4. carga W y rupturas
        m, pico_w, rotas_w = 0, 0.0, []
        if not est.rotas:
            m, pico_w, rotas_w = self.probar_carga()
        rotas = len(est.rotas) + len(rotas_w)

        # 5-6. terminos
        altura = self.altura()
        r, causa = 0.0, None
        if rotas:
            causa = "colapso"
        elif altura >= META_Y:
            if self.restantes >= 4:
                r, causa = float(self.restantes), "meta"
            else:
                causa = "meta_sin_bolas"
        elif self.restantes == 0:
            causa = "sin_bolas"
        else:
            self._generar()
            if not self._mascara.any():
                causa = "sin_acciones"
            elif self.t >= self.presupuesto:
                causa = "horizonte"
        self.terminado = causa is not None
        if self.terminado:
            self._mascara[:] = False

        if self.shaping:
            r += (0.0 if self.terminado else self._phi()) - phi

        info = {"accion": (i, j, Li, Lj, lado),
                "altura": altura,
                "m_carga": m,
                "tension_max": max(est.pico, pico_w),
                "tension_max_sin_carga": est.pico,
                "rotas": rotas,
                "colapso_por": ("propio" if est.rotas else
                                "carga" if rotas_w else None),
                "estable": est.estable,
                "esbeltez": esbeltez(self.G, self.pos),
                "acciones_validas": int(self._mascara.sum()),
                "causa": causa}
        return self._obs(), r, self.terminado, False, info

    # ------------------------------------------------------------------
    # Carga movil W (fuera del estado)
    # ------------------------------------------------------------------
    def probar_carga(self):
        """
        Sortea W con self.np_random y la aplica sobre una COPIA del espacio:
        m ~ Binomial(n, p_carga) bolas repartidas en nodos dinamicos (por
        grado o uniforme); cada una suma su masa al cuerpo del nodo. Se
        estabiliza la copia con ruptura y se descarta: el estado no cambia.
        Devuelve (m, tension pico, rotas).
        """
        m = int(self.np_random.binomial(self.restantes, self.p_carga))
        if m == 0:
            return 0, 0.0, []
        dinamicos = [n for n in self.G.nodes if n not in NODOS_FIJOS]
        if self.modelo_carga == "grado":
            w = np.array([self.G.degree(n) for n in dinamicos], float)
            prob = w / w.sum()
        else:
            prob = None
        donde = self.np_random.choice(dinamicos, size=m, p=prob)
        space, cuerpos, uniones = copy.deepcopy(
            (self.space, self.cuerpos, self.uniones))
        for n in donde:
            cuerpos[int(n)].mass += self.theta.masa
        r = estabilizar(space, uniones, self.theta.umbral)
        return m, r.pico, r.rotas

    # ------------------------------------------------------------------
    # Clonacion para busqueda en arbol
    # ------------------------------------------------------------------
    def clone_state(self):
        """
        Snapshot completo. El espacio de pymunk se copia con UN solo
        deepcopy de (espacio, cuerpos, uniones) para que el memo conserve
        las referencias cruzadas; la continuacion es identica bit a bit.
        Incluye el estado del generador aleatorio (W reproducible).
        """
        return {"fis": copy.deepcopy((self.space, self.cuerpos, self.uniones)),
                "G": self.G.copy(), "pos": dict(self.pos),
                "restantes": self.restantes, "t": self.t,
                "terminado": self.terminado,
                "mascara": self._mascara.copy(), "P": dict(self._P),
                "rng": copy.deepcopy(self.np_random.bit_generator.state)}

    def restore_state(self, snap):
        """Restaura un snapshot; se puede restaurar el mismo varias veces."""
        self.space, self.cuerpos, self.uniones = copy.deepcopy(snap["fis"])
        self.G = snap["G"].copy()
        self.pos = dict(snap["pos"])
        self.restantes, self.t = snap["restantes"], snap["t"]
        self.terminado = snap["terminado"]
        self._mascara = snap["mascara"].copy()
        self._P = dict(snap["P"])
        self.np_random.bit_generator.state = copy.deepcopy(snap["rng"])

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------
    def altura(self):
        """Altura del centro de la bola mas alta (px), como Estado.altura_max."""
        return max(p[1] for p in self.pos.values())

    def _phi(self):
        return self.altura() / META_Y

    @staticmethod
    def _a_u(p):
        return (p[0] * K_ESCALA, p[1] * K_ESCALA)

    def _unir(self, u, v, largo):
        """Arista en G con su largo de reposo y resorte en pymunk."""
        self.G.add_edge(u, v, largo=largo)
        j = unir(self.space, self.cuerpos[u], self.cuerpos[v],
                 self.theta.rigidez, self.theta.damping)
        if j is not None:
            self.uniones.append(j)

    def _leer_pos(self):
        self.pos = {n: (c.position.x / K_ESCALA, c.position.y / K_ESCALA)
                    for n, c in self.cuerpos.items()}

    def _generar(self):
        """Mascara y puntos P a partir de generar_acciones(REGLAS_T4)."""
        e = Estado(self.G, self.pos, self.restantes)
        acciones, _ = generar_acciones(e, self.k, REGLAS_T4)
        self._mascara = np.zeros(self.action_space.n, dtype=bool)
        self._P = {}
        for i, j, Li, Lj, lado, P in acciones:
            a = self.codificar(i, j, Li, Lj, lado)
            self._mascara[a] = True
            self._P[a] = P

    def _obs(self):
        pos = np.zeros((MAX_NODOS, 2), np.float32)
        nodos = np.zeros(MAX_NODOS, np.int8)
        ady = np.zeros((MAX_NODOS, MAX_NODOS), np.int8)
        lar = np.zeros((MAX_NODOS, MAX_NODOS), np.float32)
        for n, p in self.pos.items():
            pos[n] = p
            nodos[n] = 1
        for u, v, L in self.G.edges(data="largo"):
            ady[u, v] = ady[v, u] = 1
            lar[u, v] = lar[v, u] = L
        return {"posiciones": pos, "nodos": nodos,
                "adyacencia": ady.flatten(), "largos": lar.flatten(),
                "restantes": self.restantes}
