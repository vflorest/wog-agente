"""
Politicas de referencia del Taller 4 para GooPhysicsEnv.

PoliticaT4 es determinista y usa SOLO el estado observado (G, pos, n) y la
prediccion geometrica de cada accion (la P de generar_acciones, expuesta por
env.predicciones()). No simula fisica para decidir.

Para cada accion valida a con prediccion P:
  e(s, a)  = esbeltez de pos U {P} (geometria.esbeltez, D1 por defecto)
  dh(s, a) = max(altura(s), P_y) - altura(s)   (altura ganada, >= 0)
  L(s)     = {a : P_y >= META_Y}  (centro de la bola, como el entorno)

| Condicion                                         | pi(s)                          |
|---------------------------------------------------|--------------------------------|
| L != {} y (alguna a en L con e <= e_max o n < 6)  | llegar: a en L de menor e      |
| L != {}, todas exceden e_max y n >= 6             | asegurar: a en A de menor e    |
| L == {} y alguna a con e <= e_max                 | avanzar: entre esas, mayor dh  |
| L == {} y ninguna con e <= e_max                  | ensanchar: a en A de menor e   |

Empates: mayor dh; si persiste, menor indice de accion.
n = bolas restantes ANTES de colocar (env.restantes).
"""

import numpy as np

from wog.geometria import META_Y, esbeltez

REGLAS = ("llegar", "asegurar", "avanzar", "ensanchar")


class PoliticaT4:
    """
    Politica determinista del Taller 4.

    Llamarla con el entorno devuelve (accion, regla). La regla tambien queda
    en self.ultima_regla.
    """

    def __init__(self, e_max, definicion="apoyo", meta=META_Y, n_min=6):
        self.e_max = e_max
        self.definicion = definicion
        self.meta = meta
        self.n_min = n_min
        self.ultima_regla = None

    def evaluar(self, pos, predicciones):
        """Lista de (a, P, e, dh) para cada accion predicha, por indice."""
        pts = list(pos.values())
        h = max(y for _, y in pts)
        return [(a, P, esbeltez(pts + [P], self.definicion),
                 max(h, P[1]) - h)
                for a, P in sorted(predicciones.items())]

    def decidir(self, pos, restantes, predicciones):
        """Nucleo puro: (accion, regla) a partir de (pos, n, {a: P})."""
        cand = self.evaluar(pos, predicciones)
        if not cand:
            raise ValueError("sin acciones validas")

        def menor_e(cs):
            return min(cs, key=lambda c: (c[2], -c[3], c[0]))

        llegan = [c for c in cand if c[1][1] >= self.meta]
        aptas = [c for c in cand if c[2] <= self.e_max]
        if llegan:
            if restantes < self.n_min or any(c[2] <= self.e_max
                                             for c in llegan):
                c, regla = menor_e(llegan), "llegar"
            else:
                c, regla = menor_e(cand), "asegurar"
        elif aptas:
            c = min(aptas, key=lambda c: (-c[3], c[0]))
            regla = "avanzar"
        else:
            c, regla = menor_e(cand), "ensanchar"
        return c[0], regla

    def __call__(self, env):
        a, regla = self.decidir(env.pos, env.restantes, env.predicciones())
        self.ultima_regla = regla
        return a, regla


class PoliticaAleatoria:
    """Uniforme sobre la mascara de acciones validas. Devuelve (a, "aleatoria")."""

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.ultima_regla = "aleatoria"

    def __call__(self, env):
        a = int(self.rng.choice(np.flatnonzero(env.action_masks())))
        return a, "aleatoria"
