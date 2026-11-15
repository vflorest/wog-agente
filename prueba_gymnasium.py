"""
Prueba minima de Gymnasium - Taller 2, pregunta 4.

Verifica que se puede observar el estado del ambiente y ejecutar acciones
programaticamente, envolviendo el modelo de Fase 1 en la API estandar.

Muestra ademas el problema del espacio de acciones variable y su solucion
por enmascaramiento, que fue la objecion planteada por el profesor.

Ejecutar:  python3 prueba_gymnasium.py
"""

import math
import itertools

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import networkx as nx

# --------------------------------------------------------------------------
R_BOLA, MINLEN, MAXLEN = 15.0, 100.0, 200.0
L_REPOSO = (MINLEN + MAXLEN) / 2
BOLAS_DISPONIBLES, META_Y, Y_SUELO = 14, 600.0, 0.0
MAX_BOLAS = 4 + BOLAS_DISPONIBLES          # cota superior de nodos
MAX_ACCIONES = 256                          # cota superior del espacio de acciones


def dist(p, q):
    return math.dist(p, q)


def interseccion(A, B, La, Lb, lado):
    d = dist(A, B)
    if d == 0 or d > La + Lb or d < abs(La - Lb):
        return None
    a = (La**2 - Lb**2 + d**2) / (2*d)
    r = La**2 - a**2
    if r < 0:
        return None
    h = math.sqrt(r)
    ux, uy = (B[0]-A[0])/d, (B[1]-A[1])/d
    return (A[0] + a*ux - uy*h*lado, A[1] + a*uy + ux*h*lado)


def orient(a, b, c):
    v = (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
    return 0 if abs(v) < 1e-9 else (1 if v > 0 else -1)


def se_cruzan(p1, p2, p3, p4):
    if p1 in (p3, p4) or p2 in (p3, p4):
        return False
    o1, o2 = orient(p1,p2,p3), orient(p1,p2,p4)
    o3, o4 = orient(p3,p4,p1), orient(p3,p4,p2)
    return 0 not in (o1,o2,o3,o4) and o1 != o2 and o3 != o4


def largos(k):
    return [L_REPOSO] if k <= 1 else \
           [MINLEN + i*(MAXLEN-MINLEN)/(k-1) for i in range(k)]


# ==========================================================================
class WorldOfGooEnv(gym.Env):
    """Entorno Gymnasium para el nivel 1 (Fase 1: geometrico, sin fisica)."""

    metadata = {"render_modes": []}

    def __init__(self, k=2, presupuesto=BOLAS_DISPONIBLES):
        super().__init__()
        self.k = k
        self.presupuesto = presupuesto

        # Observacion: posiciones (padded), mascara de nodos validos,
        # matriz de adyacencia y bolas restantes.
        self.observation_space = spaces.Dict({
            "posiciones":  spaces.Box(-1e4, 1e4, (MAX_BOLAS, 2), np.float32),
            "nodos":       spaces.MultiBinary(MAX_BOLAS),
            "adyacencia":  spaces.MultiBinary(MAX_BOLAS * MAX_BOLAS),
            "restantes":   spaces.Discrete(BOLAS_DISPONIBLES + 1),
        })
        # Espacio de acciones FIJO, con mascara para las validas.
        # Es la solucion estandar al espacio de acciones variable.
        self.action_space = spaces.Discrete(MAX_ACCIONES)

    # ---- ciclo estandar -------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        L = L_REPOSO
        self.pos = {0:(0.,0.), 1:(L,0.), 2:(L,L), 3:(0.,L)}
        self.G = nx.complete_graph(4)
        self.restantes = self.presupuesto
        self._acciones = self._generar()
        return self._obs(), {"acciones_validas": len(self._acciones)}

    def step(self, accion):
        if accion >= len(self._acciones):
            # accion invalida: no deberia ocurrir si se usa la mascara
            return self._obs(), -1.0, False, False, {"invalida": True}

        i, j, Li, Lj, lado, P = self._acciones[accion]
        n = max(self.G.nodes) + 1
        self.G.add_node(n); self.G.add_edge(n, i); self.G.add_edge(n, j)
        self.pos[n] = P
        self.restantes -= 1
        self._acciones = self._generar()

        alcanzo = self.altura() >= META_Y
        sin_acciones = len(self._acciones) == 0
        terminado = alcanzo or self.restantes == 0 or sin_acciones

        # recompensa: bolas recolectadas si llega, 0 si no
        recompensa = float(self.restantes) if alcanzo else 0.0

        return (self._obs(), recompensa, terminado, False,
                {"acciones_validas": len(self._acciones),
                 "altura": self.altura(), "alcanzo": alcanzo})

    # ---- utilidades -----------------------------------------------------
    def altura(self):
        return max(p[1] for p in self.pos.values())

    def mascara(self):
        """Vector booleano de acciones validas. Lo consume MaskablePPO."""
        m = np.zeros(MAX_ACCIONES, dtype=bool)
        m[:len(self._acciones)] = True
        return m

    def _obs(self):
        pos = np.zeros((MAX_BOLAS, 2), np.float32)
        nodos = np.zeros(MAX_BOLAS, np.int8)
        ady = np.zeros((MAX_BOLAS, MAX_BOLAS), np.int8)
        for n, p in self.pos.items():
            pos[n] = p; nodos[n] = 1
        for u, v in self.G.edges:
            ady[u, v] = ady[v, u] = 1
        return {"posiciones": pos, "nodos": nodos,
                "adyacencia": ady.flatten(), "restantes": self.restantes}

    def _generar(self):
        if self.restantes == 0:
            return []
        val, vistos = [], set()
        LS = largos(self.k)
        borde = self._envolvente()
        for (i, j) in itertools.combinations(self.G.nodes, 2):
            for Li, Lj, lado in itertools.product(LS, LS, (+1, -1)):
                P = interseccion(self.pos[i], self.pos[j], Li, Lj, lado)
                if P is None or P[1] < Y_SUELO + R_BOLA:
                    continue
                if any(dist(P, q) < 2*R_BOLA for q in self.pos.values()):
                    continue
                nuevas = [(P, self.pos[i]), (P, self.pos[j])]
                if any(se_cruzan(a, b, self.pos[u], self.pos[v])
                       for a, b in nuevas for u, v in self.G.edges):
                    continue
                if self._dentro(P, borde):
                    continue
                clave = (i, j, round(P[0]/2), round(P[1]/2))
                if clave in vistos:
                    continue
                vistos.add(clave)
                val.append((i, j, Li, Lj, lado, P))
        return val[:MAX_ACCIONES]

    def _envolvente(self):
        pts = sorted(set(self.pos.values()))
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

    @staticmethod
    def _dentro(P, poly):
        if len(poly) < 3:
            return False
        dentro = False
        for i in range(len(poly)):
            x1, y1 = poly[i]; x2, y2 = poly[(i+1) % len(poly)]
            if ((y1 > P[1]) != (y2 > P[1])) and \
               (P[0] < (x2-x1)*(P[1]-y1)/(y2-y1+1e-15) + x1):
                dentro = not dentro
        return dentro


# ==========================================================================
print("=" * 68)
print("  1. VERIFICACION DE LA API DE GYMNASIUM")
print("=" * 68)
env = WorldOfGooEnv(k=2)
print(f"  gymnasium        : {gym.__version__}")
print(f"  action_space     : {env.action_space}")
print(f"  observation_space:")
for clave, esp in env.observation_space.spaces.items():
    print(f"      {clave:<12}: {esp}")

obs, info = env.reset(seed=0)
print(f"\n  reset() devuelve observacion con claves: {sorted(obs.keys())}")
print(f"  observacion valida segun el espacio declarado: "
      f"{env.observation_space.contains(obs)}")

print("\n" + "=" * 68)
print("  2. ESTADO OBSERVABLE TRAS reset()")
print("=" * 68)
print(f"  bolas colocadas : {int(obs['nodos'].sum())}")
print(f"  uniones         : {int(obs['adyacencia'].sum() // 2)}")
print(f"  bolas restantes : {obs['restantes']}")
print(f"  altura maxima   : {env.altura():.1f} px  (meta {META_Y:.0f})")
print(f"  acciones validas: {info['acciones_validas']} "
      f"de {env.action_space.n} posibles")

print("\n" + "=" * 68)
print("  3. EL PROBLEMA DEL ESPACIO DE ACCIONES VARIABLE")
print("=" * 68)
print("  El espacio declarado es fijo, pero las acciones realmente validas")
print("  cambian en cada paso. La mascara resuelve esa diferencia.\n")

env.reset(seed=0)
print(f"  {'paso':<6}{'validas':<10}{'declaradas':<13}{'% util'}")
print("  " + "-" * 40)
for paso in range(6):
    m = env.mascara()
    n = int(m.sum())
    print(f"  {paso:<6}{n:<10}{env.action_space.n:<13}"
          f"{100*n/env.action_space.n:.1f}%")
    if n == 0:
        break
    env.step(int(np.flatnonzero(m)[0]))

print("\n  Sin mascara, un agente elegiria acciones invalidas la mayor parte")
print("  del tiempo. MaskablePPO (SB3-contrib) consume exactamente este")
print("  vector para restringir la distribucion de la politica.")

print("\n" + "=" * 68)
print("  4. EPISODIO COMPLETO CON POLITICA ALEATORIA ENMASCARADA")
print("=" * 68)
rng = np.random.default_rng(0)
obs, info = env.reset(seed=0)
print(f"  {'paso':<6}{'accion':<9}{'validas':<10}{'altura':<10}{'restantes':<11}{'r'}")
print("  " + "-" * 54)
paso, total = 0, 0.0
while True:
    m = env.mascara()
    if not m.any():
        break
    a = int(rng.choice(np.flatnonzero(m)))
    obs, r, term, trunc, info = env.step(a)
    total += r
    paso += 1
    print(f"  {paso:<6}{a:<9}{info['acciones_validas']:<10}"
          f"{info['altura']:<10.1f}{obs['restantes']:<11}{r:.0f}")
    if term or trunc:
        break

print(f"\n  episodio terminado en {paso} pasos")
print(f"  alcanzo la meta : {info.get('alcanzo', False)}")
print(f"  recompensa total: {total:.0f}")

print("\n" + "=" * 68)
print("  5. VARIAS CORRIDAS")
print("=" * 68)
alturas, largos_ep = [], []
for semilla in range(20):
    rng = np.random.default_rng(semilla)
    obs, info = env.reset(seed=semilla)
    n = 0
    while True:
        m = env.mascara()
        if not m.any():
            break
        obs, r, term, trunc, info = env.step(int(rng.choice(np.flatnonzero(m))))
        n += 1
        if term or trunc:
            break
    alturas.append(info["altura"]); largos_ep.append(n)

print(f"  episodios            : 20")
print(f"  pasos promedio       : {np.mean(largos_ep):.1f}")
print(f"  altura media final   : {np.mean(alturas):.1f} px")
print(f"  altura maxima lograda: {max(alturas):.1f} px  (meta {META_Y:.0f})")
print(f"  episodios exitosos   : {sum(a >= META_Y for a in alturas)}/20")
print("\n  Una politica aleatoria casi nunca llega. La recompensa es dispersa,")
print("  como se anticipo en el Taller 1, y por eso hace falta busqueda")
print("  informada primero y aprendizaje con shaping despues.")

print("\n" + "=" * 68)
print("  CONCLUSION")
print("=" * 68)
print("  El estado del ambiente se observa y las acciones se ejecutan")
print("  programaticamente mediante la API estandar de Gymnasium.")
print("  El espacio de acciones variable se resuelve con enmascaramiento,")
print("  compatible con MaskablePPO de SB3-contrib.")
