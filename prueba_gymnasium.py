"""
Prueba minima de Gymnasium - Taller 2, pregunta 4.

Verifica que se puede observar el estado del ambiente y ejecutar acciones
programaticamente, envolviendo el modelo de Fase 1 en la API estandar.

Muestra ademas el problema del espacio de acciones variable y su solucion
por enmascaramiento, que fue la objecion planteada por el profesor.

Ejecutar:  python3 prueba_gymnasium.py
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from wog.geometria import (
    BOLAS_DISPONIBLES, META_Y, REGLAS_GYM_T2, Estado, generar_acciones,
)

# --------------------------------------------------------------------------
MAX_BOLAS = 4 + BOLAS_DISPONIBLES          # cota superior de nodos
MAX_ACCIONES = REGLAS_GYM_T2.max_acciones   # cota superior del espacio de acciones (256)


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
        e = Estado.inicial()
        self.pos, self.G = e.pos, e.G
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
        e = Estado(self.G, self.pos, self.restantes)
        return generar_acciones(e, self.k, REGLAS_GYM_T2)[0]


# ==========================================================================
def main():
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


if __name__ == "__main__":
    main()
