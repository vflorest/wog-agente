"""
Prueba minima de Pymunk - Taller 2, preguntas 3 y 4.

Verifica que se puede:
  (a) simular la estructura con fisica,
  (b) leer el estado programaticamente (posiciones, velocidades, tensiones),
  (c) modelar la RUPTURA de uniones, que es justamente lo que OpenGOO
      parsea pero no implementa.

Ejecutar:  python3 prueba_pymunk.py
"""

from wog.fisica import (
    C_DAMP, DT, FREQ_HZ, GRAVEDAD, K_ESCALA, K_RESORTE, L_REPOSO, MASA,
    MAXFORCE, PESO_BOLA, R_BOLA, UMBRAL_EN_PESOS, ZETA,
    colocar_bola, crear_espacio, dinamicos, estabilizar, tension, unir,
)

PASOS_MAX = 900


def simular(space, uniones, etiqueta, romper=True):
    """Estabiliza con el umbral de 12 pesos e informa el resultado."""
    r = estabilizar(space, uniones, MAXFORCE if romper else None,
                    pasos=PASOS_MAX, paso_min=60)
    if r.estable:
        print(f"  [{etiqueta}] estabilizo en el paso {r.paso} "
              f"({r.paso*DT:.2f} s simulados)")
    else:
        print(f"  [{etiqueta}] no estabilizo en {PASOS_MAX} pasos")
    if r.rotas:
        print(f"  [{etiqueta}] uniones ROTAS: {len(r.rotas)}, "
              f"la primera en el paso {r.rotas[0][0]} con F={r.rotas[0][1]:.1f}")
    else:
        print(f"  [{etiqueta}] ninguna union se rompio")
    return r.rotas


def voladizo(space, n_bolas, umbral=None):
    """Cadena horizontal anclada a la izquierda, construida INCREMENTALMENTE:
    se coloca una bola y se deja estabilizar antes de colocar la siguiente.

    Esto no es un detalle de implementacion. Construir toda la estructura de
    golpe produce un transitorio violento cuyo pico de tension no tiene
    sentido fisico y rompe uniones que una estructura asentada soportaria.
    El modelo por turnos del proyecto (una accion = colocar + estabilizar)
    es justamente lo que evita ese artefacto.
    """
    a0 = colocar_bola(space, (0, 30), fija=True)
    a1 = colocar_bola(space, (0, 30 + L_REPOSO), fija=True)
    cadena, uniones = [a0, a1], []
    pico, rotas = 0.0, 0
    for n in range(1, n_bolas + 1):
        nb = colocar_bola(space, (n * L_REPOSO * 0.87, 30 + (n % 2) * L_REPOSO))
        for anc in (cadena[-1], cadena[-2]):
            j = unir(space, anc, nb)
            if j: uniones.append(j)
        cadena.append(nb)
        r = estabilizar(space, uniones, umbral)
        pico, rotas = max(pico, r.pico), rotas + len(r.rotas)
    return uniones, pico, rotas


# ==========================================================================
def main():
    print("=" * 68)
    print("  PARAMETROS DERIVADOS DE balls.xml (escala Box2D, 1 u = 10 px)")
    print("=" * 68)
    print(f"  radio de bola      : {R_BOLA:.2f} u   ({R_BOLA/K_ESCALA:.0f} px)")
    print(f"  largo de reposo    : {L_REPOSO:.2f} u  ({L_REPOSO/K_ESCALA:.0f} px)")
    print(f"  masa (towermass)   : {MASA:.2f} u")
    print(f"  gravedad           : {GRAVEDAD:.1f}")
    print(f"  peso de una bola   : {PESO_BOLA:.2f}")
    print(f"  frecuencia / zeta  : {FREQ_HZ} Hz / {ZETA}   (hardcodeados en OpenGOO)")
    print(f"  -> rigidez k       : {K_RESORTE:.2f}")
    print(f"  -> damping c       : {C_DAMP:.2f}")
    print(f"  umbral de ruptura  : {MAXFORCE:.2f}  ({UMBRAL_EN_PESOS:.0f} pesos de bola)")
    print()
    print("  Nota: balls.xml declara maxforce=1000, pero en unidades internas")
    print("  del juego que se desconocen, porque OpenGOO nunca lee ese valor.")
    print("  Por eso el umbral se expresa como multiplo del peso de una bola.")

    # --------------------------------------------------------------------------
    print("\n" + "=" * 68)
    print("  CASO 1 - Estructura K4 anclada al suelo")
    print("=" * 68)

    space = crear_espacio()
    L = L_REPOSO
    b0 = colocar_bola(space, (0, R_BOLA), fija=True)
    b1 = colocar_bola(space, (L, R_BOLA), fija=True)
    b2 = colocar_bola(space, (L, R_BOLA + L))
    b3 = colocar_bola(space, (0, R_BOLA + L))

    uniones = []
    for a, b in [(b0,b1), (b1,b2), (b2,b3), (b3,b0), (b0,b2), (b1,b3)]:
        j = unir(space, a, b)
        if j: uniones.append(j)

    print(f"  bolas dinamicas: {len(dinamicos(space))}   uniones: {len(uniones)}")
    simular(space, uniones, "K4")

    print("\n  ESTADO OBSERVABLE (lo que leeria el agente):")
    for n, b in enumerate(dinamicos(space)):
        print(f"    bola {n}: pos=({b.position.x:6.2f}, {b.position.y:6.2f})   "
              f"vel={b.velocity.length:.3f}")
    print("  TENSION POR UNION:")
    for n, j in enumerate(uniones):
        f = tension(j)
        print(f"    union {n}: F={f:6.2f}  ({100*f/MAXFORCE:5.1f}% del umbral)")

    # --------------------------------------------------------------------------
    print("\n" + "=" * 68)
    print("  CASO 2 - Tension segun el largo del voladizo (construccion incremental)")
    print("=" * 68)
    print("  Cada bola se coloca y se deja estabilizar antes de la siguiente,")
    print("  igual que en el modelo por turnos del proyecto.\n")

    print(f"  {'bolas':<8}{'uniones':<10}{'F pico':<11}{'en pesos de bola'}")
    print("  " + "-" * 48)
    for n in range(1, 9):
        sp = crear_espacio()
        u, pico, _ = voladizo(sp, n)
        print(f"  {n:<8}{len(u):<10}{pico:<11.2f}{pico/PESO_BOLA:.1f}")

    print("\n  La tension crece de forma monotona con el largo del voladizo.")
    print("  Construyendo la misma estructura de golpe el pico NO es monotono:")
    print("  domina el transitorio del impacto y rompe uniones que una")
    print("  estructura asentada soportaria. De ahi que la construccion deba")
    print("  ser incremental, que es exactamente el modelo por turnos elegido.")

    # --------------------------------------------------------------------------
    print("\n" + "=" * 68)
    print("  CASO 3 - Efecto del umbral de ruptura")
    print("=" * 68)
    print("  El umbral es el parametro NO OBSERVABLE del que depende el optimo.\n")

    print(f"  {'umbral':<20}{'max bolas antes de colapsar'}")
    print("  " + "-" * 50)
    for pesos in [4, 8, 12, 16, 24, 32]:
        umbral = pesos * PESO_BOLA
        limite = 0
        for n in range(1, 9):
            sp = crear_espacio()
            _, _, rotas = voladizo(sp, n, umbral=umbral)
            if rotas:
                break
            limite = n
        print(f"  {pesos:>2} pesos ({umbral:5.1f})   {limite}" +
              ("+" if limite == 8 else ""))

    print("\n  Un mismo problema geometrico admite estructuras distintas segun")
    print("  este parametro. Es lo planteado en el Taller 1: el optimo no es")
    print("  un dato del problema, hay que estimarlo.")

    # --------------------------------------------------------------------------
    print("\n" + "=" * 68)
    print("  CASO 4 - Sin umbral de ruptura: el comportamiento de OpenGOO")
    print("=" * 68)
    sp = crear_espacio()
    u, pico, rotas = voladizo(sp, 8, umbral=None)
    print(f"  voladizo de 8 bolas, sin aplicar maxforce")
    print(f"  tension pico     : {pico:.2f}  ({pico/PESO_BOLA:.1f} pesos de bola)")
    print(f"  uniones rotas    : {rotas}")
    print(f"  con umbral de 12 pesos habria roto: "
          f"{'SI' if pico > 12*PESO_BOLA else 'no'}")
    print("\n  Misma estructura y misma fisica, pero sin leer maxforce la union")
    print("  se estira sin limite y nunca se rompe. Ese es el comportamiento")
    print("  de OpenGOO, y por eso no puede representar el modo de falla.")

    print("\n" + "=" * 68)
    print("  CONCLUSION")
    print("=" * 68)
    print("  Pymunk permite simular la estructura, leer posiciones, velocidades")
    print("  y tensiones, y modelar la ruptura por umbral de fuerza.")
    print("  Los tres son requisitos del proyecto que OpenGOO no cubre.")


if __name__ == "__main__":
    main()
