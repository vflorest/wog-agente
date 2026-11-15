"""
Fisica del nivel 1 con Pymunk: parametros, espacio, colocacion de bolas,
uniones, tensiones, estabilizacion y ruptura por umbral.

La construccion SIEMPRE es incremental: se coloca una bola y se estabiliza
antes de la siguiente. Construir todo de una vez produce un transitorio
violento cuyo pico de tension no tiene sentido fisico (ver CLAUDE.md).
"""

import math
from dataclasses import dataclass, field

import pymunk

# --------------------------------------------------------------------------
# ESCALA
# --------------------------------------------------------------------------
# OpenGOO usa physics::K = 0.1 -> 1 unidad de fisica = 10 px de nivel.
# Se trabaja en esas mismas unidades para que los valores de balls.xml
# sean directamente comparables.
K_ESCALA = 0.1

R_BOLA   = 15.0 * K_ESCALA      # 1.5   (radio 15 px)
L_REPOSO = 150.0 * K_ESCALA     # 15.0  (largo de reposo 150 px)
GRAVEDAD = -10.0                # og_world.cpp: b2Vec2(0, -10)
MASA     = 3.0 * K_ESCALA       # 0.3   (towermass = 3.0 en balls.xml)

# OpenGOO crea la union como b2DistanceJoint con valores hardcodeados
# (los springconst/dampfac de balls.xml NUNCA se leen):
FREQ_HZ = 1.5
ZETA    = 0.9

OMEGA     = 2 * math.pi * FREQ_HZ
K_RESORTE = MASA * OMEGA ** 2
C_DAMP    = 2 * MASA * ZETA * OMEGA

PESO_BOLA = MASA * abs(GRAVEDAD)          # 3.0

# balls.xml declara maxforce="1000", pero ese valor esta en unidades internas
# de World of Goo, que se desconocen: OpenGOO nunca lo usa, asi que no hay
# forma de verificar su escala. Se expresa el umbral como multiplo del peso
# de una bola, que si es una cantidad conocida del modelo. Calibrar contra
# el juego real es tarea pendiente.
UMBRAL_EN_PESOS = 12.0
MAXFORCE = UMBRAL_EN_PESOS * PESO_BOLA

DT = 1 / 60
PASOS_ESTABILIZAR = 600     # tope de pasos por estabilizacion
PASO_MIN = 30               # no se declara reposo antes de este paso
V_REPOSO = 0.05             # velocidad maxima para considerar reposo


# --------------------------------------------------------------------------
# Construccion
# --------------------------------------------------------------------------
def crear_espacio():
    space = pymunk.Space()
    space.gravity = (0, GRAVEDAD)
    space.iterations = 6                    # og_world.cpp: SetSimulation(6, 2, 60)
    suelo = pymunk.Segment(space.static_body, (-500, 0), (500, 0), 0.5)
    suelo.friction = 0.04                   # material rock: friction 4 * 0.01
    space.add(suelo)
    return space


def colocar_bola(space, pos, fija=False):
    if fija:
        cuerpo = pymunk.Body(body_type=pymunk.Body.STATIC)
    else:
        cuerpo = pymunk.Body(MASA, pymunk.moment_for_circle(MASA, 0, R_BOLA))
    cuerpo.position = pos
    forma = pymunk.Circle(cuerpo, R_BOLA)
    forma.friction, forma.elasticity = 0.15, 0.1     # OGBall, og_ball.cpp
    forma.filter = pymunk.ShapeFilter(group=1)       # las bolas no colisionan
    space.add(cuerpo, forma)
    return cuerpo


def unir(space, a, b):
    """Resorte amortiguado. Pymunk exige un cuerpo dinamico; una union
    entre dos fijos no tiene efecto fisico y se omite."""
    if a.body_type != pymunk.Body.DYNAMIC and b.body_type != pymunk.Body.DYNAMIC:
        return None
    j = pymunk.DampedSpring(a, b, (0, 0), (0, 0),
                            a.position.get_distance(b.position),
                            K_RESORTE, C_DAMP)
    space.add(j)
    return j


def tension(j):
    """F = k * |largo_actual - largo_reposo|"""
    return abs(K_RESORTE * (j.a.position.get_distance(j.b.position) - j.rest_length))


def dinamicos(space):
    return [b for b in space.bodies if b.body_type == pymunk.Body.DYNAMIC]


# --------------------------------------------------------------------------
# Estabilizacion y ruptura
# --------------------------------------------------------------------------
@dataclass
class Estabilizacion:
    paso: int            # ultimo paso simulado (indice desde 0)
    estable: bool        # True si se alcanzo el reposo antes del tope
    pico: float = 0.0    # tension maxima observada en cualquier union
    rotas: list = field(default_factory=list)   # [(paso, F), ...]


def estabilizar(space, uniones, umbral=None, pasos=PASOS_ESTABILIZAR,
                paso_min=PASO_MIN):
    """
    Corre la fisica hasta el reposo o hasta 'pasos'. Si 'umbral' no es None,
    toda union con tension > umbral se rompe (se quita del espacio y de la
    lista 'uniones', que se modifica en el lugar).
    """
    pico, rotas = 0.0, []
    for p in range(pasos):
        space.step(DT)
        for j in list(uniones):
            f = tension(j)
            pico = max(pico, f)
            if umbral is not None and f > umbral:
                space.remove(j)
                uniones.remove(j)
                rotas.append((p, f))
        v = max((b.velocity.length for b in dinamicos(space)), default=0)
        if p > paso_min and v < V_REPOSO:
            return Estabilizacion(p, True, pico, rotas)
    return Estabilizacion(pasos - 1, False, pico, rotas)
