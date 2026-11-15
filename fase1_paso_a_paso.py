"""
World of Goo - Nivel 1
Modelo combinatorio-geometrico (Fase 1: sin fisica)

Version narrada paso a paso.
Ejecutar con:  python3 fase1_paso_a_paso.py
Para ir mas rapido:  python3 fase1_paso_a_paso.py --rapido
"""

import sys
import time

from wog.geometria import (
    BOLAS_DISPONIBLES, L_REPOSO, MAXLEN, META_Y, MINLEN, R_BOLA,
    REGLAS_FASE1, Estado, generar_acciones,
)

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
# Acciones y descripcion del estado (la logica vive en wog.geometria)
# --------------------------------------------------------------------------
def acciones(e, explicar=False):
    """
    Toda accion es: anclar una bola nueva a un par (i, j) existente.
    Se aplican tres filtros de validez, en orden de costo creciente.
    Reglas de esta demo: REGLAS_FASE1 (isosceles, sin filtro de suelo).
    """
    validas, descartes = generar_acciones(e, k=1, reglas=REGLAS_FASE1)

    if explicar:
        total = len(validas) + sum(descartes.values())
        narrar(f"    candidatos brutos (pares x 2 lados) : {total}")
        narrar(f"    descartados por alcance             : {descartes['alcance']}")
        narrar(f"    descartados por solapamiento        : {descartes['solapa']}")
        narrar(f"    descartados por cruce de uniones    : {descartes['cruce']}")
        narrar(f"    ACCIONES VALIDAS                    : {len(validas)}")

    return validas


def describir(e):
    V, E, laman, exc = e.laman()
    narrar(f"    bolas colocadas : {V}      uniones: {E}", 0)
    narrar(f"    bolas restantes : {e.restantes}", 0)
    narrar(f"    altura maxima   : {e.altura_max():.1f} px "
           f"(meta {META_Y:.0f} px)", 0)
    narrar(f"    conteo de Laman : E={E}, 2V-3={laman}, exceso={exc} "
           f"-> {'rigida' if exc >= 0 else 'MECANISMO'}", 0)
    narrar(f"    grafo planar    : {e.es_planar()}", 0)
    narrar(f"    h(n)            : {e.heuristica():.2f} bolas")


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
    describir(e)
    narrar("\n  El exceso de Laman vale 1: el cuadrado sin diagonales", 0)
    narrar("  seria un mecanismo (se deforma en rombo). Con las dos", 0)
    narrar("  diagonales queda rigido y con una union de sobra.")

    titulo("PASO 2 - GENERACION DE ACCIONES")
    narrar("Una accion es: elegir un par de bolas ya colocadas y anclar", 0)
    narrar("una bola nueva formando triangulo. El 'lado' (+1/-1) elige a", 0)
    narrar("que lado de ese par queda la bola nueva.\n", 0)
    narrar("Se filtra en tres etapas, de mas barata a mas cara:\n", 0)
    acc = acciones(e, explicar=True)

    titulo("PASO 3 - INSPECCION DE LAS ACCIONES VALIDAS")
    for k, (i, j, _, _, lado, P) in enumerate(acc):
        narrar(f"    accion {k}: anclar en ({i},{j}) lado {lado:+d} "
               f"-> posicion ({P[0]:7.1f}, {P[1]:7.1f})", PAUSA / 2)

    titulo("PASO 4 - APLICAR UNA ACCION")
    elegida = max(acc, key=lambda a: a[5][1])     # la que mas altura gana
    i, j, _, _, lado, P = elegida
    narrar(f"Elegimos la que mas altura gana: ({i},{j}) lado {lado:+d}\n", 0)
    e = e.aplicar(elegida)
    describir(e)
    narrar("\n  Se agrego 1 nodo y 2 aristas: extension de Henneberg tipo I.", 0)
    narrar("  El exceso de Laman NO cambio. La rigidez se preserva.")

    titulo("PASO 5 - CONSTRUCCION GREEDY (heuristica: ganar altura)")
    narrar("Repetimos eligiendo siempre la accion que sube mas.\n", 0)
    narrar("paso  accion         altura  ramif.  V   E   2V-3 exc  h(n)", 0)
    narrar("-" * 62, 0)
    for paso in range(2, 9):
        accs = acciones(e)
        if not accs:
            narrar("  sin acciones validas")
            break
        ramif = len(accs)
        mejor = max(accs, key=lambda a: a[5][1])
        e = e.aplicar(mejor)
        i, j, _, _, lado, _ = mejor
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
    describir(t)
    narrar("\n  Exceso 0: el triangulo es MINIMAMENTE rigido (sin redundancia).", 0)
    narrar("  El K4 tiene exceso 1: tolera perder una union.", 0)
    narrar("  En ambos casos Henneberg preserva la rigidez.")

    narrar("\n" + "=" * 68, 0)
    narrar("  FIN", 0)
    narrar("=" * 68, 0)


if __name__ == "__main__":
    main()
