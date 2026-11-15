"""Pruebas unitarias de wog.geometria, fijadas contra la evidencia de Taller 2."""

from dataclasses import replace

import networkx as nx
import pytest

from wog.geometria import (
    L_REPOSO, REGLAS_FASE1, REGLAS_GYM_T2, REGLAS_NOTEBOOK, REGLAS_T4,
    Estado, apice, dist_segmento, envolvente, generar_acciones,
    holgura_suficiente, interseccion, largos, se_cruzan,
)


# --------------------------------------------------------------------------
# Primitivas
# --------------------------------------------------------------------------
def test_largos():
    assert largos(1) == [150.0]
    assert largos(3) == [100.0, 150.0, 200.0]
    assert largos(5) == [100.0, 125.0, 150.0, 175.0, 200.0]


def test_apice_e_interseccion_no_son_iguales_bit_a_bit():
    # Mismo punto matematico, distinto redondeo (Bugs conocidos, CLAUDE.md).
    A, B = (0.0, 0.0), (150.0, 150.0)
    p = apice(A, B, 150.0, +1)
    q = interseccion(A, B, 150.0, 150.0, +1)
    assert p == (0.0, 150.0)
    assert q != p
    assert q == pytest.approx(p, abs=1e-12)


def test_envolvente_de_k4():
    e = Estado.inicial()
    assert envolvente(e.pos.values()) == [(0.0, 0.0), (150.0, 0.0),
                                          (150.0, 150.0), (0.0, 150.0)]


# --------------------------------------------------------------------------
# se_cruzan: las dos versiones, sin unificar. Estos tests DOCUMENTAN el
# comportamiento actual; no lo validan como correcto.
# --------------------------------------------------------------------------
CRUCES = [
    # (descripcion, segmentos, permisivo, estricto)
    ("X propio",
     ((0, 0), (2, 2), (0, 2), (2, 0)), True, True),
    ("disjuntos no paralelos",
     ((0, 0), (1, 0), (5, 5), (6, 7)), False, False),
    ("extremo compartido",
     ((0, 0), (1, 0), (1, 0), (1, 1)), False, False),
    ("T: extremo de uno en el interior del otro",
     ((0, 0), (2, 0), (1, 0), (1, 1)), True, False),
    # colineales SUPERPUESTOS: ninguno de los dos modos los detecta
    ("colineales superpuestos parcialmente",
     ((0, 0), (2, 0), (1, 0), (3, 0)), False, False),
    ("colineal contenido en el otro",
     ((0, 0), (3, 0), (1, 0), (2, 0)), False, False),
    ("colineales identicos, sentido invertido",
     ((0, 0), (2, 0), (2, 0), (0, 0)), False, False),
    ("colineales superpuestos en diagonal",
     ((0, 0), (2, 2), (1, 1), (3, 3)), False, False),
    ("colineales superpuestos, escala de pixeles",
     ((0.0, 0.0), (150.0, 0.0), (75.0, 0.0), (300.0, 0.0)), False, False),
    ("colineales disjuntos",
     ((0, 0), (1, 0), (2, 0), (3, 0)), False, False),
    ("colineales con extremo compartido",
     ((0, 0), (1, 0), (1, 0), (2, 0)), False, False),
]


@pytest.mark.parametrize("desc, segs, permisivo, estricto", CRUCES,
                         ids=[c[0] for c in CRUCES])
def test_se_cruzan_modos(desc, segs, permisivo, estricto):
    assert se_cruzan(*segs) is permisivo
    assert se_cruzan(*segs, estricto=True) is estricto


# --------------------------------------------------------------------------
# Estado
# --------------------------------------------------------------------------
def test_estado_inicial_k4():
    e = Estado.inicial()
    assert e.laman() == (4, 6, 5, 1)
    assert e.es_planar()
    assert e.altura_max() == L_REPOSO
    assert e.heuristica() == pytest.approx(2.25)


def test_estado_inicial_triangulo():
    assert Estado.inicial(base="triangulo").laman() == (3, 3, 3, 0)


def test_henneberg_preserva_exceso():
    e = Estado.inicial()
    for _ in range(6):
        acc, _ = generar_acciones(e, 1, REGLAS_FASE1)
        e = e.aplicar(max(acc, key=lambda a: a[5][1]))
        assert e.laman()[3] == 1


def test_aplicar_acepta_tupla_de_4_y_de_6():
    e = Estado.inicial()
    P = (75.0, 279.9)
    a = e.aplicar((2, 3, -1, P))
    b = e.aplicar((2, 3, 150.0, 150.0, -1, P))
    assert a.pos == b.pos and set(a.G.edges) == set(b.G.edges)


# --------------------------------------------------------------------------
# Presets contra la evidencia
# --------------------------------------------------------------------------
def test_fase1_k4():
    # evidencia/fase1.txt, PASO 2 y PASO 3
    acc, d = generar_acciones(Estado.inicial(), 1, REGLAS_FASE1)
    assert (d["alcance"], d["solapa"], d["cruce"]) == (0, 4, 4)
    assert [(i, j, lado) for i, j, _, _, lado, _ in acc] == \
        [(0, 1, -1), (0, 3, +1), (1, 2, -1), (2, 3, -1)]
    assert acc[0][5][1] == pytest.approx(-129.9, abs=0.05)   # bajo el suelo


def test_gym_k4():
    # evidencia/gymnasium.txt: "acciones validas: 12 de 256 posibles"
    acc, _ = generar_acciones(Estado.inicial(), 2, REGLAS_GYM_T2)
    assert len(acc) == 12


def test_notebook_k1_y_k3():
    # salidas guardadas en Fase1_World_of_Goo.ipynb (celdas 14 y 16)
    e = Estado.inicial()
    acc, d = generar_acciones(e, 1, REGLAS_NOTEBOOK)
    assert len(acc) == 3
    assert (d["suelo"], d["alcance"], d["solapa"], d["cruce"]) == (3, 0, 2, 4)
    acc, d = generar_acciones(e, 3, REGLAS_NOTEBOOK)
    assert len(acc) == 31
    assert (d["suelo"], d["alcance"], d["solapa"], d["cruce"],
            d["duplicado"]) == (28, 10, 8, 31, 0)


@pytest.mark.parametrize("k, n0, n1", [(1, 3, 9), (2, 16, 22), (3, 31, 66),
                                       (4, 54, 133), (6, 116, 294),
                                       (8, 179, 505)])
def test_notebook_tabla_k(k, n0, n1):
    # Fase1_World_of_Goo.ipynb, celda 20 (greedy de altura, 4 pasos)
    e = Estado.inicial()
    assert len(generar_acciones(e, k, REGLAS_NOTEBOOK)[0]) == n0
    for _ in range(4):
        acc, _ = generar_acciones(e, k, REGLAS_NOTEBOOK)
        e = e.aplicar(max(acc, key=lambda x: x[5][1]))
    assert len(generar_acciones(e, k, REGLAS_NOTEBOOK)[0]) == n1


def test_reglas_t4():
    assert REGLAS_T4.suelo and REGLAS_T4.alcance and REGLAS_T4.envolvente
    assert REGLAS_T4.cruce_estricto is False
    assert REGLAS_T4.holgura
    assert REGLAS_T4.tol_duplicado is None
    assert REGLAS_T4.max_acciones is None
    assert not REGLAS_T4.usar_apice


def test_presets_taller2_sin_holgura():
    assert not (REGLAS_FASE1.holgura or REGLAS_NOTEBOOK.holgura
                or REGLAS_GYM_T2.holgura)


@pytest.mark.parametrize("k, n", [(1, 3), (2, 12), (3, 27)])
def test_t4_k4(k, n):
    acc, _ = generar_acciones(Estado.inicial(), k, REGLAS_T4)
    assert len(acc) == n


# --------------------------------------------------------------------------
# Holgura: bola nueva P anclada en A=(0,0), B=(150,0) con largos 150, lado +1
# -> P = (75, 129.9). Cada escenario agrega una union que pasa por P o
# muy cerca, en un caso que se_cruzan permisivo no detecta o detecta solo
# en parte.
# --------------------------------------------------------------------------
A, B = (0.0, 0.0), (150.0, 0.0)
P = interseccion(A, B, 150.0, 150.0, +1)
UX, UY = P[0] / 150.0, P[1] / 150.0            # direccion A -> P


def escenario(extra_pos, extra_aristas):
    pos = {0: A, 1: B, **extra_pos}
    G = nx.Graph()
    G.add_nodes_from(pos)
    G.add_edges_from([(0, 1), *extra_aristas])
    return Estado(G, pos, 5)


ESCENARIOS = {
    # union 2-3 horizontal que pasa exactamente por P: contacto en T
    "T": escenario({2: (P[0] - 100, P[1]), 3: (P[0] + 100, P[1])}, [(2, 3)]),
    # union 0-2 sobre la recta A->P: la union nueva P-A queda superpuesta
    "colineal superpuesto": escenario({2: (2 * P[0], 2 * P[1])}, [(0, 2)]),
    # igual, pero la union 0-2 desviada para pasar a 0.5 px de P
    "casi colineal 0.5 px": escenario({2: (2 * P[0] - UY, 2 * P[1] + UX)},
                                      [(0, 2)]),
}


def tiene_accion_objetivo(acc):
    return any(a[:2] == (0, 1) and a[4] == +1 for a in acc)


@pytest.mark.parametrize("nombre", ESCENARIOS)
def test_holgura_predicado(nombre):
    e = ESCENARIOS[nombre]
    assert not holgura_suficiente(P, 0, 1, e.pos, e.G.edges)


@pytest.mark.parametrize("nombre", ESCENARIOS)
def test_holgura_descarta_en_generar_acciones(nombre):
    # Se aislan los otros filtros que podrian atrapar el caso (cruce
    # estricto, sin envolvente) para que el descarte sea por holgura.
    e = ESCENARIOS[nombre]
    aislada = replace(REGLAS_T4, cruce_estricto=True, envolvente=False)
    sin, _ = generar_acciones(e, 1, replace(aislada, holgura=False))
    con, d = generar_acciones(e, 1, aislada)
    assert tiene_accion_objetivo(sin)
    assert not tiene_accion_objetivo(con)
    assert d["holgura"] >= 1


@pytest.mark.parametrize("nombre", ESCENARIOS)
def test_t4_descarta_escenarios(nombre):
    acc, _ = generar_acciones(ESCENARIOS[nombre], 1, REGLAS_T4)
    assert not tiene_accion_objetivo(acc)


def test_holgura_con_espacio_suficiente():
    # union 2-3 horizontal a 20 px por encima de P: holgura 20 >= 15
    e = escenario({2: (P[0] - 100, P[1] + 20), 3: (P[0] + 100, P[1] + 20)},
                  [(2, 3)])
    assert holgura_suficiente(P, 0, 1, e.pos, e.G.edges)


def test_dist_segmento():
    assert dist_segmento((1, 1), (0, 0), (2, 0)) == pytest.approx(1)
    assert dist_segmento((3, 0), (0, 0), (2, 0)) == pytest.approx(1)
    assert dist_segmento((1, 0), (0, 0), (0, 0)) == pytest.approx(1)


def test_sin_bolas_no_hay_acciones():
    acc, _ = generar_acciones(Estado.inicial(restantes=0), 1, REGLAS_FASE1)
    assert acc == []
