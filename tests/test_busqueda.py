"""Pruebas de wog.busqueda (A* geometrico sin fisica)."""

import networkx as nx

from wog.busqueda import a_estrella, clave
from wog.geometria import REGLAS_T4, Estado, generar_acciones


def test_meta_baja_se_alcanza_con_una_bola():
    r = a_estrella(meta=300.0)
    assert r.minimo == 1 and len(r.acciones) == 1
    assert r.alturas[-1] >= 300.0


def test_cierre_no_cambia_el_minimo():
    con, sin = a_estrella(meta=450.0), a_estrella(meta=450.0, cerrar=False)
    assert con.minimo == sin.minimo == 3
    assert con.expandidos <= sin.expandidos


def test_secuencia_reproduce_la_altura():
    r = a_estrella(meta=450.0)
    e = Estado.inicial()
    for acc in r.acciones:
        assert acc in generar_acciones(e, 2, REGLAS_T4)[0]
        e = e.aplicar(acc)
    assert e.altura_max() == r.alturas[-1] >= 450.0


def test_clave_ignora_etiquetas_pero_no_aristas():
    e = Estado.inicial()
    # mismas posiciones y aristas con etiquetas permutadas -> misma clave
    perm = {0: 2, 1: 3, 2: 0, 3: 1}
    G = nx.relabel_nodes(e.G, perm)
    pos = {perm[n]: p for n, p in e.pos.items()}
    assert clave(Estado(G, pos, 14)) == clave(e)
    # mismas posiciones, una arista menos -> clave distinta
    G2 = e.G.copy()
    G2.remove_edge(0, 2)
    assert clave(Estado(G2, e.pos, 14)) != clave(e)
