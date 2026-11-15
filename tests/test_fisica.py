"""Pruebas unitarias de wog.fisica, fijadas contra evidencia/pymunk.txt."""

import pytest

from wog.fisica import (
    C_DAMP, K_RESORTE, L_REPOSO, MAXFORCE, PESO_BOLA, R_BOLA,
    colocar_bola, crear_espacio, dinamicos, estabilizar, tension, unir,
)


def test_parametros_derivados():
    assert R_BOLA == pytest.approx(1.5)
    assert L_REPOSO == pytest.approx(15.0)
    assert PESO_BOLA == pytest.approx(3.0)
    assert K_RESORTE == pytest.approx(26.65, abs=0.005)
    assert C_DAMP == pytest.approx(5.09, abs=0.005)
    assert MAXFORCE == pytest.approx(36.0)


def k4_anclado():
    space = crear_espacio()
    L = L_REPOSO
    b = [colocar_bola(space, (0, R_BOLA), fija=True),
         colocar_bola(space, (L, R_BOLA), fija=True),
         colocar_bola(space, (L, R_BOLA + L)),
         colocar_bola(space, (0, R_BOLA + L))]
    uniones = []
    for i, j in [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)]:
        u = unir(space, b[i], b[j])
        if u:
            uniones.append(u)
    return space, uniones


def test_union_entre_fijas_se_omite():
    space, uniones = k4_anclado()
    assert len(dinamicos(space)) == 2
    assert len(uniones) == 5


def test_k4_estabiliza_como_en_evidencia():
    # evidencia/pymunk.txt, CASO 1
    space, uniones = k4_anclado()
    r = estabilizar(space, uniones, MAXFORCE, pasos=900, paso_min=60)
    assert r.estable and r.paso == 61
    assert r.rotas == []
    assert [round(tension(j), 2) for j in uniones] == \
        [2.14, 0.86, 2.14, 1.21, 1.21]


def test_ruptura_por_umbral():
    space, uniones = k4_anclado()
    r = estabilizar(space, uniones, umbral=0.5)
    assert len(r.rotas) > 0
    assert len(uniones) == 5 - len(r.rotas)
    assert all(f > 0.5 for _, f in r.rotas)


def test_sin_umbral_no_rompe():
    space, uniones = k4_anclado()
    r = estabilizar(space, uniones, umbral=None)
    assert r.rotas == [] and len(uniones) == 5
    assert r.pico > 0
