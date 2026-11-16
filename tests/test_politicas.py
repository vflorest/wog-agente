"""Pruebas de wog.politicas (politica determinista del Taller 4 y aleatoria)."""

import numpy as np
import pytest

from wog.env import GooPhysicsEnv
from wog.geometria import Estado
from wog.politicas import REGLAS, PoliticaAleatoria, PoliticaT4

# K4 ideal: altura 150, ancho D1 150 -> e(s) = 1. Con una bola nueva P,
# e(s, a) = max(150, P_y) / 150 salvo que P quede apoyada en el suelo.
K4 = Estado.inicial().pos


def decidir(pred, e_max, n=14, meta=600.0):
    return PoliticaT4(e_max, meta=meta).decidir(K4, n, pred)


def test_llegar_con_esbeltez_aceptable():
    pred = {10: (0.0, 600.0), 11: (0.0, 650.0), 12: (0.0, 300.0)}
    assert decidir(pred, e_max=4.0) == (10, "llegar")      # e = 4 <= 4


def test_llegar_por_pocas_bolas():
    pred = {10: (0.0, 650.0), 11: (0.0, 600.0), 12: (0.0, 300.0)}
    # todas las de L exceden e_max = 2, pero n = 5 < 6
    assert decidir(pred, e_max=2.0, n=5) == (11, "llegar")


def test_asegurar():
    pred = {10: (0.0, 650.0), 11: (0.0, 200.0), 12: (0.0, 160.0)}
    assert decidir(pred, e_max=2.0, n=6) == (12, "asegurar")


def test_avanzar_mayor_dh():
    pred = {10: (0.0, 200.0), 11: (0.0, 290.0), 12: (0.0, 400.0)}
    assert decidir(pred, e_max=2.0) == (11, "avanzar")      # 400 excede


def test_ensanchar():
    pred = {10: (0.0, 400.0), 11: (0.0, 350.0), 12: (0.0, 500.0)}
    assert decidir(pred, e_max=2.0) == (11, "ensanchar")


def test_ensanchar_prefiere_bola_apoyada():
    # P apoyada en el suelo amplia la base: e = 150 / 300 = 0.5
    pred = {10: (-150.0, 15.5), 11: (300.0, 100.0)}
    assert decidir(pred, e_max=0.4) == (10, "ensanchar")


class EFija(PoliticaT4):
    """e y dh fijados a mano: (a, P, e, dh) por accion."""

    def __init__(self, tabla, e_max):
        super().__init__(e_max)
        self.tabla = tabla

    def evaluar(self, pos, predicciones):
        return sorted(self.tabla)


def test_empates_mayor_dh_y_luego_menor_indice():
    lejos, meta = (0.0, 100.0), (0.0, 600.0)
    # ensanchar: misma e minima, gana mayor dh; con mismo dh, menor indice
    t = [(9, lejos, 2.0, 0.0), (7, lejos, 2.0, 30.0), (3, lejos, 2.5, 90.0)]
    assert EFija(t, 1.0).decidir(K4, 14, {}) == (7, "ensanchar")
    t = [(9, lejos, 2.0, 30.0), (7, lejos, 2.0, 30.0)]
    assert EFija(t, 1.0).decidir(K4, 14, {}) == (7, "ensanchar")
    # llegar: misma e, gana mayor dh
    t = [(1, meta, 3.0, 10.0), (2, meta, 3.0, 50.0)]
    assert EFija(t, 4.0).decidir(K4, 14, {}) == (2, "llegar")
    # avanzar: mismo dh, menor indice
    t = [(8, lejos, 1.0, 40.0), (4, lejos, 1.5, 40.0)]
    assert EFija(t, 2.0).decidir(K4, 14, {}) == (4, "avanzar")


def test_sin_acciones_lanza_error():
    with pytest.raises(ValueError):
        decidir({}, e_max=1.0)


def test_episodio_real_determinista_y_valido():
    def correr():
        env = GooPhysicsEnv()
        env.reset(seed=0)
        pi, out = PoliticaT4(3.0), []
        while True:
            a, regla = pi(env)
            assert env.action_masks()[a] and regla in REGLAS
            _, _, term, _, info = env.step(a)
            out.append((a, regla, info["causa"]))
            if term:
                return out
    assert correr() == correr()


def test_aleatoria_solo_validas_y_reproducible():
    env = GooPhysicsEnv(p_carga=0)
    env.reset(seed=0)
    pi = PoliticaAleatoria(3)
    assert pi(env)[0] == PoliticaAleatoria(3)(env)[0]
    for _ in range(50):
        a, regla = pi(env)
        assert env.action_masks()[a] and regla == "aleatoria"
