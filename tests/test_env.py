"""Pruebas de wog.env.GooPhysicsEnv (MDP del Taller 4)."""

import math

import numpy as np
import pytest

from wog.env import GooPhysicsEnv, MAX_NODOS, Theta
from wog.fisica import MAXFORCE
from wog.geometria import META_Y, REGLAS_T4, Estado, generar_acciones


def iguales(o1, o2):
    return o1.keys() == o2.keys() and all(
        np.array_equal(o1[c], o2[c]) for c in o1)


def jugar(env, semilla_env, semilla_politica, pasos=None):
    """Episodio con politica aleatoria enmascarada; devuelve la trayectoria."""
    rng = np.random.default_rng(semilla_politica)
    obs, info = env.reset(seed=semilla_env)
    tray = [(None, obs, 0.0, info)]
    while pasos is None or len(tray) <= pasos:
        a = int(rng.choice(np.flatnonzero(env.action_masks())))
        obs, r, term, trunc, info = env.step(a)
        tray.append((a, obs, r, info))
        if term or trunc:
            break
    return tray


def mismas(t1, t2):
    assert len(t1) == len(t2)
    for (a1, o1, r1, i1), (a2, o2, r2, i2) in zip(t1, t2):
        assert a1 == a2 and r1 == r2 and iguales(o1, o2)
        assert {k: v for k, v in i1.items() if k != "esbeltez"} == \
               {k: v for k, v in i2.items() if k != "esbeltez"}


# --------------------------------------------------------------------------
# Codificacion de acciones
# --------------------------------------------------------------------------
def test_espacio_de_acciones_k2():
    env = GooPhysicsEnv(k=2)
    assert env.action_space.n == math.comb(MAX_NODOS, 2) * 2 * 2 * 2 == 1224


def test_codificacion_biyectiva():
    env = GooPhysicsEnv(k=2)
    for a in range(env.action_space.n):
        assert env.codificar(*env.decodificar(a)) == a


def test_indice_estable_entre_pasos():
    # la misma tupla conserva su indice aunque cambie la lista de validas
    env = GooPhysicsEnv(k=2, p_carga=0)
    env.reset(seed=0)
    a = env.codificar(0, 3, 100.0, 100.0, +1)
    antes = env.action_masks()
    env.step(env.codificar(1, 2, 200.0, 100.0, -1))
    assert a == 16 and env.decodificar(a) == (0, 3, 100.0, 100.0, +1)
    assert antes[a] and env.action_masks()[a]


# --------------------------------------------------------------------------
# reset y s0
# --------------------------------------------------------------------------
def test_reset_determinista():
    env1, env2 = GooPhysicsEnv(), GooPhysicsEnv()
    o1, i1 = env1.reset(seed=7)
    o2, i2 = env2.reset(seed=7)
    assert iguales(o1, o2)
    assert env1.observation_space.contains(o1)
    assert env1.np_random.random() == env2.np_random.random()


def test_mascara_s0_tiene_12_acciones():
    env = GooPhysicsEnv(k=2)
    env.reset(seed=0)
    m = env.action_masks()
    assert m.sum() == 12
    # coincide con generar_acciones sobre el K4 asentado
    acc, _ = generar_acciones(Estado(env.G, env.pos, env.restantes), 2,
                              REGLAS_T4)
    assert set(np.flatnonzero(m)) == {env.codificar(*x[:5]) for x in acc}


def test_s0_asentado_y_largos_de_reposo():
    env = GooPhysicsEnv()
    obs, _ = env.reset(seed=0)
    assert env.pos[2] != (150.0, 150.0)           # la fisica lo asento
    assert env.pos[0] == (0.0, 0.0)               # nodos 0 y 1 fijos
    lar = obs["largos"].reshape(MAX_NODOS, MAX_NODOS)
    assert lar[0, 1] == lar[1, 0] == pytest.approx(150.0)
    assert lar[0, 2] == pytest.approx(150 * math.sqrt(2))
    assert all("largo" in d for _, _, d in env.G.edges(data=True))


def test_k4_s0_casi_nunca_colapsa_con_carga():
    # cordura de p_carga = 0.2 (provisional): < 1% en 200 semillas
    env = GooPhysicsEnv()
    colapsos = 0
    for s in range(200):
        env.reset(seed=s)
        _, _, rotas = env.probar_carga()
        colapsos += bool(rotas)
    assert colapsos / 200 < 0.01


# --------------------------------------------------------------------------
# step
# --------------------------------------------------------------------------
def test_accion_invalida_lanza_error():
    env = GooPhysicsEnv()
    env.reset(seed=0)
    a = int(np.flatnonzero(~env.action_masks())[0])
    with pytest.raises(ValueError):
        env.step(a)


def test_step_agrega_nodo_con_largos():
    env = GooPhysicsEnv(p_carga=0)
    env.reset(seed=0)
    obs, r, term, trunc, info = env.step(
        env.codificar(1, 2, 200.0, 100.0, -1))
    assert env.G.number_of_nodes() == 5 and obs["restantes"] == 13
    assert env.G[4][1]["largo"] == 200.0 and env.G[4][2]["largo"] == 100.0
    assert not term and not trunc and r == 0.0
    assert info["m_carga"] == 0 and info["causa"] is None
    assert math.isnan(info["esbeltez"])


# Voladizo construido con acciones k=2 del entorno, sin carga: cada bola
# se aleja de la base a ~150 px de altura. Como en evidencia/pymunk.txt
# (CASO 2), la tension crece con el largo del voladizo; con el umbral de
# 12 pesos colapsa con la 5a bola (por su propio peso).
VOLADIZO = [(1, 2, 200.0, 100.0, -1), (2, 4, 100.0, 100.0, +1),
            (4, 5, 200.0, 200.0, -1), (4, 6, 200.0, 100.0, -1),
            (6, 7, 200.0, 200.0, +1)]


def test_voladizo_colapsa():
    env = GooPhysicsEnv(p_carga=0)
    env.reset(seed=0)
    picos = []
    for n, accion in enumerate(VOLADIZO, start=1):
        _, r, term, _, info = env.step(env.codificar(*accion))
        picos.append(info["tension_max"])
        if n < len(VOLADIZO):
            assert not term and info["rotas"] == 0
    assert term and r == 0.0
    assert info["causa"] == "colapso" and info["colapso_por"] == "propio"
    assert picos == sorted(picos)
    assert picos[-2] < MAXFORCE < picos[-1]
    assert not env.action_masks().any()


def test_voladizo_aguanta_con_umbral_mayor():
    env = GooPhysicsEnv(p_carga=0, theta=Theta(umbral=2 * MAXFORCE))
    env.reset(seed=0)
    for accion in VOLADIZO:
        _, _, term, _, info = env.step(env.codificar(*accion))
    assert not term and info["causa"] is None


@pytest.mark.parametrize("restantes, r, causa", [(14, 13.0, "meta"),
                                                  (5, 4.0, "meta"),
                                                  (4, 0.0, "meta_sin_bolas")])
def test_meta_y_recompensa(monkeypatch, restantes, r, causa):
    # meta baja (200 px) para alcanzarla en un paso: r = n si n >= 4,
    # si no 0 con causa "meta_sin_bolas"
    monkeypatch.setattr("wog.env.META_Y", 200.0)
    env = GooPhysicsEnv(p_carga=0)
    env.reset(seed=0)
    env.restantes = restantes
    _, r_obs, term, _, info = env.step(env.codificar(2, 3, 200.0, 100.0, -1))
    assert info["altura"] >= 200.0
    assert term and r_obs == r and info["causa"] == causa


def test_shaping_telescopico():
    # gamma = 1 y Phi = 0 en terminales: la suma del shaping es -Phi(s0)
    base = GooPhysicsEnv()
    con = GooPhysicsEnv(shaping=True)
    t0, t1 = jugar(base, 3, 3), jugar(con, 3, 3)
    phi0 = t0[0][3]["altura"] / META_Y
    suma = sum(r for _, _, r, _ in t1) - sum(r for _, _, r, _ in t0)
    assert suma == pytest.approx(-phi0)


# --------------------------------------------------------------------------
# Reproducibilidad y clonacion
# --------------------------------------------------------------------------
def test_misma_semilla_misma_trayectoria():
    t1 = jugar(GooPhysicsEnv(), 11, 5)
    t2 = jugar(GooPhysicsEnv(), 11, 5)
    assert any(i["m_carga"] > 0 for _, _, _, i in t1[1:])   # W activa
    mismas(t1, t2)


def test_modelo_carga_uniforme():
    env = GooPhysicsEnv(modelo_carga="uniforme")
    mismas(jugar(env, 2, 2), jugar(GooPhysicsEnv(modelo_carga="uniforme"),
                                   2, 2))
    with pytest.raises(ValueError):
        GooPhysicsEnv(modelo_carga="otro")


def test_clone_restore_misma_continuacion():
    env = GooPhysicsEnv()
    jugar(env, 4, 4, pasos=2)
    snap = env.clone_state()
    rng = np.random.default_rng(9)
    acciones, salidas = [], []
    for _ in range(3):
        a = int(rng.choice(np.flatnonzero(env.action_masks())))
        acciones.append(a)
        salidas.append(env.step(a))
        if salidas[-1][2]:
            break
    for _ in range(2):                              # el snapshot se reutiliza
        env.restore_state(snap)
        for a, (o, r, term, trunc, info) in zip(acciones, salidas):
            o2, r2, term2, trunc2, info2 = env.step(a)
            assert iguales(o, o2) and (r, term, trunc) == (r2, term2, trunc2)
            assert info["tension_max"] == info2["tension_max"]
            assert info["m_carga"] == info2["m_carga"]
