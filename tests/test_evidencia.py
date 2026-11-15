"""
Regresion: cada script de Taller 2 debe reproducir byte a byte su archivo
en evidencia/. Si un refactor cambia la salida, este test lo detecta.
"""

import subprocess
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parent.parent

CASOS = [
    ("fase1_paso_a_paso.py", ["--rapido"], "fase1.txt"),
    ("prueba_pymunk.py", [], "pymunk.txt"),
    ("prueba_gymnasium.py", [], "gymnasium.txt"),
]


@pytest.mark.parametrize("script, args, evidencia", CASOS,
                         ids=[c[0] for c in CASOS])
def test_salida_igual_a_evidencia(script, args, evidencia):
    r = subprocess.run([sys.executable, script, *args], cwd=RAIZ,
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    esperado = (RAIZ / "evidencia" / evidencia).read_text()
    assert r.stdout == esperado
