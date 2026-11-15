"""
El notebook de Taller 2 queda intacto. Se ejecuta completo y se compara la
salida de texto de cada celda con la guardada en el .ipynb. Las imagenes no
se comparan (dependen del backend de matplotlib), pero si su text/plain
(por ejemplo "<Figure size 1100x550 with 2 Axes>").
"""

from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

RAIZ = Path(__file__).resolve().parent.parent
NOTEBOOK = RAIZ / "Fase1_World_of_Goo.ipynb"


def salidas_texto(celda):
    """Lista de (tipo, texto) de las salidas de una celda."""
    out = []
    for o in celda.get("outputs", []):
        if o["output_type"] == "stream":
            out.append((o["name"], o["text"]))
        elif o["output_type"] in ("display_data", "execute_result"):
            out.append((o["output_type"], o["data"].get("text/plain", "")))
        elif o["output_type"] == "error":
            out.append(("error", f"{o['ename']}: {o['evalue']}"))
    # stdout puede llegar partido en varios mensajes: se fusionan los
    # streams consecutivos del mismo tipo
    fusion = []
    for tipo, texto in out:
        if fusion and tipo in ("stdout", "stderr") and fusion[-1][0] == tipo:
            fusion[-1] = (tipo, fusion[-1][1] + texto)
        else:
            fusion.append((tipo, texto))
    return fusion


@pytest.fixture(scope="module")
def ejecutado():
    nb = nbformat.read(NOTEBOOK, as_version=4)
    NotebookClient(nb, timeout=300, kernel_name="python3",
                   resources={"metadata": {"path": str(RAIZ)}}).execute()
    return nb


def test_notebook_misma_salida(ejecutado):
    guardado = nbformat.read(NOTEBOOK, as_version=4)
    pares = [(i, a, b) for i, (a, b) in
             enumerate(zip(guardado.cells, ejecutado.cells))
             if a.cell_type == "code"]
    for i, a, b in pares:
        assert salidas_texto(b) == salidas_texto(a), f"celda {i}"
