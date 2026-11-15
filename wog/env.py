"""
Entorno Gymnasium del Taller 4 (pendiente de implementar).

Debe seguir el MDP de CLAUDE.md, no el WorldOfGooEnv de prueba_gymnasium.py:
  s = (G, pos, n); un paso = colocar bola + estabilizar; carga movil W
  sorteada cada turno, fuera del estado; recompensa n si llega con n >= 4,
  si no 0; shaping F = Phi(s') - Phi(s), Phi = altura, Phi = 0 en terminales;
  gamma = 1, horizonte 14. Acciones generadas con geometria.REGLAS_T4.
"""
