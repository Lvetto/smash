"""
Gioca in tempo reale contro un agente addestrato (vedi smash_rl/play.py).

Uso:
  python play.py --configure-pad                       # prima volta: mappa il gamepad
  python play.py checkpoints/<run>/<checkpoint>.zip    # gioca
"""
from smash_rl.play import main

if __name__ == "__main__":
    main()
