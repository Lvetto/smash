# specs/actions.py
from gymnasium import spaces
import melee

STICK_MAP = {   # 0.5 equivale a stick in posizione neutra, 0.0 a stick completamente spostato in una direzione, 1.0 a stick completamente spostato nella direzione opposta
    0: (0.5, 0.5), 1: (0.5, 1.0), 2: (0.5, 0.0), 3: (0.0, 0.5), 4: (1.0, 0.5),
    5: (0.0, 1.0), 6: (1.0, 1.0), 7: (0.0, 0.0), 8: (1.0, 0.0),
}

# mappa completa dei bottoni, con None che indica nessun bottone premuto
BUTTONS_FULL   = {0: None, 1: melee.Button.BUTTON_A, 2: melee.Button.BUTTON_B,
                  3: melee.Button.BUTTON_X, 4: melee.Button.BUTTON_Z, 5: melee.Button.BUTTON_L}

# mappa ridotta per test
BUTTONS_A_ONLY = {0: None, 1: melee.Button.BUTTON_A}

ACT_SPECS = {}

# registra una funzione di decodifica delle azioni, assegnandole un nome nel dizionario ACT_SPECS
def register_act(name, stick_map, button_map):
    """
    Registra una funzione di decodifica delle azioni, assegnandole un nome nel dizionario ACT_SPECS.

    Args:
        name (str): Il nome dell'azione da registrare.
        stick_map (dict): Mappa dei movimenti dello stick analogico.
        button_map (dict): Mappa dei bottoni disponibili.
    
    """
    n_b = len(button_map)
    space = spaces.Discrete(len(stick_map) * n_b)

    def decode(action):
        s_idx, b_idx = divmod(action, n_b)
        return stick_map[s_idx], button_map[b_idx]  # -> ((x, y), button)
    
    ACT_SPECS[name] = (space, decode)

register_act("full",   STICK_MAP, BUTTONS_FULL)     # Discrete(54)
register_act("a_only", STICK_MAP, BUTTONS_A_ONLY)   # Discrete(18)
