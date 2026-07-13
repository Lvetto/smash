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

# mappa con solo A+B
BUTTONS_A_B = {0: None, 1: melee.Button.BUTTON_A, 2: melee.Button.BUTTON_B}

# nomi leggibili delle direzioni dello stick, allineati agli indici di STICK_MAP.
# Fonte canonica: usati dal pannello dei video per etichettare azioni e neuroni.
STICK_LABELS = {
    0: "neutro", 1: "su", 2: "giu", 3: "sx", 4: "dx",
    5: "su-sx", 6: "su-dx", 7: "giu-sx", 8: "giu-dx",
}


def button_label(button) -> str:
    """Etichetta breve di un bottone (None -> '-', BUTTON_A -> 'A', ...)."""
    if button is None:
        return "-"
    return button.name.replace("BUTTON_", "")


ACT_SPECS = {}
# etichette parallele a ACT_SPECS, popolate da register_act:
#   ACT_LABELS[name][action]      -> stringa piatta per azione (es. "giu + B")
#   ACT_LAYOUTS[name]             -> {"stick_labels": [...], "button_labels": [...]}
#                                    per disegnare la griglia azioni (righe=stick, colonne=bottoni)
ACT_LABELS = {}
ACT_LAYOUTS = {}

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

    # etichette per il pannello dei video (stesso ordine di decode: action = s_idx * n_b + b_idx)
    stick_labels = [STICK_LABELS.get(s, str(stick_map[s])) for s in sorted(stick_map)]
    button_labels = [button_label(button_map[b]) for b in sorted(button_map)]
    ACT_LAYOUTS[name] = {"stick_labels": stick_labels, "button_labels": button_labels}
    labels = []
    for action in range(space.n):
        s_idx, b_idx = divmod(action, n_b)
        btn = button_labels[b_idx]
        labels.append(f"{stick_labels[s_idx]} + {btn}" if btn != "-" else stick_labels[s_idx])
    ACT_LABELS[name] = labels

register_act("full",   STICK_MAP, BUTTONS_FULL)     # Discrete(54)
register_act("a_only", STICK_MAP, BUTTONS_A_ONLY)   # Discrete(18)
register_act("a_b",    STICK_MAP, BUTTONS_A_B)      # Discrete(27)