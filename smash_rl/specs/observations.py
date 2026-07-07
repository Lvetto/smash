import numpy as np
from gymnasium import spaces

STAGE_X_MAX, STAGE_Y_MAX = 85.0, 50.0
VEL_NORM = 5.0

OBS_SPECS = {}

# decoratore che registra una funzione di costruzione delle osservazioni, assegnandole un nome nel dizionario OBS_SPECS
def register_obs(name, space):
    """
    Decoratore che registra una funzione di costruzione delle osservazioni, assegnandole un nome nel dizionario OBS_SPECS.

    Args:
        name (str): Il nome dell'osservazione da registrare.
        space (gymnasium.spaces.Space): Lo spazio delle osservazioni corrispondente.
    
    Returns:
        function: La funzione decorata che costruisce le osservazioni.
    """

    def deco(fn):
        OBS_SPECS[name] = (space, fn)
        return fn
    return deco

@register_obs("full_v1", spaces.Box(-1.0, 1.0, (32,), np.float32))
def build_full_v1(gs, ctx):
    pass


"""
def _player_block(p, other):
    pass
    #return feats  # len == 14

@register_obs("full_v1", spaces.Box(-1.0, 1.0, (32,), np.float32))
def build_full_v1(gs, ctx):
    p1, p2 = gs.players[ctx.agent_port], gs.players[ctx.opp_port]
    feats = _player_block(p1, p2) + _player_block(p2, p1)      # 14 + 14
    feats += [                                                 # + 4 globali
        (p1.position.x - p2.position.x) / (2 * STAGE_X_MAX),
        (p1.position.y - p2.position.y) / (2 * STAGE_Y_MAX),
        gs.distance / 100.0,
        gs.frame / 28800.0,
    ]
    return np.clip(np.asarray(feats, np.float32), -1.0, 1.0)

@register_obs("minimal", spaces.Box(-1.0, 1.0, (12,), np.float32))
def build_minimal(gs, ctx):
    pass
"""