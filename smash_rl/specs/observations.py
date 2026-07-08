import numpy as np
from gymnasium import spaces

STAGE_X_MAX, STAGE_Y_MAX = 85.0, 50.0
VEL_NORM = 5.0
PCT_NORM = 300.0
DIST_NORM = 100.0

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


# -- observation basate sulle properties della session (ctx.session) --

def _pos_vel_feats(ctx):
    """Blocchi [posizione, velocità] normalizzati di agente e avversario, dalle properties della session."""
    s = ctx.session
    pos = s.positions / np.array([STAGE_X_MAX, STAGE_Y_MAX])   # (n_players, 2)
    vel = s.velocities / VEL_NORM                              # (n_players, 4)
    ai, oi = ctx.agent_port - 1, ctx.opp_port - 1              # porta (da 1) -> indice (da 0)
    return np.concatenate([pos[ai], vel[ai], pos[oi], vel[oi]])

@register_obs("pos_vel", spaces.Box(-1.0, 1.0, (12,), np.float32))
def build_pos_vel(gs, ctx):
    """Solo posizioni e velocità dei due giocatori: 2 * (2 pos + 4 vel) = 12 feature."""
    return np.clip(_pos_vel_feats(ctx), -1.0, 1.0).astype(np.float32)

@register_obs("pos_vel_stats", spaces.Box(-1.0, 1.0, (17,), np.float32))
def build_pos_vel_stats(gs, ctx):
    """Posizioni, velocità, danni, vite e distanza: 2 * (2 pos + 4 vel + 1 danno + 1 vite) + 1 distanza = 17 feature."""
    s = ctx.session
    ai, oi = ctx.agent_port - 1, ctx.opp_port - 1
    pct = s.percents / PCT_NORM
    stk = s.stocks / 4.0
    feats = np.concatenate([
        _pos_vel_feats(ctx),
        [pct[ai], stk[ai], pct[oi], stk[oi]],
        [s.distance / DIST_NORM],
    ])
    return np.clip(feats, -1.0, 1.0).astype(np.float32)
