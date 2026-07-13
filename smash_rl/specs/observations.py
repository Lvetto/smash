import numpy as np
from gymnasium import spaces
from gymnasium.spaces import Box

STAGE_X_MAX, STAGE_Y_MAX = 85.0, 50.0
VEL_NORM = 5.0
PCT_NORM = 300.0
DIST_NORM = 100.0

OBS_SPECS = {}
# nomi (brevi) delle feature, paralleli a OBS_SPECS e nello stesso ordine della
# concatenazione prodotta da ciascuna build function. Usati dal pannello dei
# video per etichettare i neuroni di input. "ag" = agente, "av" = avversario.
OBS_FEATURE_NAMES = {}

# blocchi riutilizzati: pos (x, y) + vel (voluntary x/y, knockback x/y) per giocatore
_POS_VEL_NAMES = [
    "x ag", "y ag", "vx ag", "vy ag", "vx kb ag", "vy kb ag",
    "x av", "y av", "vx av", "vy av", "vx kb av", "vy kb av",
]

# decoratore che registra una funzione di costruzione delle osservazioni, assegnandole un nome nel dizionario OBS_SPECS
def register_obs(name, space, feature_names=None):
    """
    Decoratore che registra una funzione di costruzione delle osservazioni, assegnandole un nome nel dizionario OBS_SPECS.

    Args:
        name (str): Il nome dell'osservazione da registrare.
        space (gymnasium.spaces.Space): Lo spazio delle osservazioni corrispondente.
        feature_names (list[str] | None): nomi delle feature, uno per dimensione,
            nell'ordine di concatenazione (per etichettare i neuroni nel pannello).

    Returns:
        function: La funzione decorata che costruisce le osservazioni.
    """

    def deco(fn):
        OBS_SPECS[name] = (space, fn)
        if feature_names is not None:
            OBS_FEATURE_NAMES[name] = list(feature_names)
        return fn
    return deco

# -- observation basate sulle properties della session (ctx.session) --

def _pos_vel_feats(ctx):
    """Blocchi [posizione, velocità] normalizzati di agente e avversario, dalle properties della session."""
    s = ctx.session
    pos = s.positions / np.array([STAGE_X_MAX, STAGE_Y_MAX])   # (n_players, 2)
    vel = s.velocities / VEL_NORM                              # (n_players, 4)
    ai, oi = ctx.agent_port - 1, ctx.opp_port - 1              # porta (da 1) -> indice (da 0)
    return np.concatenate([pos[ai], vel[ai], pos[oi], vel[oi]])

@register_obs("pos_vel", spaces.Box(-1.0, 1.0, (12,), np.float32), _POS_VEL_NAMES)
def build_pos_vel(gs, ctx):
    """Solo posizioni e velocità dei due giocatori: 2 * (2 pos + 4 vel) = 12 feature."""
    return np.clip(_pos_vel_feats(ctx), -1.0, 1.0).astype(np.float32)

@register_obs("pos_vel_stats", spaces.Box(-1.0, 1.0, (17,), np.float32),
              _POS_VEL_NAMES + ["% ag", "stock ag", "% av", "stock av", "dist"])
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

@register_obs("full_v1", spaces.Box(-1.0, 1.0, (32,), np.float32))
def build_full_v1(gs, ctx):     # da non usare, per ora rimane per non dover cambiare i default altrove
    raise NotImplementedError(
        "usa 'pos_vel' o 'pos_vel_stats', questa verra rimossa a breve"
    )

PERCENT_NORM = 150.0    # tecnicamente è sui 300, ma il grosso della partita avviene sotto al 150%
MAX_STOCKS = 4.0

@register_obs("pos_vel_facing_state", Box(low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32),  # 12 (pos/vel) + 6
              _POS_VEL_NAMES + ["facing ag", "facing av", "% ag", "% av", "stock ag", "stock av"])
def pos_vel_facing_state(gs, ctx):
    base = _pos_vel_feats(ctx)                     # 12 dim, riusata as-is
    s = ctx.session
    ai, oi = ctx.agent_port - 1, ctx.opp_port - 1
    facing  = s.facings                            # già ±1
    percent = s.percents / PERCENT_NORM
    stock   = s.stocks   / MAX_STOCKS
    extra = np.array([facing[ai],  facing[oi],
                      percent[ai], percent[oi],
                      stock[ai],   stock[oi]], dtype=np.float32)
    return np.concatenate([base, extra])

@register_obs("pos_vel_distances_stats", Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32),  # 12 (pos/vel) + 8
              _POS_VEL_NAMES + ["facing ag", "facing av", "% ag", "% av",
                                "stock ag", "stock av", "dist", "dist"])
def pos_vel_distances_stats(gs, ctx):
    base = _pos_vel_feats(ctx)                     # 12 dim, riusata as-is
    s = ctx.session
    ai, oi = ctx.agent_port - 1, ctx.opp_port - 1
    facing  = s.facings                            # già ±1
    percent = s.percents / PERCENT_NORM
    stock   = s.stocks   / MAX_STOCKS
    distance = s.distance / DIST_NORM
    extra = np.array([facing[ai],  facing[oi],
                      percent[ai], percent[oi],
                      stock[ai],   stock[oi],
                      distance, distance], dtype=np.float32)
    return np.concatenate([base, extra])

# observation "completa" che include: posizioni, velocità (tutte e 4), stocks, danni, facing (convertito da bool a float tra -1 e 1),
#                                     posizione relativa dell'avversario (dx, dy; dx col segno del facing dell'agente), frame rimanenti di hitstun,
#                                     numero di salti rimanenti, on_ground, off_stage, invulnerable (booleani di stato)
# costanti per la regolarizzazione (empiriche, basate su un analisi dei replay)
full_obs_spec_costants = {
    "norm_x_pos": 100,
    "norm_y_pos": 50,
    "norm_x_vel_voluntary": 2,
    "norm_y_vel_voluntary": 2,
    "norm_x_vel_knockback": 5,
    "norm_y_vel_knockback": 5,
    "norm_percent": 150,
    "norm_stocks": 4,
    "norm_hitstun": 50,
    "norm_jumps": 2,
}

def _get_full_obs_features(ctx):
    """Costruisce le features dell'osservazione completa, normalizzate secondo le costanti definite sopra."""
    s = ctx.session
    ai, oi = ctx.agent_port - 1, ctx.opp_port - 1

    # Posizioni e velocità normalizzate
    pos = s.positions / np.array([full_obs_spec_costants["norm_x_pos"], full_obs_spec_costants["norm_y_pos"]])
    vel = s.velocities / np.array([full_obs_spec_costants["norm_x_vel_voluntary"], full_obs_spec_costants["norm_y_vel_voluntary"],
                                   full_obs_spec_costants["norm_x_vel_knockback"], full_obs_spec_costants["norm_y_vel_knockback"]])

    # Percentuali e stocks normalizzati
    percent = s.percents / full_obs_spec_costants["norm_percent"]
    stock = s.stocks / full_obs_spec_costants["norm_stocks"]

    # facing (la classe session o converte già in ±1)
    facing = s.facings.astype(np.float32)

    # Posizione relativa dell'avversario (già normalizzata, pos è in unità di norm_x_pos/norm_y_pos)
    dx = (pos[oi][0] - pos[ai][0]) * facing[ai]   # >0 = avversario davanti
    dy = pos[oi][1] - pos[ai][1]                   # niente facing sull'asse verticale

    # Hitstun, salti rimanenti e stati booleani
    hitstun = s.hitstun_frames / full_obs_spec_costants["norm_hitstun"]
    jumps_remaining = s.jumps_left / full_obs_spec_costants["norm_jumps"]
    on_ground = s.on_ground.astype(np.float32)
    off_stage = s.off_stage.astype(np.float32)
    invulnerable = s.invulnerable.astype(np.float32)

    # Combinazione di tutte le features in un unico array. Gli scalari vanno racchiusi in
    # un array 1-D (np.array([...])): np.concatenate non accetta scalari 0-D indicizzati
    # direttamente da array (percent[ai], facing[ai], ecc.) insieme ad array 1-D.
    scalars_ai = np.array([percent[ai], stock[ai], facing[ai]], dtype=np.float32)
    scalars_oi = np.array([percent[oi], stock[oi], facing[oi]], dtype=np.float32)
    shared = np.array([
        dx, dy, hitstun[ai], hitstun[oi],
        jumps_remaining[ai], jumps_remaining[oi],
        on_ground[ai], on_ground[oi],
        off_stage[ai], off_stage[oi],
        invulnerable[ai], invulnerable[oi],
    ], dtype=np.float32)

    features = np.concatenate([
        pos[ai], vel[ai], scalars_ai,
        pos[oi], vel[oi], scalars_oi,
        shared,
    ])

    return features

_FULL_OBS_NAMES = [
    # agente: pos, vel, %, stock, facing
    "x ag", "y ag", "vx ag", "vy ag", "vx kb ag", "vy kb ag", "% ag", "stock ag", "facing ag",
    # avversario: idem
    "x av", "y av", "vx av", "vy av", "vx kb av", "vy kb av", "% av", "stock av", "facing av",
    # condivise
    "dx rel", "dy rel", "hitstun ag", "hitstun av", "salti ag", "salti av",
    "terra ag", "terra av", "fuori ag", "fuori av", "invuln ag", "invuln av",
]

@register_obs("full_obs", Box(low=-np.inf, high=np.inf, shape=(30,), dtype=np.float32), _FULL_OBS_NAMES)
def build_full_obs(gs, ctx):
    """Costruisce l'osservazione completa, normalizzata secondo le costanti definite sopra."""
    features = _get_full_obs_features(ctx)
    return np.clip(features, -2.0, 2.0).astype(np.float32)  # le normalizzazioni sono per lo più 95esimi percentili, quindi +-2 dovrebbe bastare

