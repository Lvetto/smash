import melee

REWARD_FNS = {}

def register_reward(name):
    """
    Decoratore che registra una funzione di calcolo della ricompensa, assegnandole un nome nel dizionario REWARD_FNS.

    Args:
        name (str): Il nome della funzione di ricompensa da registrare.

    Returns:
        function: La funzione decorata che calcola la ricompensa.
    """

    def deco(fn):
        REWARD_FNS[name] = fn
        return fn
    return deco

W_DMG, W_STOCK = 0.01, 1.0

@register_reward("v1")
def reward_v1(prev_gs, gs, ctx):
    """
    Reward simmetrico frame-a-frame: +0.01 per punto di danno inflitto (-0.01 se subito),
    +1 per stock tolto (-1 se perso). Con ~100 danni per stock i due termini sono comparabili.
    """
    if prev_gs is None or gs.frame < 0:   # nessuno stato precedente o countdown di inizio match
        return 0.0

    me,  opp  = gs.players[ctx.agent_port],      gs.players[ctx.opp_port]
    pme, popp = prev_gs.players[ctx.agent_port], prev_gs.players[ctx.opp_port]

    d_opp_pct = int(opp.percent) - int(popp.percent)   # >0 = ho fatto danno (buono)
    d_my_pct  = int(me.percent)  - int(pme.percent)    # >0 = ho subito danno (cattivo)
    d_opp_stk = int(popp.stock)  - int(opp.stock)      # >0 = ho tolto uno stock (buono)
    d_my_stk  = int(pme.stock)   - int(me.stock)       # >0 = ho perso uno stock (cattivo)

    # dopo la perdita di una vita la percentuale torna a 0: quel delta non è danno reale
    if d_opp_stk != 0: d_opp_pct = 0
    if d_my_stk  != 0: d_my_pct  = 0
    # stock aumentato = respawn/restart, non è una kill
    if d_opp_stk < 0:  d_opp_stk = d_my_stk = 0
    if d_my_stk  < 0:  d_my_stk  = d_opp_stk = 0

    return W_DMG * (d_opp_pct - d_my_pct) + W_STOCK * (d_opp_stk - d_my_stk)


W_DMG_2 = 0.001
W_STOCK_2 = 1.0
@register_reward("v2")
def reward_v2(prev_gs, gs, ctx):
    """
    Versione di v1 che da più peso agli stock rispetto ai danni
    """

    if prev_gs is None or gs.frame < 0:   # nessuno stato precedente o countdown di inizio match
        return 0.0

    me,  opp  = gs.players[ctx.agent_port],      gs.players[ctx.opp_port]
    pme, popp = prev_gs.players[ctx.agent_port], prev_gs.players[ctx.opp_port]

    d_opp_pct = int(opp.percent) - int(popp.percent)   # >0 = ho fatto danno (buono)
    d_my_pct  = int(me.percent)  - int(pme.percent)    # >0 = ho subito danno (cattivo)
    d_opp_stk = int(popp.stock)  - int(opp.stock)      # >0 = ho tolto uno stock (buono)
    d_my_stk  = int(pme.stock)   - int(me.stock)       # >0 = ho perso uno stock (cattivo)

    # dopo la perdita di una vita la percentuale torna a 0: quel delta non è danno reale
    if d_opp_stk != 0: d_opp_pct = 0
    if d_my_stk  != 0: d_my_pct  = 0
    # stock aumentato = respawn/restart, non è una kill
    if d_opp_stk < 0:  d_opp_stk = d_my_stk = 0
    if d_my_stk  < 0:  d_my_stk  = d_opp_stk = 0

    return W_DMG_2 * (d_opp_pct - d_my_pct) + W_STOCK_2 * (d_opp_stk - d_my_stk)


# Classificatore d'attacco della libreria (il costruttore legge dei CSV: va creato una volta sola).
_FRAMEDATA = melee.FrameData()
_ATTACK_CACHE: dict = {}   # memo (character, action) -> bool

def _is_attacking(player) -> bool:
    """True se il player sta eseguendo un'animazione d'attacco (con hitbox), grab inclusi.

    getattr con fallback a None: se il player non espone .action/.character (es. un fake
    gamestate nei test) la reward degrada a v2 invece di crashare, e un'eccezione qui
    appenderebbe il worker di training. In produzione PlayerState.action/.character esistono.
    """
    action = getattr(player, "action", None)
    character = getattr(player, "character", None)
    if action is None or character is None:
        return False
    key = (character, action)
    hit = _ATTACK_CACHE.get(key)
    if hit is None:
        hit = _FRAMEDATA.is_attack(character, action)
        _ATTACK_CACHE[key] = hit
    return hit


W_ATTACK_3 = 0.001

@register_reward("v3")
def reward_v3(prev_gs, gs, ctx):
    """
    Come v2 (danni/stock), più una penalità W_ATTACK_3 all'inizio di ogni attacco
    dell'agente, per scoraggiare gli attacchi a vuoto.

    L'inizio dell'attacco è rilevato come edge sull'animazione (attacca ora e non nel
    frame precedente): la reward è chiamata su frame consecutivi, quindi la penalità
    scatta una sola volta per attacco senza bisogno di stato globale.
    """
    r = reward_v2(prev_gs, gs, ctx)
    if prev_gs is None or gs.frame < 0:   # nessuno stato precedente o countdown di inizio match
        return r

    me  = gs.players[ctx.agent_port]
    pme = prev_gs.players[ctx.agent_port]
    if _is_attacking(me) and not _is_attacking(pme):
        r -= W_ATTACK_3
    return r


W_ATTACK_4 = 0.0001

@register_reward("v4")
def reward_v4(prev_gs, gs, ctx):
    """
    Come v2 (danni/stock), più una penalità W_ATTACK_4 all'inizio di ogni attacco
    dell'agente, per scoraggiare gli attacchi a vuoto.

    L'inizio dell'attacco è rilevato come edge sull'animazione (attacca ora e non nel
    frame precedente): la reward è chiamata su frame consecutivi, quindi la penalità
    scatta una sola volta per attacco senza bisogno di stato globale.
    """
    r = reward_v2(prev_gs, gs, ctx)
    if prev_gs is None or gs.frame < 0:   # nessuno stato precedente o countdown di inizio match
        return r

    me  = gs.players[ctx.agent_port]
    pme = prev_gs.players[ctx.agent_port]
    if _is_attacking(me) and not _is_attacking(pme):
        r -= W_ATTACK_4
    return r
