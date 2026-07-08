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

