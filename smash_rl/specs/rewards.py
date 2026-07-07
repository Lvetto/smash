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
    pass

"""
@register_reward("v1")
def reward_v1(prev_gs, gs, ctx):
    if prev_gs is None or gs.frame < 0:
        return 0.0
    me,  opp  = gs.players[ctx.agent_port],  gs.players[ctx.opp_port]
    pme, popp = prev_gs.players[ctx.agent_port], prev_gs.players[ctx.opp_port]

    d_opp_pct = int(opp.percent) - int(popp.percent)
    d_my_pct  = int(me.percent)  - int(pme.percent)
    d_opp_stk = int(popp.stock)  - int(opp.stock)
    d_my_stk  = int(pme.stock)   - int(me.stock)

    if d_opp_stk != 0: d_opp_pct = 0.0
    if d_my_stk  != 0: d_my_pct  = 0.0
    if d_opp_stk < 0:  d_opp_stk = d_my_stk = 0
    if d_my_stk  < 0:  d_my_stk  = d_opp_stk = 0

    return W_DMG * (d_opp_pct - d_my_pct) + W_STOCK * (d_opp_stk - d_my_stk)

"""