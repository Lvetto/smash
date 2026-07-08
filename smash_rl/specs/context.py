from dataclasses import dataclass

@dataclass(frozen=True)
class Ctx:
    """
    Tutto il contesto necessario per l'esecuzione di un episodio di allenamento o valutazione.

    Attributes:
        agent_port (int): La porta del giocatore agente.
        opp_port (int): La porta del giocatore avversario.
        session (MeleeSession): La sessione di gioco; le observation leggono le
            quantità (posizioni, velocità, danni, ...) dalle sue properties.
    """
    agent_port: int = 1
    opp_port: int = 2
    session: object = None
