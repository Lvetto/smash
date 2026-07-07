from dataclasses import dataclass

@dataclass(frozen=True)
class Ctx:
    """
    Tutto il contesto necessario per l'esecuzione di un episodio di allenamento o valutazione.

    Attributes:
        agent_port (int): La porta del giocatore agente.
        opp_port (int): La porta del giocatore avversario.
    """
    agent_port: int = 1
    opp_port: int = 2
