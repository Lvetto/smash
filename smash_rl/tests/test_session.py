from smash_rl.session import MeleeSession, MeleeConfig
import numpy as np
import pytest

@pytest.mark.dolphin
def test_melee_session():
    config = MeleeConfig.from_env()
    assert isinstance(config, MeleeConfig), "La configurazione dell'ambiente non è stata creata correttamente."
    
    with MeleeSession(config=config) as session:

        session.advance_to_in_game(dbg=True)

        assert session.is_in_match, "La partita non è iniziata correttamente."
        assert isinstance(session.positions, np.ndarray), "Le posizioni dei giocatori non sono disponibili."
        assert isinstance(session.stocks, np.ndarray), "Le vite dei giocatori non sono disponibili."
        assert isinstance(session.percents, np.ndarray), "I percentuali dei giocatori non sono disponibili."
        assert isinstance(session.controller_ports, np.ndarray), "Le porte dei controller non sono disponibili."

        session.hard_reset()
        session.advance_to_in_game(dbg=True)
        assert session.is_in_match, "La partita non è iniziata correttamente dopo il reset."
