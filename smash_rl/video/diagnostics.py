"""
Diagnostica del modello su un replay .slp, senza avviare Dolphin.

libmelee sa leggere i .slp direttamente (Console(is_dolphin=False)): il loop
di step() produce gli stessi GameState del gioco live, quindi riusiamo le
observation function di specs/observations.py così come sono.

NOTA: la policy greedy ricalcolata qui può differire dagli input registrati
nel replay (durante il training c'erano frame_skip ed esplorazione, e il
checkpoint può essere successivo alla partita): i record dicono cosa farebbe
*ora* il modello su quegli stati, non cosa fece allora.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import melee
import numpy as np
import torch
from melee.slpfilestreamer import SLPFileStreamer
from stable_baselines3 import DQN

from smash_rl.session import MeleeSession
from smash_rl.specs.actions import ACT_SPECS
from smash_rl.specs.context import Ctx
from smash_rl.specs.observations import OBS_SPECS

# header .slp: oggetto ubjson { "raw": array di uint8 con lunghezza int32 ... }
_SLP_HEADER = b"{U\x03raw[$U#l"
_METADATA_MARKER = b"U\x08metadata"


class LenientSLPFileStreamer(SLPFileStreamer):
    """
    Come SLPFileStreamer, ma legge anche i replay NON finalizzati.

    I .slp scritti durante il training non vengono chiusi correttamente
    (Dolphin viene ucciso, non fermato): la lunghezza dell'elemento "raw"
    resta al placeholder 0 e manca il blocco "metadata", quindi ubjson
    fallisce. I dati dei frame però ci sono: qui li estraiamo a mano.
    """

    def connect(self):
        try:
            return super().connect()    # file finalizzato: percorso normale
        except Exception:
            pass

        data = Path(self._path).read_bytes()
        if not data.startswith(_SLP_HEADER):
            raise ValueError(f"non è un file .slp: {self._path}")

        length = int.from_bytes(data[11:15], "big")
        start = len(_SLP_HEADER) + 4
        if length > 0:
            raw = data[start:start + length]
        else:
            # non finalizzato: il raw arriva fino al metadata (se c'è) o a EOF
            end = data.find(_METADATA_MARKER, start)
            raw = data[start:end if end != -1 else len(data)]

        self._contents = raw
        return True


class ReplaySessionShim(MeleeSession):
    """
    Sostituto minimo di MeleeSession per rileggere i replay: niente Dolphin,
    solo il _gamestate corrente. Eredita le properties vere (positions,
    velocities, stocks, percents, distance), che sono ciò che le observation
    function leggono via ctx.session. Stesso pattern di tests/helpers.FakeSession.
    """

    def __init__(self, n_players: int = 2):
        # niente super().__init__(): nessuna config reale, nessun Dolphin
        self.players = [None] * n_players  # serve solo len() in _player_values
        self.old_stocks = [4] * n_players
        self._gamestate = None
        self._external_console = None
        self.console = None
        self.display = None

    def set_gamestate(self, gs) -> None:
        self._gamestate = gs


def iter_replay_gamestates(slp_path):
    """
    Genera i GameState di un replay .slp, nell'ordine, fino a EOF.
    Un errore di parsing a metà file (replay troncato dal kill di Dolphin)
    viene trattato come fine del replay, con un warning.
    """
    slp_path = Path(slp_path)
    if not slp_path.is_file():
        raise FileNotFoundError(f"replay non trovato: {slp_path}")
    console = melee.Console(path=str(slp_path), is_dolphin=False,
                            allow_old_version=True)
    console._slippstream = LenientSLPFileStreamer(str(slp_path))
    console.connect()
    while True:
        try:
            gs = console.step()
        except Exception as e:
            warnings.warn(f"replay troncato ({slp_path.name}): parsing interrotto ({e})")
            return
        if gs is None:
            return
        yield gs


def slp_frame_bounds(slp_path) -> tuple[int, int]:
    """(primo, ultimo) indice di frame del replay (il primo è tipicamente -123)."""
    first = last = None
    for gs in iter_replay_gamestates(slp_path):
        if first is None:
            first = int(gs.frame)
        last = int(gs.frame)
    if first is None:
        raise ValueError(f"nessun frame nel replay: {slp_path}")
    return first, last


def load_model(model_path) -> DQN:
    """Carica un checkpoint DQN per sola inferenza (senza riallocare il replay buffer)."""
    return DQN.load(str(model_path), device="cpu", custom_objects={"buffer_size": 1})


def _validate_specs(model, obs_name: str, act_name: str):
    """Stessi controlli di play.py: il checkpoint deve combaciare con gli specs."""
    obs_space, build_obs = OBS_SPECS[obs_name]
    act_space, decode_act = ACT_SPECS[act_name]
    if model.observation_space.shape != obs_space.shape:
        raise ValueError(f"il checkpoint si aspetta obs {model.observation_space.shape}, "
                         f"ma '{obs_name}' produce {obs_space.shape}: usa obs_name giusto")
    if model.action_space.n != act_space.n:
        raise ValueError(f"il checkpoint ha {model.action_space.n} azioni, "
                         f"ma '{act_name}' ne definisce {act_space.n}: usa act_name giusto")
    return build_obs, decode_act


def _hook_activations(q_net_seq):
    """
    Registra forward hook su ogni modulo di attivazione del Sequential della
    QNetwork; ritorna (lista_output, lista_handle). Gli output vengono
    sovrascritti a ogni forward, nell'ordine dei layer.
    """
    captured = []
    handles = []

    def make_hook(idx):
        def hook(_module, _inp, out):
            captured[idx] = out.detach().numpy().squeeze(0).copy()
        return hook

    act_types = (torch.nn.ReLU, torch.nn.Tanh, torch.nn.Sigmoid, torch.nn.ELU,
                 torch.nn.LeakyReLU)
    for module in q_net_seq:
        if isinstance(module, act_types):
            captured.append(None)
            handles.append(module.register_forward_hook(make_hook(len(captured) - 1)))
    return captured, handles


def extract_frame_records(slp_path, model_path, *,
                          obs_name: str = "pos_vel", act_name: str = "a_only",
                          agent_port: int = 1, opp_port: int = 2,
                          capture_activations: bool = True) -> dict[int, dict]:
    """
    Rilegge il replay e calcola, per ogni frame in partita, cosa 'pensa' il
    modello: osservazione, Q-values, azione greedy decodificata e (opzionale)
    attivazioni dei layer nascosti.

    Ritorna un dict {frame: record} con chiavi = gs.frame (parte da -123;
    l'azione di gioco inizia a frame 0). Ogni record contiene:
      obs (float32[obs_dim]), q_values (float32[n_actions]), action (int),
      stick ((x, y)), button (melee.Button | None),
      activations (list di float32[n_units], una per layer nascosto, o None),
      percents, stocks (tuple agente/avversario), positions ((n,2)), distance.
    """
    model = load_model(model_path)
    build_obs, decode_act = _validate_specs(model, obs_name, act_name)

    n_players = max(agent_port, opp_port)
    shim = ReplaySessionShim(n_players=n_players)
    ctx = Ctx(agent_port=agent_port, opp_port=opp_port, session=shim)
    ai, oi = agent_port - 1, opp_port - 1

    q_net = model.policy.q_net
    captured, handles = ([], [])
    if capture_activations:
        captured, handles = _hook_activations(q_net.q_net)

    records: dict[int, dict] = {}
    try:
        with torch.no_grad():
            for gs in iter_replay_gamestates(slp_path):
                if gs.menu_state != melee.Menu.IN_GAME:
                    continue
                if not all(p in gs.players for p in range(1, n_players + 1)):
                    continue

                shim.set_gamestate(gs)
                obs = build_obs(gs, ctx)
                obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                q = q_net(obs_t).numpy().squeeze(0)
                action = int(q.argmax())
                stick, button = decode_act(action)

                records[int(gs.frame)] = {
                    "obs": obs,
                    "q_values": q.astype(np.float32),
                    "action": action,
                    "stick": stick,
                    "button": button,
                    "activations": [a for a in captured] if capture_activations else None,
                    "percents": (shim.percents[ai], shim.percents[oi]),
                    "stocks": (shim.stocks[ai], shim.stocks[oi]),
                    "positions": shim.positions.copy(),
                    "distance": shim.distance,
                }
    finally:
        for h in handles:
            h.remove()

    if not records:
        raise ValueError(f"nessun frame in partita nel replay: {slp_path}")
    return records
