import argparse
import configparser
import os
import subprocess
from pathlib import Path

import melee
from dotenv import load_dotenv
from stable_baselines3 import DQN

from smash_rl.session import MeleeConfig, MeleeSession, PlayerSpec
from smash_rl.specs.actions import ACT_SPECS
from smash_rl.specs.context import Ctx
from smash_rl.specs.observations import OBS_SPECS

DEFAULT_PAD_CONFIG = Path("configs/gcpad_human.ini")
PAD_HOME = Path("configs/dolphin_pad_home")  # home persistente solo per mappare il pad


def run_match(session, predict, ctx, build_obs, decode_act, frame_skip=3):
    """
    Gioca una partita: una decisione ogni frame_skip frame, input applicato con la
    stessa logica press/release di MeleeEnv._do_step (press sul primo frame della
    finestra). Ritorna l'ultimo gamestate quando il match finisce o si esce dall'IN_GAME.
    """
    if len(session.stocks):
        session.old_stocks = list(session.stocks)   # match_over non deve scattare su stock stantii

    k = 0
    (x, y), button = (0.5, 0.5), None
    gs = session._gamestate
    while True:
        if gs is not None:
            if gs.menu_state != melee.Menu.IN_GAME or session.match_over:
                return gs
            if k % frame_skip == 0:
                obs = build_obs(gs, ctx)
                (x, y), button = decode_act(int(predict(obs)))
            session.apply_input(player_idx=ctx.agent_port - 1, button=button,
                                stick_x=x, stick_y=y, press=(k % frame_skip == 0))
            k += 1
        gs = session.step()   # senza FFW blocca fino al frame successivo: dà il ritmo a 60fps


def _extract_pad_section(src_ini: Path, out_path: Path) -> None:
    """Estrae la sezione [GCPad1] mappata nella GUI e la salva come template portabile."""
    parser = configparser.ConfigParser()
    parser.optionxform = str   # i nomi delle chiavi di Dolphin vanno preservati
    parser.read(src_ini)

    if not parser.has_section("GCPad1") or not parser["GCPad1"].get("Device"):
        raise RuntimeError(
            f"nessun mapping trovato in {src_ini}: in Dolphin configura la Porta 1 "
            "(Standard Controller -> Configure) e salva prima di chiudere"
        )

    out = configparser.ConfigParser()
    out.optionxform = str
    out.add_section("GCPad")   # il nome della sezione è irrilevante: conta solo il mapping
    for key, value in parser["GCPad1"].items():
        out.set("GCPad", key, value)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        out.write(f)


def configure_pad(out_path: Path = DEFAULT_PAD_CONFIG) -> None:
    """Apre la GUI di Dolphin per mappare il gamepad, poi salva il mapping in out_path."""
    load_dotenv()
    exe = MeleeConfig._resolve_dolphin_exe(Path(os.environ["DOLPHIN_NETPLAY_DIR"]))
    PAD_HOME.mkdir(parents=True, exist_ok=True)

    print("Si apre Dolphin: Options -> Controller Settings -> Port 1 -> Standard Controller")
    print("-> Configure, mappa il tuo gamepad, chiudi la finestra di config e poi Dolphin.")
    subprocess.run([str(exe), "-u", str(PAD_HOME.absolute())])

    _extract_pad_section(PAD_HOME / "Config" / "GCPadNew.ini", out_path)
    print(f"Mapping salvato in {out_path}")


def _print_result(session, agent_char):
    stocks = session.stocks
    if len(stocks) < 2:
        print("Partita interrotta.")
        return
    if stocks[1] <= 0:
        print(f"L'agente ({agent_char.name}) ha vinto: {int(stocks[0])} stock rimasti.")
    elif stocks[0] <= 0:
        print(f"Hai vinto! {int(stocks[1])} stock rimasti.")
    else:
        print(f"Match resettato (stock: agente {int(stocks[0])}, tu {int(stocks[1])}).")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", nargs="?", help="checkpoint .zip dell'agente")
    parser.add_argument("--configure-pad", action="store_true",
                        help="mappa il gamepad nella GUI di Dolphin e salva il template")
    parser.add_argument("--char", default="FOX", help="personaggio dell'agente")
    parser.add_argument("--stage", default="FINAL_DESTINATION")
    parser.add_argument("--obs", default="pos_vel", choices=sorted(OBS_SPECS))
    parser.add_argument("--act", default="a_only", choices=sorted(ACT_SPECS))
    parser.add_argument("--frame-skip", type=int, default=3)
    parser.add_argument("--pad-config", type=Path, default=DEFAULT_PAD_CONFIG)
    args = parser.parse_args(argv)

    if args.configure_pad:
        configure_pad(args.pad_config)
        return
    if args.model is None:
        parser.error("serve il path di un checkpoint .zip (o --configure-pad)")

    # buffer_size=1: senza, load riallocherebbe il replay buffer intero (inutile per giocare)
    model = DQN.load(args.model, device="cpu", custom_objects={"buffer_size": 1})

    obs_space, build_obs = OBS_SPECS[args.obs]
    act_space, decode_act = ACT_SPECS[args.act]
    if model.observation_space.shape != obs_space.shape:
        raise SystemExit(f"il checkpoint si aspetta obs {model.observation_space.shape}, "
                         f"ma '{args.obs}' produce {obs_space.shape}: usa --obs giusto")
    if model.action_space.n != act_space.n:
        raise SystemExit(f"il checkpoint ha {model.action_space.n} azioni, "
                         f"ma '{args.act}' ne definisce {act_space.n}: usa --act giusto")

    agent_char = melee.Character[args.char.upper()]
    stage = melee.Stage[args.stage.upper()]

    opponent_char = melee.Character.MARTH

    config = MeleeConfig.for_play(human_pad_config=args.pad_config)
    session = MeleeSession(config=config,
                           players=[PlayerSpec(agent_char, cpu_level=0),
                                    PlayerSpec(opponent_char, human=False, cpu_level=7)])
    ctx = Ctx(agent_port=1, opp_port=2, session=session)
    predict = lambda obs: model.predict(obs, deterministic=True)[0]

    print(f"Agente: {agent_char.name} (porta 1) - tu sei sulla porta 2. Ctrl-C per uscire.")
    try:
        with session:
            while True:
                print("Scegli il personaggio col tuo pad e avvia la partita "
                      "(l'agente si sceglie da solo)...")
                session.advance_to_in_game(stage=stage, timeout=None)  # i menu li guida l'umano
                print("Partita avviata!")
                run_match(session, predict, ctx, build_obs, decode_act,
                          frame_skip=args.frame_skip)
                _print_result(session, agent_char)
    except KeyboardInterrupt:
        print("\nChiusura...")


if __name__ == "__main__":
    main()
