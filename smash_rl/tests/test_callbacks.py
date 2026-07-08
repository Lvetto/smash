"""Unit test di EpisodeMetricsCallback con logger e infos finti."""
from types import SimpleNamespace

from smash_rl.callbacks import EpisodeMetricsCallback


def test_episode_metrics_recorded_only_on_episode_end():
    records = []
    cb = EpisodeMetricsCallback()
    cb.model = SimpleNamespace(  # BaseCallback.logger delega a self.model.logger
        logger=SimpleNamespace(record_mean=lambda key, val: records.append((key, val))))
    cb.locals = {"infos": [
        {"episode": {"P1_stocks": 2, "P2_stocks": 0, "P1_percent": 50.0, "P2_percent": 0.0}},
        {},  # env che non ha appena finito un episodio: va ignorato
    ]}

    assert cb._on_step() is True

    assert dict(records) == {
        "rollout/agent_stock_end": 2,
        "rollout/agent_percent_end": 50.0,
        "rollout/opp_stock_end": 0,
        "rollout/opp_percent_end": 0.0,
    }
