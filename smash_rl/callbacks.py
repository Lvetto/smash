from stable_baselines3.common.callbacks import BaseCallback


class EpisodeMetricsCallback(BaseCallback):
    """
    Logga su tensorboard stock e danni di fine episodio (medie sul rollout).
    Richiede che gli env siano wrappati in Monitor con
    info_keywords=("P1_stocks", "P2_stocks", "P1_percent", "P2_percent")
    (è quello che fa factory.make_env).
    """

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            ep = info.get("episode")
            if ep is None:
                continue  # questo env non ha appena finito un episodio

            self.logger.record_mean("rollout/agent_stock_end", ep["P1_stocks"])
            self.logger.record_mean("rollout/agent_percent_end", ep["P1_percent"])
            self.logger.record_mean("rollout/opp_stock_end", ep["P2_stocks"])
            self.logger.record_mean("rollout/opp_percent_end", ep["P2_percent"])
        return True
