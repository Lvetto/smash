"""
Rendering del pannello laterale di diagnostica (un'immagine BGR per frame).

Tutto disegnato con primitive opencv: a 60 record al secondo matplotlib
costerebbe ~15 ms/frame contro <1 ms di cv2 (matplotlib viene usato solo una
volta, per campionare la colormap). Sezioni, dall'alto:

  - testo: azione greedy, stick+bottone, percent/stock;
  - grafico Q-values su finestra scorrevole (argmax evidenziato);
  - barplot delle Q-value del frame corrente, una barra per azione (ranking);
  - diagramma della rete (colonne di neuroni colorati per attivazione).
"""
from __future__ import annotations

from collections import deque

import cv2
import numpy as np

# LUT viridis (256, 3) in BGR: matplotlib serve solo qui, a import time
from matplotlib import colormaps

_VIRIDIS_BGR = (colormaps["viridis"](np.linspace(0, 1, 256))[:, 2::-1] * 255).astype(np.uint8)

_BG = (24, 24, 24)
_FG = (230, 230, 230)
_DIM = (110, 110, 110)
_AGENT_COLOR = (80, 220, 120)   # verde
_OPP_COLOR = (80, 100, 235)     # rosso
_ACCENT = (60, 200, 255)        # giallo/arancio
_FONT = cv2.FONT_HERSHEY_SIMPLEX

_STICK_NAMES = {
    (0.5, 0.5): "neutro", (0.5, 1.0): "su", (0.5, 0.0): "giu",
    (0.0, 0.5): "sinistra", (1.0, 0.5): "destra",
    (0.0, 1.0): "su-sx", (1.0, 1.0): "su-dx",
    (0.0, 0.0): "giu-sx", (1.0, 0.0): "giu-dx",
}


def _color(v: float) -> tuple:
    """v in [0,1] -> colore BGR dalla LUT viridis."""
    idx = int(np.clip(v, 0.0, 1.0) * 255)
    return tuple(int(c) for c in _VIRIDIS_BGR[idx])


def _put(img, text, xy, scale=0.45, color=_FG, thick=1):
    cv2.putText(img, text, xy, _FONT, scale, color, thick, cv2.LINE_AA)


class PanelRenderer:
    """
    Disegna il pannello frame per frame. Tiene internamente la storia per i
    grafici scorrevoli: chiamare render() in ordine di frame.
    """

    def __init__(self, height: int, width: int = 480, *,
                 qval_window: int = 180, max_units_per_column: int = 32):
        self.h, self.w = height, width
        self.window = qval_window
        self.max_units = max_units_per_column
        self.q_history: deque = deque(maxlen=qval_window)        # array (n_actions,)

        # ripartizione verticale delle sezioni
        self.y_text = 0
        self.y_qplot = int(height * 0.22)
        self.y_barplot = int(height * 0.47)
        self.y_net = int(height * 0.68)
        self.margin = 10

    # -- sezioni --

    def _draw_text(self, img, rec) -> None:
        x, y = self.margin, self.y_text + 22
        stick, button = rec["stick"], rec["button"]
        stick_name = _STICK_NAMES.get(tuple(stick), str(stick))
        btn_name = button.name.replace("BUTTON_", "") if button is not None else "-"
        _put(img, f"azione {rec['action']}: stick {stick_name}  bottone {btn_name}",
             (x, y), 0.5, _ACCENT)
        _put(img, f"maxQ {rec['q_values'].max():+.3f}", (x, y + 22), 0.45)
        _put(img, f"agente  {rec['percents'][0]:5.1f}%  stock {int(rec['stocks'][0])}",
             (x, y + 44), 0.45, _AGENT_COLOR)
        _put(img, f"avvers. {rec['percents'][1]:5.1f}%  stock {int(rec['stocks'][1])}",
             (x, y + 66), 0.45, _OPP_COLOR)

        # mini-box con la posizione dello stick
        box = 60
        bx, by = self.w - box - self.margin, self.y_text + 12
        cv2.rectangle(img, (bx, by), (bx + box, by + box), _DIM, 1)
        sx = int(bx + stick[0] * box)
        sy = int(by + (1.0 - stick[1]) * box)   # y dello stick: 1.0 = su
        cv2.circle(img, (bx + box // 2, by + box // 2), 2, _DIM, -1)
        cv2.circle(img, (sx, sy), 5, _ACCENT, -1)

    def _draw_series(self, img, y0, y1, series, colors, *, y_range=None, title=""):
        """Polilinee su finestra scorrevole. series: lista di array (t,)."""
        x0, x1 = self.margin, self.w - self.margin
        cv2.rectangle(img, (x0, y0), (x1, y1), _DIM, 1)
        if title:
            _put(img, title, (x0 + 4, y0 + 16), 0.4, _DIM)
        if not series or len(series[0]) < 2:
            return
        allv = np.concatenate(series)
        lo, hi = (allv.min(), allv.max()) if y_range is None else y_range
        if hi - lo < 1e-6:
            hi = lo + 1e-6
        t = len(series[0])
        xs = x0 + 2 + (np.arange(t) * (x1 - x0 - 4) / max(self.window - 1, 1))
        for vals, color in zip(series, colors):
            ys = y1 - 2 - (np.clip(vals, lo, hi) - lo) * (y1 - y0 - 4) / (hi - lo)
            pts = np.stack([xs, ys], axis=1).astype(np.int32)
            cv2.polylines(img, [pts], False, color, 1, cv2.LINE_AA)
        _put(img, f"{hi:+.2f}", (x1 - 58, y0 + 16), 0.35, _DIM)
        _put(img, f"{lo:+.2f}", (x1 - 58, y1 - 6), 0.35, _DIM)

    def _draw_qplot(self, img, rec) -> None:
        y0, y1 = self.y_qplot, self.y_barplot - 8
        qh = np.array(self.q_history)          # (t, n_actions)
        if qh.size == 0:
            return
        argmax = rec["action"] if rec is not None else int(qh[-1].argmax())
        series = [qh[:, a] for a in range(qh.shape[1])]
        colors = [_DIM] * qh.shape[1]
        colors[argmax] = _ACCENT
        # l'argmax per ultimo, sopra le altre linee
        order = [a for a in range(qh.shape[1]) if a != argmax] + [argmax]
        self._draw_series(img, y0, y1, [series[a] for a in order],
                          [colors[a] for a in order], title="Q-values")

    def _draw_action_bars(self, img, rec) -> None:
        """Barplot delle Q-value del frame corrente: una barra per azione,
        altezza relativa a [min, max] tra le azioni (ranking), argmax evidenziato."""
        y0, y1 = self.y_barplot, self.y_net - 8
        x0, x1 = self.margin, self.w - self.margin
        cv2.rectangle(img, (x0, y0), (x1, y1), _DIM, 1)
        _put(img, "azioni (Q-value)", (x0 + 4, y0 + 16), 0.4, _DIM)

        q = rec["q_values"]
        n = len(q)
        argmax = rec["action"]
        lo, hi = float(q.min()), float(q.max())
        span = max(hi - lo, 1e-6)

        label_gutter = 46   # spazio riservato alle etichette hi/lo, fuori dalle barre
        bars_x1 = x1 - label_gutter
        plot_y0, plot_y1 = y0 + 22, y1 - 4
        plot_h = plot_y1 - plot_y0
        slot = (bars_x1 - x0 - 4) / n
        bar_w = max(int(slot * 0.7), 1)
        for a in range(n):
            cx = int(x0 + 2 + slot * a + slot / 2)
            bar_h = int((float(q[a]) - lo) / span * plot_h)
            color = _ACCENT if a == argmax else _DIM
            cv2.rectangle(img, (cx - bar_w // 2, plot_y1 - bar_h),
                          (cx + bar_w // 2, plot_y1), color, -1)
        _put(img, f"{hi:+.2f}", (bars_x1 + 6, y0 + 16), 0.35, _DIM)
        _put(img, f"{lo:+.2f}", (bars_x1 + 6, y1 - 6), 0.35, _DIM)

    def _column(self, values, norm):
        """Sottocampiona una colonna di attivazioni e la normalizza in [0,1]."""
        v = np.asarray(values, dtype=np.float32)
        if len(v) > self.max_units:
            step = len(v) // self.max_units
            v = v[::step][:self.max_units]
        return norm(v)

    def _draw_net(self, img, rec) -> None:
        y0, y1 = self.y_net, self.h - self.margin
        _put(img, "rete (attivazioni)", (self.margin + 4, y0 + 16), 0.4, _DIM)
        y0 += 22

        # colonne: obs in [-1,1] -> [0,1]; hidden ReLU con tanh; Q normalizzati
        cols = [self._column(rec["obs"], lambda v: (v + 1.0) / 2.0)]
        for act in (rec["activations"] or []):
            cols.append(self._column(act, lambda v: np.tanh(v / 2.0)))
        q = rec["q_values"]
        cols.append(self._column(q, lambda v: (v - v.min()) / max(np.ptp(v), 1e-6)))

        x_positions = np.linspace(self.margin + 30, self.w - self.margin - 30, len(cols))
        argmax = rec["action"]
        for ci, (col, cx) in enumerate(zip(cols, x_positions)):
            n = len(col)
            ys = np.linspace(y0 + 8, y1 - 8, n)
            r = int(np.clip((ys[1] - ys[0]) / 2 + 1, 3, 7)) if n > 1 else 6
            for ui, (v, cy) in enumerate(zip(col, ys)):
                cv2.circle(img, (int(cx), int(cy)), r, _color(float(v)), -1)
                if ci == len(cols) - 1 and ui == argmax:
                    cv2.circle(img, (int(cx), int(cy)), r + 3, _ACCENT, 2)

    # -- API --

    def render(self, record: dict | None) -> np.ndarray:
        """
        Pannello per un frame. record=None (frame senza dati: pre-partita,
        buco nel replay) -> pannello spento; la storia dei grafici non avanza.
        """
        img = np.full((self.h, self.w, 3), _BG, np.uint8)
        cv2.line(img, (0, 0), (0, self.h), _DIM, 1)

        if record is None:
            _put(img, "nessun dato", (self.margin, 30), 0.6, _DIM)
            return img

        self.q_history.append(record["q_values"])

        self._draw_text(img, record)
        self._draw_qplot(img, record)
        self._draw_action_bars(img, record)
        if record.get("activations") is not None:
            self._draw_net(img, record)
        return img
