"""
Rendering del pannello laterale di diagnostica (un'immagine BGR per frame).

Tutto disegnato con primitive opencv: a 60 record al secondo matplotlib
costerebbe ~15 ms/frame contro <1 ms di cv2 (matplotlib viene usato solo una
volta, per campionare la colormap). Sezioni, dall'alto:

  - controller: sagoma GameCube con stick e bottone dell'azione greedy evidenziati;
  - grafico Q-values su finestra scorrevole (argmax evidenziato);
  - griglia delle Q-value del frame corrente (righe = stick, colonne = bottoni);
  - diagramma della rete (colonne di neuroni colorati per attivazione), con i
    nomi delle feature sugli input e l'etichetta dell'azione sull'argmax.

Le etichette (nomi feature, layout azioni) sono opzionali: se non fornite al
costruttore il pannello ripiega su un rendering non etichettato (barplot delle
azioni, neuroni anonimi), così i vecchi call site continuano a funzionare.
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

    Args:
        height, width: dimensioni del pannello (height = altezza del video).
        obs_labels: nomi delle feature di input (uno per dimensione dell'obs),
            per etichettare i neuroni di input; None = neuroni anonimi.
        act_labels: nome per azione (es. "giu + B"), per l'azione corrente e
            l'argmax nel diagramma; None = "azione N".
        act_layout: {"stick_labels": [...], "button_labels": [...]} per la
            griglia azioni (righe = stick, colonne = bottoni); None = barplot.
    """

    def __init__(self, height: int, width: int = 480, *,
                 qval_window: int = 180, max_units_per_column: int = 32,
                 obs_labels: list | None = None,
                 act_labels: list | None = None,
                 act_layout: dict | None = None):
        self.h, self.w = height, width
        self.window = qval_window
        self.max_units = max_units_per_column
        self.q_history: deque = deque(maxlen=qval_window)        # array (n_actions,)

        self.obs_labels = obs_labels
        self.act_labels = act_labels
        self.act_layout = act_layout

        # ripartizione verticale delle sezioni
        self.y_controller = 0
        self.y_qplot = int(height * 0.20)
        self.y_grid = int(height * 0.42)
        self.y_net = int(height * 0.62)
        self.margin = 10

    # -- helper etichette --

    def _act_label(self, action: int) -> str:
        if self.act_labels is not None and 0 <= action < len(self.act_labels):
            return self.act_labels[action]
        return f"azione {action}"

    # -- sezioni --

    def _draw_button(self, img, label, center, radius, pressed) -> None:
        """Bottone della sagoma: pieno _ACCENT se premuto, altrimenti contorno _DIM."""
        cx, cy = center
        on = (label == pressed)
        if on:
            cv2.circle(img, (cx, cy), radius, _ACCENT, -1, cv2.LINE_AA)
        else:
            cv2.circle(img, (cx, cy), radius, _DIM, 1, cv2.LINE_AA)
        (tw, th), _ = cv2.getTextSize(label, _FONT, 0.4, 1)
        _put(img, label, (cx - tw // 2, cy + th // 2), 0.4, _BG if on else _DIM)

    def _draw_trigger(self, img, label, box, pressed) -> None:
        """Trigger/dorsale (L, Z) come rettangolino in alto."""
        x0, y0, x1, y1 = box
        on = (label == pressed)
        cv2.rectangle(img, (x0, y0), (x1, y1), _ACCENT if on else _DIM, -1 if on else 1)
        (tw, th), _ = cv2.getTextSize(label, _FONT, 0.4, 1)
        _put(img, label, ((x0 + x1) // 2 - tw // 2, (y0 + y1) // 2 + th // 2),
             0.4, _BG if on else _DIM)

    def _draw_controller(self, img, rec) -> None:
        """Sagoma GameCube: stick analogico (posizione corrente) + cluster bottoni."""
        stick = rec["stick"]
        button = rec["button"]
        pressed = button.name.replace("BUTTON_", "") if button is not None else None

        top, bottom = self.y_controller + 2, self.y_qplot - 20

        # dorsali in alto: L a sinistra, Z a destra
        self._draw_trigger(img, "L", (self.margin, top, self.margin + 70, top + 18), pressed)
        self._draw_trigger(img, "Z", (self.w - self.margin - 70, top,
                                       self.w - self.margin, top + 18), pressed)

        body_top = top + 26
        cy = (body_top + bottom) // 2

        # stick analogico (sinistra): gate ottagonale + posizione corrente
        gate_cx = self.margin + 55
        rg = max(min(42, (bottom - body_top) // 2 - 2), 12)
        oct_pts = [(int(gate_cx + rg * np.cos(np.pi / 8 + k * np.pi / 4)),
                    int(cy - rg * np.sin(np.pi / 8 + k * np.pi / 4))) for k in range(8)]
        cv2.polylines(img, [np.array(oct_pts, np.int32)], True, _DIM, 1, cv2.LINE_AA)
        cv2.circle(img, (gate_cx, cy), 3, _DIM, -1)
        sx = int(gate_cx + (stick[0] - 0.5) * 2 * rg)   # x: 0=sx, 1=dx
        sy = int(cy - (stick[1] - 0.5) * 2 * rg)        # y: 1=su (schermo invertito)
        cv2.line(img, (gate_cx, cy), (sx, sy), _ACCENT, 1, cv2.LINE_AA)
        cv2.circle(img, (sx, sy), 7, _ACCENT, -1, cv2.LINE_AA)

        # cluster bottoni (destra): A grande al centro, B/X/Y attorno
        bcx, bcy, u = self.w - 100, cy, 15
        self._draw_button(img, "A", (bcx, bcy), int(1.2 * u), pressed)
        self._draw_button(img, "B", (bcx - int(1.5 * u), bcy + int(0.7 * u)), int(0.7 * u), pressed)
        self._draw_button(img, "X", (bcx + int(1.4 * u), bcy - int(0.2 * u)), int(0.7 * u), pressed)
        self._draw_button(img, "Y", (bcx + int(0.2 * u), bcy - int(1.5 * u)), int(0.7 * u), pressed)

        # nome dell'azione corrente sotto la sagoma
        _put(img, self._act_label(rec["action"]), (self.margin, bottom + 15), 0.5, _ACCENT)

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
        y0, y1 = self.y_qplot, self.y_grid - 8
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

    def _draw_action_grid(self, img, rec) -> None:
        """Griglia delle Q-value del frame corrente: righe = direzioni stick,
        colonne = bottoni; colore per Q normalizzata a [min, max], argmax evidenziato."""
        y0, y1 = self.y_grid, self.y_net - 8
        x0, x1 = self.margin, self.w - self.margin
        _put(img, "azioni (Q-value)", (x0 + 4, y0 + 14), 0.4, _DIM)

        stick_labels = self.act_layout["stick_labels"]
        button_labels = self.act_layout["button_labels"]
        n_rows, n_cols = len(stick_labels), len(button_labels)

        q = rec["q_values"]
        argmax = rec["action"]
        lo, hi = float(q.min()), float(q.max())
        span = max(hi - lo, 1e-6)

        row_gutter = 44        # etichette righe (direzioni stick)
        grid_x0 = x0 + row_gutter
        grid_y0 = y0 + 34      # titolo + header colonne
        cell_w = (x1 - grid_x0) / n_cols
        cell_h = (y1 - grid_y0) / n_rows

        # header colonne (bottoni)
        for c, blab in enumerate(button_labels):
            cx = int(grid_x0 + cell_w * (c + 0.5))
            (tw, _), _ = cv2.getTextSize(blab, _FONT, 0.35, 1)
            _put(img, blab, (cx - tw // 2, grid_y0 - 4), 0.35, _DIM)

        for r, slab in enumerate(stick_labels):
            ry = int(grid_y0 + cell_h * (r + 0.5))
            _put(img, slab, (x0 + 2, ry + 4), 0.32, _DIM)
            for c in range(n_cols):
                action = r * n_cols + c   # come divmod(action, n_b): s_idx=r, b_idx=c
                v = (float(q[action]) - lo) / span
                cx0, cy0 = int(grid_x0 + cell_w * c) + 1, int(grid_y0 + cell_h * r) + 1
                cx1, cy1 = int(grid_x0 + cell_w * (c + 1)) - 1, int(grid_y0 + cell_h * (r + 1)) - 1
                cv2.rectangle(img, (cx0, cy0), (cx1, cy1), _color(v), -1)
                if action == argmax:
                    cv2.rectangle(img, (cx0, cy0), (cx1, cy1), _ACCENT, 2)

    def _draw_action_bars(self, img, rec) -> None:
        """Fallback senza layout: barplot delle Q-value del frame corrente, una
        barra per azione, altezza relativa a [min, max] (ranking), argmax evidenziato."""
        y0, y1 = self.y_grid, self.y_net - 8
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
        _put(img, "rete (attivazioni)", (self.margin + 4, y0 + 14), 0.4, _DIM)
        top = y0 + 22

        # colonne: input (con label), hidden (sottocampionate), output (tutte le azioni)
        obs = np.asarray(rec["obs"], dtype=np.float32)
        input_col = self._column(obs, lambda v: (v + 1.0) / 2.0)
        hidden_cols = [self._column(a, lambda v: np.tanh(v / 2.0))
                       for a in (rec["activations"] or [])]
        q = np.asarray(rec["q_values"], dtype=np.float32)
        out_col = (q - q.min()) / max(np.ptp(q), 1e-6)   # niente sottocampionamento: argmax preciso
        cols = [input_col] + hidden_cols + [out_col]
        n_cols = len(cols)

        # gutter a sinistra per i nomi delle feature (solo se etichette allineate ai neuroni)
        labels_aligned = (self.obs_labels is not None
                          and len(input_col) == len(obs) == len(self.obs_labels))
        label_gutter = 78 if labels_aligned else 0
        x_positions = np.linspace(self.margin + 30 + label_gutter,
                                  self.w - self.margin - 30, n_cols)

        argmax = rec["action"]
        for ci, (col, cx) in enumerate(zip(cols, x_positions)):
            is_input, is_output = ci == 0, ci == n_cols - 1
            n = len(col)
            ys = np.linspace(top + 6, y1 - 6, n) if n > 1 else np.array([(top + y1) / 2])
            r = int(np.clip((ys[1] - ys[0]) / 2 + 1, 2, 7)) if n > 1 else 6
            for ui, (v, cyf) in enumerate(zip(col, ys)):
                cyi = int(cyf)
                cv2.circle(img, (int(cx), cyi), r, _color(float(v)), -1)
                if is_input and labels_aligned:
                    _put(img, self.obs_labels[ui], (self.margin + 2, cyi + 3), 0.3, _DIM)
                if is_output and ui == argmax:
                    cv2.circle(img, (int(cx), cyi), r + 3, _ACCENT, 2)
                    lbl = self._act_label(argmax)
                    if self.act_labels is not None:
                        (tw, _), _ = cv2.getTextSize(lbl, _FONT, 0.3, 1)
                        _put(img, lbl, (int(cx) - r - 4 - tw, cyi + 3), 0.3, _ACCENT)

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

        self._draw_controller(img, record)
        self._draw_qplot(img, record)
        if self.act_layout is not None:
            self._draw_action_grid(img, record)
        else:
            self._draw_action_bars(img, record)
        if record.get("activations") is not None:
            self._draw_net(img, record)
        return img
