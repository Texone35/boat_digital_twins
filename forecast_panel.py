"""
Dock de prévision : pilonnement, roulis et tangage — passé réel, prévision
LSQ et vérité sur l'horizon, fin de la zone de prédictibilité, période calme.
Redessiné à chaque nouvelle prévision seulement (toutes les ~5 s).
"""

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QLabel, QVBoxLayout, QWidget

from waveforecast.forecast import CALM_HEAVE, CALM_ROLL, HISTORY, HORIZON

# (indice d'état, libellé, facteur d'affichage, seuil de période calme)
_DOF = [(2, 'Pilonnement (m)', 1.0, CALM_HEAVE),
        (3, 'Roulis (°)', np.degrees(1.0), np.degrees(CALM_ROLL)),
        (4, 'Tangage (°)', np.degrees(1.0), None)]


class ForecastPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.status = QLabel('Prévision en attente des premières mesures…')
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.RichText)
        self.fig = Figure(figsize=(9, 3.2), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.axes = self.fig.subplots(1, 3, sharex=True)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.addWidget(self.status)
        lay.addWidget(self.canvas, 1)

    def show_forecast(self, fc, history):
        """fc : waveforecast.forecast.Forecast ; history : [(t, heave, roll, pitch)]."""
        hist = np.array(history) if history else np.empty((0, 4))
        tau = fc.times - fc.t0
        for ax, (j, label, k, thr) in zip(self.axes, _DOF):
            ax.clear()
            if len(hist):
                ax.plot(hist[:, 0] - fc.t0, hist[:, j - 1] * k, 'k', lw=1.5, label='réel (passé)')
            ax.plot(tau, fc.truth[:, j] * k, color='0.6', lw=1.2, ls='--', label='vérité')
            ax.plot(tau, fc.predicted[:, j] * k, 'C0', lw=1.8, label='prévu (LSQ)')
            ax.axvline(0, color='k', lw=0.8)
            ax.axvspan(fc.zone_end, HORIZON, color='C3', alpha=0.08)
            ax.axvline(fc.zone_end, color='C3', ls=':', lw=1.2)
            if thr is not None:
                for s in (-1, 1):
                    ax.axhline(s * thr, color='C2', ls=':', lw=0.8)
            ax.set_title(label, fontsize=9)
            ax.set_xlim(-HISTORY, HORIZON)
            ax.tick_params(labelsize=8)
            ax.set_xlabel('t − maintenant (s)', fontsize=8)
        self.axes[0].legend(fontsize=7, loc='upper left')
        self.canvas.draw_idle()

        verdict = ('<b style="color:#1e8449">OUI</b>' if fc.calm
                   else '<b style="color:#c0392b">NON</b>')
        self.status.setText(
            f'Prévision émise à t = {fc.t0:.0f} s — zone de prédictibilité : '
            f'<b>{fc.zone_end:.0f} s</b> (au-delà, zone rouge : houle jamais mesurée). '
            f'Période calme sur {fc.calm_window:.0f} s (|pilonnement| &lt; {CALM_HEAVE:g} m, '
            f'|roulis| &lt; {np.degrees(CALM_ROLL):g}°) : {verdict}')
