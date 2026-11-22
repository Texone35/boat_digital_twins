"""
Main runner — jumeau numérique : mer JONSWAP + navire Cummins 6-DDL.

Fenêtre Qt : vue 3D PyVista au centre, panneau de contrôle à droite pour
modifier l'état de mer et la navigation en direct, et choisir le navire
(type et caractéristiques), remplacé à chaud.
"""

import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import QApplication, QDockWidget, QMainWindow
from pyvistaqt import QtInteractor

import jonswap as sea_mod
from boat import HydroModel
from boat_presets import PRESETS, DEFAULT_BOAT
from forecast_panel import ForecastPanel
from scene import Scene
from shipid import calibrate as calib
from simulation import Simulation
from ui_panel import ControlPanel
from waveforecast.forecast import ShipForecast

# Pas de physique (s) : une frame affichée au plus tous les DT de temps réel
DT = 0.05
# Pas de physique max. par frame affichée pour rattraper le temps réel
# quand une frame prend plus de DT à calculer et afficher.
MAX_SUBSTEPS = 4


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Jumeau numérique — mer JONSWAP + navire Cummins 6-DDL')
        self.resize(1600, 900)

        self.sea = sea_mod.SeaState()
        self.sim = Simulation(self.sea, PRESETS[DEFAULT_BOAT], dt=DT)

        # Rendu déclenché uniquement par la boucle (pas de timer auto)
        self.plotter = QtInteractor(self, auto_update=False)
        self.setCentralWidget(self.plotter.interactor)
        self.scene = Scene(self.plotter, self.sim)

        self.panel = ControlPanel(self.sea.params, self.sim.params, self._preview_boat)
        dock = QDockWidget('Contrôles', self)
        dock.setWidget(self.panel)
        dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

        # ── Mode Prévision ──
        # Une prévision (≈ 0.4 s de calcul) tourne dans un thread pendant que
        # le jumeau continue d'avancer ; _pending la référence jusqu'à son
        # résultat (mis à None pour ignorer une prévision devenue obsolète).
        self.forecast = None
        self._pending = None
        self._ghost_lead = 20.0
        self._executor = ThreadPoolExecutor(max_workers=1)
        self.fc_panel = ForecastPanel()
        self.fc_dock = QDockWidget('Prévision', self)
        self.fc_dock.setWidget(self.fc_panel)
        self.fc_dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.fc_dock)
        self.fc_dock.hide()

        # ── Calibration ──
        # model_params : ce que le jumeau croit du navire (panneau, prévision) ;
        # sim.params : le navire « réel » simulé (une variante cachée du
        # modèle si hidden). Le roulis est identifié toutes les 20 s.
        self.model_params = self.sim.params
        self.hidden = False
        self.monitor = calib.RollMonitor()
        self._calib_pending = None
        self._estimate = None            # BoatParams calibrés (dernière identification)
        self._revealed = False
        self._calib_shown = None         # dernier état affiché (rafraîchi ≤ 1 fois / s)
        self._connect_panel()

        # ── Boucle temps réel ──
        # Le temps simulé suit l'horloge réelle : si une frame a pris plus de
        # DT, la physique avance de plusieurs pas (MAX_SUBSTEPS au plus).
        # Timer précis à 5 ms : rattrape vite sans boucle active.
        self.paused = False
        self.follow = True
        self.frame = 0
        self.fps = 0.0
        self._t0 = time.perf_counter()
        self._last_render = self._t0
        self._last_xy = self.sim.boat_xy()
        self._needs_redraw = True
        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.PreciseTimer)
        self.timer.timeout.connect(self._tick)
        self.timer.start(5)

    # ── Panneau → simulation ──
    def _connect_panel(self):
        p = self.panel
        p.seaChanged.connect(self._on_sea_changed)
        p.speedChanged.connect(self.sim.set_speed)
        p.headingChanged.connect(lambda deg: self._redraw_after(
            self.sim.set_heading, np.radians(deg)))
        p.pauseToggled.connect(self._on_pause)
        p.resetRequested.connect(self._on_reset)
        p.recenterRequested.connect(lambda: self._redraw_after(self.scene.recenter_camera))
        p.followToggled.connect(lambda on: setattr(self, 'follow', on))
        p.boatApplied.connect(self._on_boat_applied)
        p.forecastToggled.connect(self._on_forecast)
        p.ghostLeadChanged.connect(lambda s: setattr(self, '_ghost_lead', s))
        p.hiddenToggled.connect(self._on_hidden)
        p.calibrateRequested.connect(self._on_calibrate)

    def _redraw_after(self, fn, *args):
        fn(*args)
        self._needs_redraw = True

    def _on_sea_changed(self, values):
        values = dict(values)
        if 'theta_mean' in values:
            values['theta_mean'] = np.radians(values['theta_mean'])
        self.sea.set_target(**values)
        self._reset_calibration()               # la fenêtre de roulis ne vaut plus
        if self.paused:
            # En pause, la transition est appliquée d'un coup pour être visible
            self.sea.advance(np.inf, ref=self.sim.boat_xy())
            self._needs_redraw = True

    def _on_boat_applied(self, params):
        self.model_params = params
        self.sim.set_boat(calib.hidden_ship(params) if self.hidden else params)
        self.sim.set_speed(params.speed)
        self._reset_calibration()
        if self.forecast:                       # reste en station, prévision relancée
            self.forecast.set_model(params, self.sim)
            self.forecast.place(self.sim)
            self.panel.speed_row.setValue(0.0)
            self._pending = None
        self.scene.set_boat(self.sim.boat)
        self.scene.recenter_camera()
        self._last_xy = self.sim.boat_xy()
        self._needs_redraw = True

    def _on_pause(self, paused):
        self.paused = paused
        if not paused:
            self._t0 = time.perf_counter() - self.frame * DT
        self._needs_redraw = True

    def _on_reset(self):
        if self.forecast:
            self.forecast.place(self.sim)
            self._pending = None
        else:
            self.sim.reset()
        self.scene.recenter_camera()
        self._last_xy = self.sim.boat_xy()
        self._needs_redraw = True

    def _on_forecast(self, on):
        if on:
            self.forecast = ShipForecast(self.sim)
            self.forecast.set_model(self.model_params, self.sim)
            self.panel.speed_row.setValue(0.0)
            self.scene.set_probes(self.forecast.x_probes)
            self.fc_dock.show()
        else:
            self.forecast.close(self.sim)
            self.forecast = None
            self.scene.set_probes(None)
            self.fc_dock.hide()
        self._pending = None
        self._reset_calibration()
        self.panel.set_sea_values(self.sea.params)
        self.panel.set_sea_locked(on)
        self.scene.on_sea_changed()
        self.scene.recenter_camera()
        self._last_xy = self.sim.boat_xy()
        self._needs_redraw = True

    def _forecast_tick(self):
        """Enregistre le mouvement, lance / récupère les prévisions, place le fantôme."""
        fc = self.forecast
        fc.record(self.sim)
        if self._pending is None:
            if fc.due(self.sim.t):
                self._pending = self._executor.submit(fc.compute, *fc.start(self.sim))
        elif self._pending.done():
            fc.last = self._pending.result()
            self._pending = None
            self.fc_panel.show_forecast(fc.last, list(fc.history))
        if fc.last is not None:
            self.scene.set_ghost(fc.last.state_at(self.sim.t + self._ghost_lead))

    # ── Calibration ──
    def _on_hidden(self, on):
        """Navire réel = modèle, ou variante aux caractéristiques cachées."""
        self.hidden = on
        self.sim.set_boat(calib.hidden_ship(self.model_params) if on else self.model_params)
        self.scene.set_boat(self.sim.boat)
        self._pending = None
        self._reset_calibration()
        self._needs_redraw = True

    def _reset_calibration(self):
        self.monitor.clear()
        self._calib_pending = self._estimate = None
        self._revealed = False
        self._calib_shown = None

    def _on_calibrate(self):
        """Le modèle du jumeau (panneau, prévision) prend les valeurs identifiées."""
        if self._estimate is None:
            return
        self.model_params = self.panel.set_model_params(self._estimate)
        if self.forecast:
            self.forecast.set_model(self.model_params, self.sim)
            self._pending = None
            self.forecast._next = self.sim.t           # nouvelle prévision tout de suite
        self._revealed = True
        self._calib_shown = None

    def _calibration_tick(self, wave):
        """Enregistre le roulis, lance / récupère l'identification glissante."""
        sim = self.sim
        if abs(sim.state[6]) > 0.3 or abs(sim.hydro.U_target) > 0.3:
            # L'équation identifiée suppose le navire à poste fixe (ω_e constante)
            self.monitor.clear()
            self._show_calibration('moving')
            return
        self.monitor.record(sim.t, sim.state[3], np.arctan(wave[2]))
        if self._calib_pending is None:
            if self.monitor.ready(sim.t):
                self._calib_pending = self._executor.submit(
                    calib.identify_roll, *self.monitor.start(sim.t))
        elif self._calib_pending.done():
            ident = self._calib_pending.result()
            self._calib_pending = None
            self._estimate = calib.calibrated_params(self.model_params, self.sea, ident)
            self._estimate_t, self._calib_shown = sim.t, None
        self._show_calibration(int(sim.t))

    def _show_calibration(self, key):
        """Texte du groupe Calibration, rafraîchi seulement quand `key` change."""
        if key == self._calib_shown:
            return
        self._calib_shown = key
        if key == 'moving':
            self.panel.show_calibration(
                '<span style="color:#c0392b">Identification suspendue : navire en route '
                '(mettre la vitesse à 0).</span>', self._estimate is not None)
            return
        dur = self.monitor.duration()
        if self._estimate is None:
            self.panel.show_calibration(
                f'Roulis enregistré : {dur:.0f} s / {calib.MIN_WINDOW:.0f} s avant la '
                f'première identification…', False)
            return
        m, c, r = self.model_params, self._estimate, self.sim.params
        rows = [('GM_T', 'GM_T', ' m', 2), ('ζ roulis', 'zeta_roll', '', 3),
                ('b_q', 'roll_quad', ' /rad', 2)]
        html = (f'Identifié sur {min(dur, calib.WINDOW):.0f} s (t = {self._estimate_t:.0f} s) :'
                '<table cellspacing="4"><tr><td></td><td><b>estimé</b></td><td>modèle</td>'
                + ('<td>réel</td>' if self.hidden and self._revealed else '') + '</tr>')
        for label, key_, unit, dec in rows:
            html += (f'<tr><td>{label}</td><td><b>{getattr(c, key_):.{dec}f}{unit}</b></td>'
                     f'<td>{getattr(m, key_):.{dec}f}</td>'
                     + (f'<td>{getattr(r, key_):.{dec}f}</td>' if self.hidden and self._revealed
                        else '') + '</tr>')
        html += '</table>'
        if self.hidden and not self._revealed:
            html += '<span style="color:gray">Valeurs réelles révélées après calibration.</span>'
        self.panel.show_calibration(html, True)

    def _preview_boat(self, params):
        """Erreurs de validation, ou résumé des grandeurs dérivées du navire."""
        errors = params.validate()
        if errors:
            return errors, ''
        Tn = HydroModel(params, self.sea).natural_periods()
        return [], (f'Déplacement : <b>{params.mass/1e3:,.0f} t</b>'.replace(',', ' ')
                    + f'<br>Périodes propres : pilonnement <b>{Tn["heave"]:.1f} s</b>, '
                    f'roulis <b>{Tn["roll"]:.1f} s</b>, tangage <b>{Tn["pitch"]:.1f} s</b>'
                    f'<br><span style="color:gray">Résonance quand la période de '
                    f'rencontre Te s\'en approche (Tp actuel : {self.sea.Tp:.1f} s).</span>')

    # ── Boucle ──
    def _tick(self):
        if self.paused:
            if self._needs_redraw:
                Z, Dx, Dy, wave = self.sim.fields()
                self._draw(Z, Dx, Dy, wave, 1)
            return

        target = int((time.perf_counter() - self._t0) / DT)
        if target <= self.frame:
            return                                  # en avance sur le temps réel
        if target - self.frame > MAX_SUBSTEPS:
            # Trop en retard : on décale l'origine plutôt que d'accumuler
            self._t0 += (target - self.frame - MAX_SUBSTEPS) * DT
            target = self.frame + MAX_SUBSTEPS
        n_sub, self.frame = target - self.frame, target

        Z, Dx, Dy, wave = self.sim.step(n_sub)
        if self.forecast:
            self._forecast_tick()
        self._calibration_tick(wave)
        self._draw(Z, Dx, Dy, wave, n_sub)

    def _draw(self, Z, Dx, Dy, wave, n_sub):
        self.scene.on_sea_changed()
        xy = self.sim.boat_xy()
        if self.follow:
            self.scene.translate_camera(xy[0] - self._last_xy[0], xy[1] - self._last_xy[1])
        self._last_xy = xy

        now = time.perf_counter()
        inst = 1.0 / max(now - self._last_render, 1e-3)
        self.fps = inst if self.fps == 0.0 else 0.9 * self.fps + 0.1 * inst
        self._last_render = now

        self.scene.update(Z, Dx, Dy, wave, self.frame, n_sub,
                          fps=self.fps, paused=self.paused)
        self.plotter.render()
        self._needs_redraw = False

    def closeEvent(self, event):
        self.timer.stop()
        self._executor.shutdown(wait=False, cancel_futures=True)
        self.plotter.close()
        super().closeEvent(event)


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
