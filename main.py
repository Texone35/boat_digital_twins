"""
Main runner — jumeau numérique : mer JONSWAP + navire Cummins 6-DDL.

Fenêtre Qt : vue 3D PyVista au centre, panneau de contrôle à droite pour
modifier l'état de mer et la navigation en direct, et choisir le navire
(type et caractéristiques), remplacé à chaud.
"""

import sys
import time

import numpy as np
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import QApplication, QDockWidget, QMainWindow
from pyvistaqt import QtInteractor

import jonswap as sea_mod
from boat import HydroModel
from boat_presets import PRESETS, DEFAULT_BOAT
from scene import Scene
from simulation import Simulation
from ui_panel import ControlPanel

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

    def _redraw_after(self, fn, *args):
        fn(*args)
        self._needs_redraw = True

    def _on_sea_changed(self, values):
        values = dict(values)
        if 'theta_mean' in values:
            values['theta_mean'] = np.radians(values['theta_mean'])
        self.sea.set_target(**values)
        if self.paused:
            # En pause, la transition est appliquée d'un coup pour être visible
            self.sea.advance(np.inf, ref=self.sim.boat_xy())
            self._needs_redraw = True

    def _on_boat_applied(self, params):
        self.sim.set_boat(params)
        self.sim.set_speed(params.speed)
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
        self.sim.reset()
        self.scene.recenter_camera()
        self._last_xy = self.sim.boat_xy()
        self._needs_redraw = True

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
        self.plotter.close()
        super().closeEvent(event)


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
