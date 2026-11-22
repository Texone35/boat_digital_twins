"""
Panneau de contrôle Qt du jumeau numérique : état de mer (en direct),
navigation (en direct) et choix / caractéristiques du navire (appliqués à
chaud). Le panneau ne connaît ni la physique ni la scène : il émet des
signaux, et reçoit une fonction d'aperçu pour afficher les grandeurs
dérivées d'un navire avant de l'appliquer.
"""

from dataclasses import replace

from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (QCheckBox, QColorDialog, QComboBox, QDoubleSpinBox,
                            QFormLayout, QGroupBox, QHBoxLayout, QLabel,
                            QPushButton, QScrollArea, QSlider, QVBoxLayout,
                            QWidget)

from jonswap import SEA_PRESETS
from boat_presets import PRESETS

CUSTOM = 'Personnalisé'

# (clé, libellé, min, max, pas, décimales, unité) — curseurs de mer
SEA_FIELDS = [
    ('Hs',         'Hauteur Hs',     0.5, 18.0, 0.1,  1, ' m'),
    ('Tp',         'Période Tp',     4.0, 18.0, 0.1,  1, ' s'),
    ('gamma',      'Pic γ',          1.0,  7.0, 0.1,  1, ''),
    ('theta_mean', 'Direction',      0.0, 360.0, 1.0, 0, ' °'),
    ('spread',     'Étroitesse s',   1.0, 10.0, 0.5,  1, ''),
    ('choppiness', 'Crêtes choppy',  0.0,  1.2, 0.05, 2, ''),
]

# Caractéristiques modifiables du navire (plages de saisie larges : la
# validation détaillée est faite par BoatParams.validate)
BOAT_FIELDS = [
    ('L',               'Longueur L',            1.0, 500.0, 1.0,  1, ' m'),
    ('B',               'Largeur B',             0.5, 100.0, 0.5,  1, ' m'),
    ('T',               "Tirant d'eau T",        0.1,  40.0, 0.1,  1, ' m'),
    ('Cb',              'Coef. de bloc Cb',      0.20,  1.0, 0.01, 2, ''),
    ('Cwp',             'Coef. flottaison Cwp',  0.20,  1.0, 0.01, 2, ''),
    ('GM_T',            'GM transversal',       -2.0,  10.0, 0.05, 2, ' m'),
    ('GM_L',            'GM longitudinal',      -10.0, 2000.0, 1.0, 0, ' m'),
    ('kxx',             'Giration roulis /B',    0.10,  0.6, 0.01, 2, ''),
    ('kyy',             'Giration tangage /L',   0.10,  0.5, 0.01, 2, ''),
    ('freeboard_ratio', 'Franc-bord / T',        0.0,   3.0, 0.05, 2, ''),
    ('zeta_roll',       'Amort. roulis ζ',       0.005, 0.5, 0.005, 3, ''),
    ('roll_quad',       'Amort. roulis quadr.',  0.0,   5.0, 0.05, 2, ' /rad'),
]


def _spin(vmin, vmax, step, decimals, unit, value):
    s = QDoubleSpinBox()
    s.setRange(vmin, vmax)
    s.setSingleStep(step)
    s.setDecimals(decimals)
    s.setSuffix(unit)
    s.setValue(value)
    s.setKeyboardTracking(False)      # valeur émise à la validation, pas à chaque touche
    return s


class SliderRow(QWidget):
    """Curseur couplé à un champ numérique ; émet valueChanged(float)."""
    valueChanged = Signal(float)

    def __init__(self, vmin, vmax, step, decimals, unit, value):
        super().__init__()
        self._vmin, self._step = vmin, step
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, round((vmax - vmin) / step))
        self.spin = _spin(vmin, vmax, step, decimals, unit, value)
        self.spin.setFixedWidth(110)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.spin)
        self._sync_slider(value)
        self.slider.valueChanged.connect(self._from_slider)
        self.spin.valueChanged.connect(self._from_spin)

    def _sync_slider(self, v):
        self.slider.blockSignals(True)
        self.slider.setValue(round((v - self._vmin) / self._step))
        self.slider.blockSignals(False)

    def _from_slider(self, i):
        self.spin.blockSignals(True)
        self.spin.setValue(self._vmin + i * self._step)
        self.spin.blockSignals(False)
        self.valueChanged.emit(self.spin.value())

    def _from_spin(self, v):
        self._sync_slider(v)
        self.valueChanged.emit(v)

    def value(self):
        return self.spin.value()

    def setValue(self, v):
        """Changement programmatique, sans émettre de signal."""
        self.spin.blockSignals(True)
        self.spin.setValue(v)
        self.spin.blockSignals(False)
        self._sync_slider(v)


class ControlPanel(QWidget):
    seaChanged = Signal(dict)          # paramètres de mer modifiés (theta_mean en degrés)
    speedChanged = Signal(float)       # m/s
    headingChanged = Signal(float)     # degrés
    pauseToggled = Signal(bool)
    resetRequested = Signal()
    recenterRequested = Signal()
    followToggled = Signal(bool)
    boatApplied = Signal(object)       # BoatParams
    forecastToggled = Signal(bool)     # mode Prévision
    ghostLeadChanged = Signal(float)   # avance de la coque fantôme (s)
    hiddenToggled = Signal(bool)       # navire réel aux caractéristiques cachées
    calibrateRequested = Signal()      # appliquer l'estimation au modèle

    def __init__(self, sea_params, boat_params, preview_fn, parent=None):
        """
        sea_params : paramètres de mer initiaux (theta_mean en rad)
        boat_params : BoatParams initial
        preview_fn(params) -> (erreurs: list[str], résumé: str)
        """
        super().__init__(parent)
        self._preview_fn = preview_fn
        self._applied = boat_params
        self._color = boat_params.color

        content = QWidget()
        lay = QVBoxLayout(content)
        lay.addWidget(self._build_sea_group(sea_params))
        lay.addWidget(self._build_nav_group(boat_params))
        lay.addWidget(self._build_forecast_group())
        lay.addWidget(self._build_boat_group(boat_params))
        lay.addWidget(self._build_calibration_group())
        lay.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)
        self.setMinimumWidth(380)
        self._update_preview()

    # ── Mer ──
    def _build_sea_group(self, sea_params):
        box = self.sea_box = QGroupBox('Mer (effet immédiat, transition douce)')
        form = QFormLayout(box)
        self.sea_preset = QComboBox()
        self.sea_preset.addItems([CUSTOM, *SEA_PRESETS])
        self.sea_preset.currentTextChanged.connect(self._on_sea_preset)
        form.addRow('État de mer', self.sea_preset)

        self.sea_rows = {}
        for key, label, vmin, vmax, step, dec, unit in SEA_FIELDS:
            v = sea_params[key]
            if key == 'theta_mean':
                v = (v * 180.0 / 3.141592653589793) % 360.0
            row = SliderRow(vmin, vmax, step, dec, unit, v)
            row.valueChanged.connect(lambda val, k=key: self._on_sea_value(k, val))
            self.sea_rows[key] = row
            form.addRow(label, row)
        self._match_sea_preset()

        hint = QLabel('Direction : sens de propagation de la houle (0° = vers +X).')
        hint.setWordWrap(True)
        hint.setStyleSheet('color: gray; font-size: 11px;')
        form.addRow(hint)
        return box

    def _on_sea_preset(self, name):
        if name == CUSTOM:
            return
        values = SEA_PRESETS[name]
        for k, v in values.items():
            self.sea_rows[k].setValue(v)
        self.seaChanged.emit(dict(values))

    def _on_sea_value(self, key, value):
        self._match_sea_preset()
        self.seaChanged.emit({key: value})

    def _match_sea_preset(self):
        """Affiche le préréglage correspondant aux curseurs, sinon « Personnalisé »."""
        name = next((n for n, vals in SEA_PRESETS.items()
                     if all(abs(self.sea_rows[k].value() - v) < 1e-6
                            for k, v in vals.items())), CUSTOM)
        self.sea_preset.blockSignals(True)
        self.sea_preset.setCurrentText(name)
        self.sea_preset.blockSignals(False)

    # ── Navigation ──
    def _build_nav_group(self, boat_params):
        box = QGroupBox('Navigation (effet immédiat)')
        form = QFormLayout(box)
        self.speed_row = SliderRow(0.0, 15.0, 0.1, 1, ' m/s', boat_params.speed)
        self.speed_row.valueChanged.connect(self.speedChanged)
        form.addRow('Vitesse', self.speed_row)
        self.heading_row = SliderRow(0.0, 360.0, 1.0, 0, ' °', 0.0)
        self.heading_row.valueChanged.connect(self.headingChanged)
        form.addRow('Cap', self.heading_row)

        self.follow = QCheckBox('Caméra suit le bateau')
        self.follow.setChecked(True)
        self.follow.toggled.connect(self.followToggled)
        form.addRow(self.follow)

        btns = QHBoxLayout()
        self.pause_btn = QPushButton('Pause')
        self.pause_btn.setCheckable(True)
        self.pause_btn.toggled.connect(self._on_pause)
        recenter = QPushButton('Recentrer vue')
        recenter.clicked.connect(self.recenterRequested)
        reset = QPushButton('Réinitialiser')
        reset.setToolTip('Bateau replacé au centre, au repos')
        reset.clicked.connect(self.resetRequested)
        for b in (self.pause_btn, recenter, reset):
            btns.addWidget(b)
        form.addRow(btns)
        return box

    # ── Prévision ──
    def _build_forecast_group(self):
        box = QGroupBox('Prévision de houle')
        form = QFormLayout(box)
        self.forecast_chk = QCheckBox('Mode prévision')
        self.forecast_chk.setToolTip(
            'Houle unidirectionnelle figée (Hs, Tp, γ courants), 4 sondes 300-450 m en '
            'amont, navire en station. Prévision LSQ toutes les 5 s sur 60 s.')
        self.forecast_chk.toggled.connect(self.forecastToggled)
        form.addRow(self.forecast_chk)
        self.ghost_row = SliderRow(0.0, 60.0, 1.0, 0, ' s', 20.0)
        self.ghost_row.valueChanged.connect(self.ghostLeadChanged)
        form.addRow('Coque fantôme à t +', self.ghost_row)
        hint = QLabel('Régler la mer avant d\'activer (mer modérée : le modèle est linéaire) ; '
                      'elle est ensuite figée. Cap oblique (ex. 60°) pour voir du roulis.')
        hint.setWordWrap(True)
        hint.setStyleSheet('color: gray; font-size: 11px;')
        form.addRow(hint)
        return box

    # ── Calibration ──
    def _build_calibration_group(self):
        box = QGroupBox('Calibration sur le roulis mesuré')
        form = QFormLayout(box)
        self.hidden_chk = QCheckBox('Navire réel aux caractéristiques cachées')
        self.hidden_chk.setToolTip(
            'Le navire simulé (« réel ») reçoit un GM_T plus faible et des amortissements de '
            'roulis tirés au hasard ; le panneau garde le modèle, que la calibration recale.')
        self.hidden_chk.toggled.connect(self.hiddenToggled)
        form.addRow(self.hidden_chk)
        self.calib_status = QLabel('Enregistrement du roulis…')
        self.calib_status.setWordWrap(True)
        self.calib_status.setTextFormat(Qt.RichText)
        form.addRow(self.calib_status)
        self.calib_btn = QPushButton('Calibrer le modèle')
        self.calib_btn.setEnabled(False)
        self.calib_btn.clicked.connect(self.calibrateRequested)
        form.addRow(self.calib_btn)
        hint = QLabel('Identification (erreur de sortie) toutes les 20 s sur les 300 dernières '
                      'secondes de roulis, navire à poste fixe (vitesse nulle) ; houle de travers '
                      'conseillée.')
        hint.setWordWrap(True)
        hint.setStyleSheet('color: gray; font-size: 11px;')
        form.addRow(hint)
        return box

    def show_calibration(self, html, can_apply):
        self.calib_status.setText(html)
        self.calib_btn.setEnabled(can_apply)

    def set_model_params(self, params):
        """Le formulaire et le modèle appliqué deviennent `params` (sans signal).
        Retourne les paramètres tels qu'arrondis par le formulaire."""
        self._fill_boat(params)
        self._applied = self.form_params()
        self._update_preview()
        return self._applied

    def set_sea_locked(self, locked):
        self.sea_box.setEnabled(not locked)

    def set_sea_values(self, sea_params):
        """Curseurs de mer alignés sur la mer (sans émettre de signal)."""
        for key, row in self.sea_rows.items():
            v = sea_params[key]
            row.setValue((v * 180.0 / 3.141592653589793) % 360.0 if key == 'theta_mean' else v)
        self._match_sea_preset()

    def _on_pause(self, paused):
        self.pause_btn.setText('Reprendre' if paused else 'Pause')
        self.pauseToggled.emit(paused)

    # ── Navire ──
    def _build_boat_group(self, boat_params):
        box = QGroupBox('Navire (appliqué à chaud)')
        form = QFormLayout(box)
        self.boat_type = QComboBox()
        self.boat_type.addItems(list(PRESETS))
        self.boat_type.setCurrentText(boat_params.name)
        self.boat_type.currentTextChanged.connect(self._on_boat_type)
        form.addRow('Type', self.boat_type)

        self.boat_spins = {}
        for key, label, vmin, vmax, step, dec, unit in BOAT_FIELDS:
            s = _spin(vmin, vmax, step, dec, unit, getattr(boat_params, key))
            s.valueChanged.connect(self._update_preview)
            self.boat_spins[key] = s
            form.addRow(label, s)

        self.color_btn = QPushButton()
        self.color_btn.clicked.connect(self._pick_color)
        self._show_color()
        form.addRow('Couleur coque', self.color_btn)

        self.preview = QLabel()
        self.preview.setWordWrap(True)
        self.preview.setTextFormat(Qt.RichText)
        form.addRow(self.preview)

        btns = QHBoxLayout()
        self.apply_btn = QPushButton('Appliquer')
        self.apply_btn.setDefault(True)
        self.apply_btn.clicked.connect(self._apply)
        restore = QPushButton('Rétablir le préréglage')
        restore.clicked.connect(lambda: self._fill_boat(PRESETS[self.boat_type.currentText()]))
        btns.addWidget(self.apply_btn)
        btns.addWidget(restore)
        form.addRow(btns)
        return box

    def _on_boat_type(self, name):
        self._fill_boat(PRESETS[name])

    def _fill_boat(self, params):
        for key, s in self.boat_spins.items():
            s.blockSignals(True)
            s.setValue(getattr(params, key))
            s.blockSignals(False)
        self._color = params.color
        self._show_color()
        self._update_preview()

    def _pick_color(self):
        c = QColorDialog.getColor(QColor(self._color), self, 'Couleur de la coque')
        if c.isValid():
            self._color = c.name()
            self._show_color()
            self._update_preview()

    def _show_color(self):
        self.color_btn.setText(self._color)
        self.color_btn.setStyleSheet(f'background-color: {self._color};')

    def form_params(self):
        """BoatParams décrit par le formulaire."""
        base = PRESETS[self.boat_type.currentText()]
        values = {k: s.value() for k, s in self.boat_spins.items()}
        p = replace(base, color=self._color, **values)
        if p != base:
            p = replace(p, name=f'{base.name} (modifié)')
        return p

    def _update_preview(self, *_):
        p = self.form_params()
        errors, summary = self._preview_fn(p)
        if errors:
            self.preview.setText('<span style="color:#c0392b">'
                                 + '<br>'.join(errors) + '</span>')
        else:
            self.preview.setText(summary)
        self.apply_btn.setEnabled(not errors)
        pending = p.as_dict() | {'speed': 0} != self._applied.as_dict() | {'speed': 0}
        self.apply_btn.setText('Appliquer *' if pending else 'Appliquer')

    def _apply(self):
        p = self.form_params()
        if p.validate():
            return
        # Nouveau type : vitesse de croisière du préréglage ; même type
        # modifié : vitesse courante conservée
        base = self.boat_type.currentText()
        if not self._applied.name.startswith(base):
            self.speed_row.setValue(p.speed)
        else:
            p = replace(p, speed=self.speed_row.value())
        self._applied = p
        self.boatApplied.emit(p)
        self._update_preview()

    # ── Synchronisation depuis la simulation ──
    def set_heading(self, deg):
        self.heading_row.setValue(deg % 360.0)
