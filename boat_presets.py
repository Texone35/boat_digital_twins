"""
Préréglages de navires. Ordres de grandeur réalistes ; GM_L estimé par
BM_L ≈ Cwp²·L² / (12·Cb·T) (KG ≈ KB négligé devant BM_L).
"""

from boat import BoatParams

PRESETS = {p.name: p for p in [
    BoatParams(name='Cargo', L=100.0, B=20.0, T=8.0, Cb=0.61, Cwp=0.85,
               GM_T=1.2, GM_L=120.0, freeboard_ratio=0.5, speed=2.5,
               color='#FFD700'),
    BoatParams(name='Porte-conteneurs', L=250.0, B=32.0, T=12.0, Cb=0.65, Cwp=0.85,
               GM_T=1.5, GM_L=480.0, freeboard_ratio=1.0, speed=8.0,
               color='#B03A2E'),
    BoatParams(name='Frégate', L=125.0, B=15.0, T=4.5, Cb=0.50, Cwp=0.75,
               GM_T=1.0, GM_L=320.0, freeboard_ratio=1.1, speed=10.0,
               color='#8E9AA6'),
    BoatParams(name='Chalutier', L=35.0, B=9.0, T=4.0, Cb=0.55, Cwp=0.82,
               GM_T=0.8, GM_L=30.0, freeboard_ratio=0.6, speed=4.0,
               color='#2E6DB4'),
    BoatParams(name='Remorqueur', L=30.0, B=10.0, T=4.5, Cb=0.60, Cwp=0.85,
               GM_T=1.5, GM_L=20.0, freeboard_ratio=0.5, speed=3.0,
               color='#E67E22'),
    BoatParams(name='Yacht à moteur', L=25.0, B=6.0, T=1.8, Cb=0.45, Cwp=0.75,
               GM_T=1.0, GM_L=36.0, freeboard_ratio=1.2, speed=6.0,
               color='#F4F6F7'),
]}

DEFAULT_BOAT = 'Cargo'
