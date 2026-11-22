# Entraîner le PINN sur Google Colab

Le PINN (wavepinn/pinn.py) tourne sur CPU, mais l'entraînement utile (≥ 20 000 itérations par
fenêtre) demande un GPU. Procédure : Colab → *Exécution* → *Modifier le type d'exécution* → **GPU T4**,
puis une cellule par bloc ci-dessous.

```python
# 1. Code (branche du projet) et dépendances — torch, numpy, scipy, matplotlib sont déjà sur Colab
!git clone -b feature/wave-pinn https://github.com/Texone35/boat_digital_twins.git
%cd boat_digital_twins
import torch; print(torch.cuda.get_device_name(0))
```

```python
# 2. Grille d'essais sur le petit domaine (sondes à 250-100 m) : chaque ligne isole un effet.
#    Référence CPU (3000 it.) : pinn.zone ≈ 0.15-0.16 contre lsq.zone ≈ 0.04.
base = "domain=small n_colloc=4096 lr=0.002 ff_mode=dispersion ff_sigma=0.1 w_data=10 phys_ramp=500"
for extra in ["iters=10000",                                  # plus d'itérations seulement
              "iters=30000",
              "iters=30000 causal_eps=1",                     # + causalité (ε annelé ε/100 → ε)
              "iters=30000 causal_eps=10",
              "iters=30000 adapt_every=100",                  # + pondération adaptative
              "iters=30000 causal_eps=1 lbfgs_iters=1000"]:   # + L-BFGS final
    !python scripts/try_pinn.py {base} {extra}
```

```python
# 3. Domaine complet (sondes à 600-300 m, horizon ≈ 60 s) avec le meilleur réglage de l'étape 2
!python scripts/try_pinn.py domain=full n_colloc=8192 lr=0.002 ff_mode=dispersion ff_sigma=0.1 w_data=10 phys_ramp=1000 iters=50000 causal_eps=1 lbfgs_iters=1000
```

```python
# 4. Récupérer les résultats : à recoller dans la conversation / le dépôt
print(open('results/pinn_trials.jsonl').read())
```

Évaluation complète sur les mers de test, une fois un réglage retenu (long : 18 mers × N fenêtres) :

```python
!python scripts/make_dataset.py
!python scripts/run_eval.py --method pinn --t0-step 60 iters=20000 n_colloc=8192 ff_mode=dispersion ff_sigma=0.1 w_data=10 causal_eps=5
from google.colab import files; files.download('results/pinn/metrics.json')
```

À lire dans chaque ligne de `pinn_trials.jsonl` :

- `probe_fit` : erreur aux sondes / Hs (ajustement des données, doit être ≈ bruit 0.03) ;
- `pinn.zone` vs `lsq.zone` : erreur au navire dans la zone de prédictibilité — **le critère** ;
- `pinn.beyond` : hors zone, ≈ 0.25 (prédiction nulle) est sain, > 0.3 = fausses vagues ;
- `final_losses` : résidus kin/dyn/lap (relatifs², viser < 1e-4) et `causal_min` (→ 1 quand
  tout le domaine est résolu ; reste ≈ 0 si causal_eps est trop grand).
