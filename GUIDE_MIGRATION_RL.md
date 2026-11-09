# Guide Complet de Migration RL & Modernisation — Race Simulation F1

Ce document réunit l'analyse complète du projet **Race Simulation RL**, les recommandations d'algorithmes issues des cours de Reinforcement Learning (UMONS - Pr. Xavier Siebert), l'évaluation de la pile logicielle Deep Learning obsolète, et le guide pas-à-pas de migration vers un écosystème moderne et pérenne basé sur **PyTorch**, **Gymnasium** et **Stable-Baselines3**.

---

## Sommaire

1. [Diagnostic de la pile Deep Learning actuelle](#1-diagnostic-de-la-pile-deep-learning-actuelle)
2. [Nouvelle Stack Cible (Moderne, Stable & Pérenne)](#2-nouvelle-stack-cible-moderne-stable--pérenne)
3. [Idées d'implémentation RL par ordre de difficulté](#3-idées-dimplémentation-rl-par-ordre-de-difficulté)
4. [Métriques d'évaluation recommandées](#4-métriques-dévaluation-recommandées)
5. [Guide des modifications de code](#5-guide-des-modifications-de-code)
   - [5.1 `requirements.txt`](#51-requirementstxt)
   - [5.2 Correctifs Numpy (`preprocessor.py` & `rl_environment_single_agent.py`)](#52-correctifs-numpy)
   - [5.3 Correctif Matplotlib (`rl_evaluate_policy.py`)](#53-correctif-matplotlib)
   - [5.4 Nouvel environnement Gymnasium (`rl_environment_gymnasium.py`)](#54-nouvel-environnement-gymnasium)
   - [5.5 Nouveau script d'entraînement SB3 (`main_train_sb3.py`)](#55-nouveau-script-dentraînement-sb3)
6. [Option : 100% Zéro TensorFlow via ONNX Runtime](#6-option--100-zéro-tensorflow-via-onnx-runtime)
7. [Commandes d'installation et de lancement](#7-commandes-dinstallation-et-de-lancement)

---

## 1. Diagnostic de la pile Deep Learning actuelle

### Ce qui est actuellement en place dans le projet

| Composant | Technologie | Détails |
|---|---|---|
| **Algorithme RL** | **DQN** (Deep Q-Network) via `tf-agents` | Single-agent & multi-agent |
| **Réseau** | Fully Connected `(64, 64)` | Input: observation encodée, Output: Q-values discrètes |
| **Exploration** | **Boltzmann** polynomiale | Décroissance de 1.0 à 0.01 sur 250k steps |
| **Replay buffer** | `TFUniformReplayBuffer` (200k max) | Batch size = 64 |
| **Environnement** | `tf_py_environment.PyEnvironment` | Simulation F1 séquentielle lap par lap |
| **Modèles de prédiction** | Réseaux de neurones TFLite (`.tflite`) | Utilisés dans `vse_supervised.py` et `vse_reinforcement.py` |

### Le problème majeur : `tensorflow==2.3.0` (Novembre 2020)

`tensorflow==2.3.0` a atteint sa fin de vie (End-Of-Life) en **2022**.

| Risque | Niveau | Impact concret |
|---|---|---|
| **Incompatibilité Python** | 🔴 Bloquant | TF 2.3 ne fonctionne que sur **Python ≤ 3.8** (lui-même EOL). Crashe sous Python 3.9, 3.10, 3.11+. |
| **Incompatibilité GPU** | 🔴 Bloquant | TF 2.3 requiert CUDA 10.1 / cuDNN 7. Impossible d'utiliser les GPU récents (Nvidia RTX 30xx, 40xx, etc.). |
| **`tf-agents` obsolète** | 🔴 Bloquant | API dépréciée, quasi abandonnée par Google au profit de JAX/RLlib/Torch. |
| **Failles de sécurité** | 🟠 Élevé | Multiples CVE non corrigées sur TF 2.3. |
| **Numpy dépréciations** | 🔴 Bloquant | Crash immédiat avec Numpy 1.24+ (`AttributeError: module 'numpy' has no attribute 'bool' / 'float'`). |

---

## 2. Nouvelle Stack Cible (Moderne, Stable & Pérenne)

Le choix recommandé est une transition complète vers **PyTorch + Gymnasium + Stable-Baselines3** :

| Outil / Package | Ancienne version | Nouvelle version | Justification |
|---|---|---|---|
| **Python** | 3.8 (obsolète) | **3.11** | Supporté jusqu'en 2027, mature, rapide |
| **numpy** | 1.23.5 | **1.26.4** | Dernière version 1.x ultra-stable (évite les breaking changes de Numpy 2.x) |
| **scipy** | 1.7.3 | **1.13.1** | Compatible numpy 1.26 et cvxpy |
| **matplotlib** | 3.5.3 | **3.9.2** | Graphiques d'évaluation et télémétrie |
| **cvxpy** | 1.2.1 | **1.5.3** | Moteurs d'optimisation QP/OSQP |
| **pandas** | non fixé | **2.2.3** | Manipulation de données de télémétrie |
| **torch** | non présent | **2.4.1** | Le standard mondial de recherche et industrie RL, support CUDA complet |
| **stable-baselines3** | absent | **2.3.2** | Bibliothèque de référence RL (DQN, PPO, SAC, TD3) |
| **gymnasium** | absent | **0.29.1** | Standard maintenu par la Farama Foundation (remplace Gym) |
| **tensorflow-cpu** | 2.3.0 | **2.17.0** *(option A)* | Uniquement pour charger les modèles `.tflite` existants |
| **onnxruntime** | absent | **1.19.2** *(option B)* | Permet de **supprimer 100% de TensorFlow** |

---

## 3. Idées d'implémentation RL par ordre de difficulté

### Niveau 1 — Facile (Améliorations directes de DQN)

1. **Dueling DQN** :
   - Décompose la valeur Q en fonction d'état $V(s)$ et avantage d'action $A(s, a)$ : $Q(s, a) = V(s) + (A(s, a) - \frac{1}{|A|}\sum A(s, a'))$.
   - **Pourquoi en F1 ?** Dans 90% des tours, la bonne action est de "rester en piste" ; séparer l'état global et l'avantage de pitter accélère fortement l'apprentissage.
   - En SB3 : inclus nativement ou via custom policy.

2. **Double DQN (DDQN)** :
   - Évite la surestimation des Q-valeurs en sélectionnant l'action avec le réseau en ligne et en l'évaluant avec le réseau cible.
   - En SB3 : activé par défaut dans `stable_baselines3.DQN`.

3. **Reward Shaping Avancé** :
   - Pénalité progressive sur l'usure excessive des pneus au-delà du seuil critique (cliff).
   - Bonus d'opportunité en cas de Safety Car / Virtual Safety Car (pit sous FCY coûte ~10-15s de moins).
   - Bonus de défense contre l'undercut.

---

### Niveau 2 — Intermédiaire (Algorithmes Modernes)

1. **Prioritized Experience Replay (PER)** :
   - Échantillonne en priorité les transitions ayant la plus grande erreur TD $|\delta_t|$.
   - Les tours cruciaux (Safety Car soudain, fin de vie du pneu) sont revisités beaucoup plus souvent que les tours calmes.

2. **PPO (Proximal Policy Optimization)** :
   - Algorithme On-Policy Actor-Critic avec clipping de substitution.
   - Beaucoup plus stable que DQN face aux variations stochastiques (accidents, météo, Safety Car).
   - Facile à tester avec `from stable_baselines3 import PPO`.

3. **N-step TD Learning** :
   - Calcule le retour sur $N$ étapes plutôt qu'une seule ($R_t + \gamma R_{t+1} + \dots + \gamma^N \max Q$).
   - Permet de propager rapidement l'effet d'un pit stop sur plusieurs tours.

---

### Niveau 3 — Avancé (Multi-Agent & Environnements Partiels)

1. **MARL (Multi-Agent Reinforcement Learning)** :
   - Entraîner simultanément plusieurs voitures ou une écurie de 2 voitures (tactique d'équipe : bloquer un rival, faire l'undercut avec une voiture pour forcer l'adversaire à réagir).
   - Approches : MAPPO, QMIX, ou Independent PPO (IPPO).

2. **Modélisation POMDP & Recurrent Policies (LSTM / GRU)** :
   - En F1 réelle, l'état complet des rivaux (dégradation réelle des pneus adverses, stratégie cachée) n'est pas directement observable.
   - Utilisation de `RecurrentPPO` (SB3 Contrib) pour exploiter l'historique temporel des temps au tour.

---

### Niveau 4 — Recherche & État de l'art

1. **Decision Transformer (Offline RL)** :
   - Utiliser les historiques de courses réelles F1 pour apprendre des politiques de stratégie sous forme de séquence (Return-to-go, State, Action) sans interaction en direct.

2. **MCTS + RL (Monte Carlo Tree Search façon AlphaZero)** :
   - Combiner un réseau de valeur avec la simulation prédictive pour planifier les scénarios d'arrêts aux stands à horizon 5–15 tours.

---

## 4. Métriques d'évaluation recommandées

Pour évaluer rigoureusement votre agent RL face aux adversaires VSE supervisés ou d'autres agents :

| Métrique | Unité | Interprétation |
|---|---|---|
| **Final Race Position** | 1 à 20 | Position moyenne à l'arrivée (sur 10 000 simulations) |
| **Podium Rate / Win Rate** | % | Pourcentage d'arrivées dans le Top 3 ou en 1ère position |
| **Total Race Time** | Secondes | Temps total de course simulé (à comparer à la stratégie optimale théorique) |
| **Cumulative Discounted Return** | Score | Récompense totale accumulée par épisode |
| **Safety Car Exploit Rate** | % | Taux de pit stop déclenché immédiatement sous SC / VSC |
| **Pit Stop Count Distribution** | Nombre | Distribution du nombre d'arrêts (1 stop, 2 stops, 3 stops) |
| **Rule Compliance** | % | 100% requis : utilisation d'au moins 2 types de gommes slicks différents |

---

## 5. Guide des modifications de code

### 5.1 `requirements.txt`

Remplacer le contenu du fichier par :

```txt
# Calcul scientifique
numpy==1.26.4
scipy==1.13.1
matplotlib==3.9.2
pandas==2.2.3
scikit-learn==1.5.2
tqdm==4.66.5

# Optimisation (racesim_basic)
cvxpy==1.5.3

# Deep Learning — Inférence TFLite existante
tensorflow-cpu==2.17.0

# Deep Learning — RL Moderne (PyTorch & Gymnasium)
torch==2.4.1
stable-baselines3==2.3.2
gymnasium==0.29.1
```

---

### 5.2 Correctifs Numpy

Numpy 1.24+ a supprimé les alias obsolètes `np.bool` et `np.float`. Deux corrections simples sont requises :

#### Dans `machine_learning/src/preprocessor.py` (Ligne 232) :
```python
# AVANT :
idx_true = np.nonzero(X_conv_[:, conv_idx:conv_idx + self.col_info[raw_idx]['no_classes']].astype(np.bool))[1]

# APRÈS :
idx_true = np.nonzero(X_conv_[:, conv_idx:conv_idx + self.col_info[raw_idx]['no_classes']].astype(bool))[1]
```

#### Dans `machine_learning_rl_training/src/rl_environment_single_agent.py` (Ligne 557) :
```python
# AVANT :
def __calculate_reward_position(self, delta_position: np.int32) -> np.float:

# APRÈS :
def __calculate_reward_position(self, delta_position: np.int32) -> float:
```

---

### 5.3 Correctif Matplotlib

#### Dans `machine_learning_rl_training/src/rl_evaluate_policy.py` (vers ligne 21) :
Ajouter l'import manquant qui provoquait un crash à l'évaluation :
```python
import numpy as np
import matplotlib.pyplot as plt   # <-- AJOUTER CETTE LIGNE
```

---

### 5.4 Nouvel environnement Gymnasium (`machine_learning_rl_training/src/rl_environment_gymnasium.py`)

Créez le fichier `rl_environment_gymnasium.py` qui encapsule la logique de course avec l'API standard `gymnasium.Env` :

```python
"""
Wrapper Gymnasium standard pour l'environnement de course F1.
Compatible avec Stable-Baselines3 (DQN, PPO, A2C, etc.).
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import racesim
import machine_learning
import math
import random


class RaceSimulationGymnasium(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self,
                 race_pars_file: str,
                 mcs_pars_file: str,
                 vse_type: str = "supervised",
                 use_prob_infl: bool = True,
                 create_rand_events: bool = True,
                 controlled_driver: str = None):
        super().__init__()

        self.race_pars_file = race_pars_file
        self.mcs_pars_file = mcs_pars_file
        self.vse_type = vse_type
        self.use_prob_infl = use_prob_infl
        self.create_rand_events = create_rand_events
        self.controlled_driver = controlled_driver

        self.pars_in = None
        self.driver_initials = None
        self.race = self._create_race()

        self.available_compounds = self.race.vse.vse_pars["param_dry_compounds"]
        self.all_driver_initials = sorted([
            self.race.drivers_list[i].initials for i in range(self.race.no_drivers)
        ])
        self.idx_driver = next(
            (i for i, d in enumerate(self.race.drivers_list)
             if d.initials == self.driver_initials), None
        )
        self.average_laptimes_presim = self.race.execute_presim_average_laptimes()
        self.idx_driver_behind_prev_lap = None

        # Preprocessor identique
        self.cat_preprocessor = machine_learning.src.preprocessor.Preprocessor(
            feature_types=["cat"] * 7, bins_buck=[]
        )
        self.cat_preprocessor.fit_cat_dict(X_cat_str=self.available_compounds, featurename="cur_compound")
        self.cat_preprocessor.fit_cat_dict(X_cat_str=self.all_driver_initials, featurename="driver_initials")

        no_rows = max(len(self.available_compounds), 2, 5, len(self.all_driver_initials))
        X_cat_fit = np.zeros((no_rows, 7))
        X_cat_fit[:, 0] = 1.0
        X_cat_fit[:, 6] = 1.0
        X_cat_fit[:len(self.available_compounds), 0] = self.cat_preprocessor.transform_cat_dict(
            X_cat_str=self.available_compounds, featurename="cur_compound")
        X_cat_fit[0, 1] = True
        X_cat_fit[:5, 2] = list(range(5))
        X_cat_fit[0, 3] = True
        X_cat_fit[0, 4] = True
        X_cat_fit[0, 5] = True
        X_cat_fit[:len(self.all_driver_initials), 6] = self.cat_preprocessor.transform_cat_dict(
            X_cat_str=self.all_driver_initials, featurename="driver_initials")
        self.cat_preprocessor.fit(X=X_cat_fit)

        # Espaces d'observation et d'action
        obs_size = 4 + self.cat_preprocessor.no_transf_cols
        n_actions = len(self.available_compounds) + 1

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(n_actions)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.race = self._create_race()
        self.idx_driver = next(
            (i for i, d in enumerate(self.race.drivers_list)
             if d.initials == self.driver_initials), None
        )
        self.idx_driver_behind_prev_lap = None
        self.race._RaceReinftrain__simulate_lap_start()
        obs = self._calculate_observation()
        return obs, {}

    def step(self, action):
        position_before = self.race.positions[self.race.cur_lap, self.idx_driver]
        laptime_before  = self.race.laptimes[self.race.cur_lap, self.idx_driver]

        if action > 0:
            self.race.drivers_list[self.idx_driver].strategy_info.append(
                [self.race.cur_lap, self.available_compounds[action - 1], 0, 0.0]
            )

        self.race._RaceReinftrain__simulate_lap()

        position_after  = self.race.positions[self.race.cur_lap - 1, self.idx_driver]
        delta_position  = position_before - position_after
        laptime_after   = self.race.laptimes[self.race.cur_lap - 1, self.idx_driver]
        t_pitdrive_inlap = laptime_after - laptime_before

        obs    = self._calculate_observation()
        reward = 0.1 * (self._calculate_reward_laptime(t_pitdrive_inlap)
                        + self._calculate_reward_position(delta_position))

        terminated = self.race.cur_lap >= self.race.race_pars["tot_no_laps"]

        if terminated:
            self.race._RaceReinftrain__simulate_lap_end()
            reward += self._calculate_reward_final_position()
            used_2compounds = len({
                x[1] for x in self.race.drivers_list[self.idx_driver].strategy_info
            }) > 1
            if not used_2compounds:
                reward -= 10.0

        return obs, float(reward), terminated, False, {}

    def _create_race(self):
        sim_opts = {"use_prob_infl": self.use_prob_infl,
                    "create_rand_events": self.create_rand_events,
                    "use_vse": True, "no_sim_runs": 1, "no_workers": 1,
                    "use_print": False, "use_print_result": False, "use_plot": False}

        self.pars_in, vse_paths = racesim.src.import_pars.import_pars(
            use_print=False, use_vse=True,
            race_pars_file=self.race_pars_file, mcs_pars_file=self.mcs_pars_file)

        for driver in self.pars_in['vse_pars']['vse_type']:
            self.pars_in['vse_pars']['vse_type'][driver] = self.vse_type

        participants = self.pars_in["race_pars"]["participants"]
        if self.controlled_driver is None:
            self.driver_initials = random.choice(participants)
        else:
            if self.controlled_driver not in participants:
                raise ValueError(f"Pilote {self.controlled_driver} non trouvé.")
            self.driver_initials = self.controlled_driver

        self.pars_in['vse_pars']['vse_type'][self.driver_initials] = 'reinforcement_training'
        self.pars_in['event_pars']['fcy_data']['phases'] = []
        self.pars_in['event_pars']['retire_data']['retirements'] = []

        racesim.src.check_pars.check_pars(sim_opts=sim_opts, pars_in=self.pars_in)

        race = racesim.src.race_reinftrain.RaceReinftrain(
            race_pars=self.pars_in["race_pars"], driver_pars=self.pars_in["driver_pars"],
            car_pars=self.pars_in["car_pars"], tireset_pars=self.pars_in["tireset_pars"],
            track_pars=self.pars_in["track_pars"], vse_pars=self.pars_in["vse_pars"],
            vse_paths=vse_paths, use_prob_infl=sim_opts["use_prob_infl"],
            create_rand_events=sim_opts["create_rand_events"],
            monte_carlo_pars=self.pars_in["monte_carlo_pars"],
            event_pars=self.pars_in["event_pars"])

        idx_cur = next((i for i, d in enumerate(race.drivers_list)
                        if d.initials == self.driver_initials), None)
        race.retire_data['retirements'][idx_cur] = None
        return race

    def _calculate_observation(self) -> np.ndarray:
        cur_lap      = self.race.cur_lap
        position     = self.race.positions[cur_lap, self.idx_driver]
        tire_age     = self.race.drivers_list[self.idx_driver].car.tireset.age_degr
        cur_compound = self.race.drivers_list[self.idx_driver].car.tireset.compound
        team         = self.race.drivers_list[self.idx_driver].car.manufacturer
        strategy_info = self.race.drivers_list[self.idx_driver].strategy_info

        used_2compounds = len({x[1] for x in strategy_info}) > 1

        idx_act_phase_rel = self.race.fcy_handling["idxs_act_phase"][self.idx_driver]
        fcy_type = self.race.fcy_data["phases"][idx_act_phase_rel][2] if idx_act_phase_rel is not None else None
        fcy_stat_curlap = 0

        if fcy_type is not None:
            fcy_start_end_progs = self.race.fcy_handling["start_end_prog"][self.idx_driver]
            if cur_lap - fcy_start_end_progs[0] >= 0.05 and math.isclose(fcy_start_end_progs[1], cur_lap):
                if fcy_type == 'VSC':
                    fcy_stat_curlap = 1 if cur_lap - fcy_start_end_progs[0] <= 1.0 else 2
                else:
                    fcy_stat_curlap = 3 if cur_lap - fcy_start_end_progs[0] <= 1.0 else 4

        t_lost = (self.pars_in['track_pars']['t_pit_tirechange_min']
                  + self.pars_in['car_pars'][team]['t_pit_tirechange_add'])
        if fcy_stat_curlap == 0:
            t_lost += self.pars_in['track_pars']['t_pitdrive_inlap'] + self.pars_in['track_pars']['t_pitdrive_outlap']
        elif fcy_stat_curlap in [1, 2, 3]:
            t_lost += self.pars_in['track_pars']['t_pitdrive_inlap_fcy'] + self.pars_in['track_pars']['t_pitdrive_outlap_fcy']
        else:
            t_lost += self.pars_in['track_pars']['t_pitdrive_inlap_sc'] + self.pars_in['track_pars']['t_pitdrive_outlap_sc']

        racetimes_tmp = self.race.racetimes[self.race.cur_lap - 1] + self.race.laptimes[self.race.cur_lap]
        est_pos_loss = len([x for x in racetimes_tmp
                            if racetimes_tmp[self.idx_driver] < x <= (racetimes_tmp[self.idx_driver] + t_lost)])

        bool_driving = self.race.bool_driving[self.race.cur_lap]
        positions    = self.race.positions[self.race.cur_lap]

        if positions[self.idx_driver] != 1:
            idx_ahead = [i for i in range(len(positions)) if positions[i] == positions[self.idx_driver] - 1][0]
            interval  = racetimes_tmp[self.idx_driver] - racetimes_tmp[idx_ahead]
        else:
            interval = np.inf

        if positions[self.idx_driver] == self.race.no_drivers:
            ahead = np.inf
        else:
            idx_behind = [i for i in range(len(positions)) if positions[i] == positions[self.idx_driver] + 1][0]
            ahead = np.inf if not bool_driving[idx_behind] else racetimes_tmp[idx_behind] - racetimes_tmp[self.idx_driver]

        close_behind = interval <= 1.5
        close_ahead  = ahead    <= 1.5

        defendable_undercut = False
        if self.idx_driver_behind_prev_lap in self.race.pit_driver_idxs:
            gap = racetimes_tmp[self.idx_driver_behind_prev_lap] - racetimes_tmp[self.idx_driver] - t_lost
            if 0.0 < gap <= 10.0:
                defendable_undercut = True

        if positions[self.idx_driver] == self.race.no_drivers:
            self.idx_driver_behind_prev_lap = None
        else:
            self.idx_driver_behind_prev_lap = next(
                (i for i, pos in enumerate(positions) if pos == positions[self.idx_driver] + 1), None)

        X_cat = np.zeros(7)
        X_cat[0] = self.cat_preprocessor.transform_cat_dict([cur_compound], "cur_compound")[0]
        X_cat[1] = used_2compounds
        X_cat[2] = fcy_stat_curlap
        X_cat[3] = close_behind
        X_cat[4] = close_ahead
        X_cat[5] = defendable_undercut
        X_cat[6] = self.cat_preprocessor.transform_cat_dict([self.driver_initials], "driver_initials")[0]

        obs = np.zeros(4 + self.cat_preprocessor.no_transf_cols, dtype=np.float32)
        obs[0] = cur_lap / self.race.race_pars["tot_no_laps"]
        obs[1] = (position - 1) / (self.race.no_drivers - 1)
        obs[2] = est_pos_loss / (self.race.no_drivers - 1)
        obs[3] = tire_age / self.race.race_pars["tot_no_laps"]
        obs[4:] = self.cat_preprocessor.transform(X=X_cat, dtype_out=np.float32)
        return obs

    def _calculate_reward_laptime(self, t_pitdrive_inlap: float) -> float:
        cur_lap = self.race.cur_lap
        cur_laptime = float(self.race.laptimes[cur_lap, self.idx_driver]) + t_pitdrive_inlap

        idx_act = self.race.fcy_handling["idxs_act_phase"][self.idx_driver]
        fcy_type = self.race.fcy_data["phases"][idx_act][2] if idx_act is not None else None

        if fcy_type is None:
            return self.average_laptimes_presim[self.idx_driver] - cur_laptime

        fcy_progs = self.race.fcy_handling["start_end_prog"][self.idx_driver]
        if cur_lap - 1 >= fcy_progs[0] and math.isclose(fcy_progs[1], cur_lap):
            delta = 1.0
        elif cur_lap - 1 >= fcy_progs[0]:
            delta = 1 - (cur_lap - fcy_progs[1])
        else:
            delta = fcy_progs[1] - fcy_progs[0]

        avg = delta * self.race.track.t_lap_fcy + (1.0 - delta) * self.average_laptimes_presim[self.idx_driver]
        reward = round(avg - cur_laptime, 6)

        if fcy_type == 'SC':
            rt = self.race.racetimes[cur_lap - 1, self.idx_driver] + self.race.laptimes[cur_lap, self.idx_driver]
            sc_ghost = (self.race.fcy_handling["sc_ghost_racetimes"][self.idx_driver]
                        + self.race.positions[cur_lap, self.idx_driver] * self.race.race_pars['min_t_dist_sc'])
            if rt <= sc_ghost:
                reward = 0.0

        return reward

    def _calculate_reward_position(self, delta_position) -> float:
        reward = (self.race.positions[self.race.cur_lap - 1, self.idx_driver]
                  - self.race.positions[self.race.cur_lap, self.idx_driver]
                  + delta_position)
        return float(5.0 * reward)

    def _calculate_reward_final_position(self) -> float:
        avg_times = {self.race.drivers_list[i].initials: t
                     for i, t in enumerate(self.average_laptimes_presim)
                     if self.race.bool_driving[self.race.race_pars["tot_no_laps"]][i]}
        sorted_drivers = [k for k, v in sorted(avg_times.items(), key=lambda x: x[1])]
        pos_avg = sorted_drivers.index(self.driver_initials) + 1
        pos_race = self.race.positions[self.race.get_last_compl_lap(idx=self.idx_driver), self.idx_driver]
        return float(pos_avg - pos_race)
```

---

### 5.5 Nouveau script d'entraînement SB3 (`main_train_sb3.py`)

Créez le script racine `main_train_sb3.py` pour lancer l'entraînement DQN ou PPO avec logging TensorBoard :

```python
"""
Script d'entraînement moderne avec Stable-Baselines3 (PyTorch).
Remplace main_train_rl_agent_dqn.py sans dépendre de tf-agents.
"""

import os
import numpy as np
import racesim
from machine_learning_rl_training.src.rl_environment_gymnasium import RaceSimulationGymnasium
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env

# Paramètres de simulation
race              = "Spa_2019"
controlled_driver = "HAM"
vse_others        = "supervised"
mcs_pars_file     = "pars_mcs.ini"
race_pars_file    = f"pars_{race}.ini"
num_timesteps     = 250_000
output_dir        = "machine_learning_rl_training/output"
os.makedirs(output_dir, exist_ok=True)

# Vérification course sèche
pars_in = racesim.src.import_pars.import_pars(
    use_print=False, use_vse=False,
    race_pars_file=race_pars_file, mcs_pars_file=mcs_pars_file)[0]
for driver in pars_in["driver_pars"]:
    if any(s[1] in ["I", "W"] for s in pars_in["driver_pars"][driver]["strategy_info"]):
        raise RuntimeError(f"Course {race} humide : entraînement non supporté.")

# Environnements
env_kwargs = dict(
    race_pars_file=race_pars_file,
    mcs_pars_file=mcs_pars_file,
    vse_type=vse_others,
    use_prob_infl=True,
    create_rand_events=True,
    controlled_driver=controlled_driver
)

train_env = RaceSimulationGymnasium(**env_kwargs)
eval_env  = RaceSimulationGymnasium(**env_kwargs)

check_env(train_env, warn=True)

# Modèle DQN Stable-Baselines3 (PyTorch)
model = DQN(
    policy="MlpPolicy",
    env=train_env,
    learning_rate=1e-3,
    buffer_size=200_000,
    learning_starts=200,
    batch_size=64,
    gamma=0.97,
    train_freq=1,
    target_update_interval=1,
    policy_kwargs={"net_arch": [64, 64]},
    exploration_fraction=0.5,
    exploration_final_eps=0.01,
    verbose=1,
    tensorboard_log=os.path.join(output_dir, "tensorboard"),
    device="auto",  # Utilise automatiquement CUDA si GPU disponible
)

# Callback d'évaluation
eval_callback = EvalCallback(
    eval_env,
    n_eval_episodes=100,
    eval_freq=50_000,
    best_model_save_path=os.path.join(output_dir, f"best_{race}"),
    verbose=1,
)

# Lancement de l'entraînement
print(f"INFO: Lancement entraînement DQN sur {race} (Pilote: {controlled_driver})...")
model.learn(total_timesteps=num_timesteps, callback=eval_callback, progress_bar=True)

# Sauvegarde finale
final_model_path = os.path.join(output_dir, f"dqn_{race}_{controlled_driver}")
model.save(final_model_path)
print(f"RESULT: Modèle sauvegardé avec succès dans {final_model_path}.zip")

# Évaluation statistique finale
num_eval_races = 10_000
returns, positions = [], []
print(f"INFO: Évaluation sur {num_eval_races} courses simulées...")
for i in range(num_eval_races):
    obs, _ = eval_env.reset()
    done, ep_return = False, 0.0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = eval_env.step(int(action))
        ep_return += reward
        done = terminated or truncated

    final_lap = eval_env.race.get_last_compl_lap(idx=eval_env.idx_driver)
    pos = int(eval_env.race.positions[final_lap, eval_env.idx_driver])
    returns.append(ep_return)
    positions.append(pos)
    if (i + 1) % 2500 == 0:
        print(f"  Progression: {i+1}/{num_eval_races}")

print(f"RESULT: Return moyen : {np.mean(returns):.3f}")
print(f"RESULT: Position finale moyenne : {np.mean(positions):.2f}")
print(f"RESULT: Taux de podium : {np.mean([p <= 3 for p in positions]):.1%}")
```

---

## 6. Option : 100% Zéro TensorFlow via ONNX Runtime

Si vous souhaitez éliminer intégralement TensorFlow du projet (y compris pour l'inférence des modèles de stratégie supervisés déjà fournis), utilisez **ONNX Runtime** :

### Conversion des modèles `.tflite` vers `.onnx` (une seule fois) :
```bash
pip install tf2onnx onnxruntime
python -m tf2onnx.convert --tflite model.tflite --output model.onnx
```

### Remplacement dans `racesim/src/vse_supervised.py` et `vse_reinforcement.py` :
```python
# Remplacer :
import tensorflow as tf
self.nnmodel = {"interpreter": tf.lite.Interpreter(model_path=path)}
self.nnmodel["interpreter"].allocate_tensors()
# ...

# Par :
import onnxruntime as ort
self.nnmodel = ort.InferenceSession(path)
input_name = self.nnmodel.get_inputs()[0].name

# Lors de l'inférence :
output = self.nnmodel.run(None, {input_name: data})[0]
```

---

## 7. Commandes d'installation et de lancement

### Étape 1 : Créer un nouvel environnement virtuel propre (Python 3.11)
Dans votre terminal WSL Ubuntu :
```bash
cd ~/projects/race-simulation-rl
python3.11 -m venv venv_rl_modern
source venv_rl_modern/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Étape 2 : Lancer l'entraînement
```bash
python main_train_sb3.py
```

### Étape 3 : Visualiser l'apprentissage dans TensorBoard
```bash
tensorboard --logdir machine_learning_rl_training/output/tensorboard
```
Ouvrez ensuite `http://localhost:6006` dans votre navigateur.
