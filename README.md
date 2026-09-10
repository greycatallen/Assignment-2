# Assignment 2 — Training a Ms. Pac-Man Agent with a DQN

Class 3 assignment. A Deep Q-Network is trained on `ALE/MsPacman-v5` with the supplied notebook,
then evaluated against its own untrained starting point under identical settings. I ran the
experiment four times, sweeping the learning rate.

Starter notebook adapted from [pepealonso95/pacman-dqn](https://github.com/pepealonso95/pacman-dqn).

> **Headline result: training made the agent worse in every run, and the cause is not the learning
> rate.** In my headline run the evaluation mean fell from **492 to 304**, with all five seeds
> regressing — while the *training* score over the same run rose to a peak of 691. Comparing gameplay
> recordings across runs shows why: the trained networks **collapse onto a handful of near-constant
> policies**, and several recorded games are byte-for-byte identical to each other and to the
> untrained network's. See [The finding](#the-finding).

## Open and run

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/greycatallen/Assignment-2/blob/main/pacman_dqn.ipynb)

- **Colab:** click the badge, choose Runtime → Change runtime type → T4 GPU, then Runtime → Run all.
  The first cell installs every package. My three values are already set in section 1.
- **Local Jupyter / VS Code:** clone this repo, select a Python 3.11–3.13 kernel, `pip install -r
  requirements.txt`, and Run All.

## My three hyperparameters

Settings for the headline run, the notebook in this repository, and everything in `results/`
outside the subfolders.

| Setting | Value | Why I chose it |
|---|---|---|
| Exploration | `0.20` | Keeps one training move in five random after warm-up. Ms. Pac-Man rewards nothing for standing still, so the agent needs forced variety to keep meeting pellets and ghosts it would not choose itself. I held it constant across all four runs so the learning rate stayed the only variable. **My results now implicate this setting — see the next experiment.** |
| Episodes | `100` | My first run used 5 episodes and produced only 388 learning updates, far too few to conclude anything. 100 episodes gives ~57,500 decisions and ~14,100 updates. |
| Learning rate | `0.002` | Chosen in response to my `0.01` run, where loss spiked to 16.09 in episode 2 and the agent ended worse than untrained — the signature of too large a step. `0.002` is a fifth of that but still 20× the 0.0001 reference, keeping the experiment informative rather than reverting to the default. |

Every other setting is the notebook's fixed classroom default. Evaluation settings are untouched in
all four runs: the same five seeds (101, 202, 303, 404, 505), 5% exploration, and the same
3,000-decision cap, applied identically before and after training.

## What I expected

I predicted the smaller step would repair the damage `0.01` had done, and that **improvement would
continue** through training rather than stalling early — a training score that kept climbing and an
evaluation mean that finally landed above the untrained baseline of 492.

## What I observed

The training half came true, then faded. The evaluation half did not happen at all.

### All five before/after evaluation scores

Same five seeds, 5% exploration, same step cap, untrained network as baseline.
Full data: [`results/comparison.json`](results/comparison.json).

| Seed | Untrained | Trained | Change |
|---:|---:|---:|---:|
| 101 | 350 | 210 | −140 |
| 202 | 500 | 380 | −120 |
| 303 | 320 | 270 | −50 |
| 404 | 800 | 430 | −370 |
| 505 | 490 | 230 | −260 |
| **Mean** | **492.0** | **304.0** | **−188.0** |
| Median | 490 | 270 | −220 |

**Every seed regressed.** Both mean and median fell. There is no outlier to argue about.

### Training plot

![Training dashboard](results/training_dashboard.png)

The loss panel confirms the step size was fixed: mean update loss stays between **0.028 and 0.095
for the whole run** — no spike, no `NaN`. At `0.01` the same panel peaked at 16.09.

The score panel is where my prediction looked correct:

| Episode | 25-game average training score |
|---:|---:|
| 1 | 130.0 |
| 10 | 469.0 |
| 25 | 610.0 |
| 50 | 611.6 |
| 60 | **691.2** |
| 100 | 649.2 |

Training score climbs to a peak of 691 near episode 60, then eases to 649. Best single training game
was **2,160** at episode 70. Unlike my `0.01` run this rise is not a warm-up artefact — it continues
long after exploration settles at 0.20 in episode 2.

### The finding

Two facts have to be reconciled:

| Measurement | Exploration | Weights | Mean score |
|---|---|---|---:|
| Trained agent, during training | 0.20 | updating | ~650–690 |
| Trained agent, at evaluation | 0.05 | frozen | **304** |
| Untrained network, at evaluation | 0.05 | frozen | 492 |

The agent scores well when one move in five is random and badly when only one in twenty is. Its own
untrained weights beat it at 5%.

**The gameplay recordings show why.** Evaluation is fully deterministic — the environment is seeded
per evaluation seed and the exploration RNG is seeded alongside it — so two recordings of the same
seed are identical only if the network chose the same action at every step. Hashing the intermediate
clips across the three 100-episode runs:

| Run | ep 0 (untrained) | ep 25 | ep 50 | ep 75 | ep 100 |
|---|---|---|---|---|---|
| lr 0.01 | `5fc7e8` | `c44157` | **`5fc7e8`** | `ade1eb` | `1b48ad` |
| lr 0.002 (headline) | `5fc7e8` | `9d29d7` | `2c8cd1` | **`5fc7e8`** | `516583` |
| lr 0.0012 | `5fc7e8` | **`d1d9c4`** | **`d1d9c4`** | **`d1d9c4`** | `ade1eb` |

Three results fall out:

1. **At lr 0.0012 the agent played an identical game at episodes 25, 50 and 75.** Fifty episodes and
   roughly 7,000 weight updates changed its behaviour on that seed not at all.
2. **At lr 0.01 (ep 50) and lr 0.002 (ep 75), the trained agent reproduced the untrained network's
   trajectory exactly** (`5fc7e8`). Its greedy policy chose the same actions as random weights.
3. **`ade1eb` appears in two different runs** — lr 0.01 at episode 75 and lr 0.0012 at episode 100.
   Different learning rates and training histories, identical behaviour.

Fifteen recordings contain only a handful of distinct trajectories. The networks are not learning bad
policies; they are **collapsing onto a small set of near-constant-action policies**. That explains the
whole pattern: a constant-action policy is carried by the 20% random moves during training and exposed
at 5% during evaluation, and it is why the trained scores cluster so tightly (170–240 at lr 0.0012).

It also reframes the sweep. The differences between 408, 304 and 218 are not degrees of learning —
they are which degenerate policy each run happened to land on.

The check that would pin this down completely is loading a checkpoint and printing the distribution of
`argmax` actions over a batch of states; I infer collapse from identical trajectories rather than
having measured the action distribution directly.

### Gameplay

Every clip is the first 20 seconds at 4× speed, playing twice before stopping. The trained clip is the
best of the five evaluation games, chosen on full-game score.

| Untrained (baseline) | Trained, best of five |
|---|---|
| ![Untrained](results/untrained.gif) | ![Trained](results/trained_best.gif) |

Intermediate samples every 25 episodes ([`results/demo_scores.json`](results/demo_scores.json)), all on
seed 101, whose untrained baseline was 350:

| Episode 25 | Episode 50 | Episode 75 | Episode 100 |
|---|---|---|---|
| ![ep25](results/episode_0025.gif) | ![ep50](results/episode_0050.gif) | ![ep75](results/episode_0075.gif) | ![ep100](results/episode_0100.gif) |
| 480 | 180 | 350 | 210 |

These run at evaluation exploration and never establish a trend above the 350 baseline.

### Actual training budget

From [`results/training_summary.json`](results/training_summary.json) and [`results/config.json`](results/config.json):

| | |
|---|---|
| Status | `completed` — not interrupted |
| Episodes completed | 100 of 100 requested |
| Total decisions | 57,529 |
| Learning updates | **14,133** |
| Elapsed training time | 367.4 s (6.1 min), including periodic sample capture |
| Hardware | CPU (Colab, Linux x86-64), PyTorch 2.9.0+cpu, Python 3.13.15 |

The run reported CPU rather than GPU. It made little practical difference — the Atari emulator is
single-threaded CPU work that dominates the loop — but the hardware is recorded honestly as CPU.
Per-episode data is in [`results/training.csv`](results/training.csv).

## All four runs

Same five evaluation seeds, 5% evaluation exploration, and training exploration 0.20 throughout. The
untrained baseline is identical in every run because `SEED = 42` fixes the starting weights.

| Run | Learning rate | Episodes | Updates | Mean score during training (ε 0.20) | Evaluation mean (ε 0.05) | Evidence |
|---|---:|---:|---:|---:|---:|---|
| Setup check | 0.0001 | 5 | 388 | 316.0 | 730 | [`setup_check_5ep/`](results/setup_check_5ep) |
| Second | 0.01 | 100 | 13,488 | 505.7 | 408 | [`lr_0.01_100ep/`](results/lr_0.01_100ep) |
| **Headline** | **0.002** | **100** | **14,133** | **637.8** | **304** | `results/` |
| Fourth | 0.0012 | 100 | 14,280 | 638.6 | 218 | [`lr_0.0012_100ep/`](results/lr_0.0012_100ep) |

Baseline evaluation mean is **492.0** for all four.

Two things stand out. Across the three comparable 100-episode runs, the mean score *during* training
rose (506 → 638 → 639) while the *evaluation* mean fell (408 → 304 → 218) — the two measures move in
opposite directions. And the 730 in the first row is not a real result: that run had 388 updates, its
gain came from one seed, and its median fell. It scores highest precisely because it barely trained,
and therefore had not yet collapsed.

## How the agent works, in plain language

- **Observations.** The agent never sees the game's memory or a list of objects. It sees **four
  consecutive game screens**, shrunk to 84 × 84 grayscale images and stacked. Four frames rather than
  one is what makes motion visible: a single still image cannot tell you whether a ghost is
  approaching or retreating.
- **Actions.** It picks one of the **nine joystick moves** — eight directions plus no-op — and holds it
  for four emulator frames, so one decision covers about a fifteenth of a second.
- **Rewards.** The reward is the **game points** earned in that interval: pellets, power pellets, edible
  ghosts, fruit. During training these are clipped to [−1, 1] so no single event dominates an update,
  but every score in this README is the raw, unclipped game score.
- **Learning.** Each experience goes into a replay memory. Every four decisions the agent samples 32
  past experiences at random and nudges its value estimates toward the reward received plus the
  discounted value of the next state, judged by a slowly-updated target network. Random sampling breaks
  the correlation between consecutive frames; the target network stops the agent chasing its own moving
  estimate.

## One limitation

**Training score is not a valid measure of what the agent has learned, and for two runs I treated it as
one.** The agent is trained at 0.20 exploration and scored at 0.05, so a policy that is bad on its own
can post good training numbers as long as random moves keep rescuing it. My training curve peaked at 691
in the run whose evaluation was worst, and rose across the sweep exactly as evaluation fell. Nothing in
the score panel warns you; only the held-out evaluation does. Identical gameplay recordings were what
finally made the failure legible, and they are not part of the notebook's standard output.

The second limit is on the conclusion. I have one run per learning rate and one seed, and evaluation
spans five games with scores ranging 70–840. That is enough to say all four runs failed to beat the
baseline, but not enough to rank them against each other — and the collapse evidence suggests the
ranking is close to meaningless anyway.

## My next experiment

**Change only `EXPLORATION`, from 0.20 to 0.05.** Learning rate stays at 0.002 and the budget stays at
100 episodes.

I am leaving the learning rate alone because I have now tested four values — 0.0001, 0.01, 0.002 and
0.0012 — and all three trained runs collapsed the same way. Going from 0.01 to 0.002 removed the loss
spike entirely and improved every training-side measure, and the evaluation mean still fell. A fifth
value would tune a variable that is already behaving.

What the evidence points at is the mismatch between how the agent collects experience and how it is
judged. It trains at 20% randomness and is scored at 5%, and that gap — ~690 during training against
304 at evaluation — is the largest unexplained number in this report. Training at 0.05 makes the two
match, so the replay memory fills with the states the agent will actually face when scored, and its
greedy policy is trained on its own consequences rather than on outcomes random moves produced.

The prediction, stated so it can fail: training score should **drop** toward the 300s while the
evaluation mean **rises** toward or above 492, and the intermediate recordings should stop repeating.
If evaluation stays low and the clips still hash identically, exploration is not the cause either, and
the next thing I would raise is the 5,000-transition replay memory — about ten episodes of experience,
small enough that the agent may simply be overwriting its own history faster than it can learn from it.

## Repository contents

| Path | What it is |
|---|---|
| [`pacman_dqn.ipynb`](pacman_dqn.ipynb) | The executed notebook for the headline run |
| [`results/comparison.json`](results/comparison.json) | All five before/after scores and means |
| [`results/config.json`](results/config.json) | Hyperparameters, hardware, exact package versions |
| [`results/training.csv`](results/training.csv) | One row per training episode |
| [`results/training_summary.json`](results/training_summary.json) | Episodes, decisions, updates, elapsed time, status |
| [`results/baseline.json`](results/baseline.json) | Untrained evaluation detail |
| [`results/demo_scores.json`](results/demo_scores.json) | Scores for the four intermediate samples |
| [`results/training_dashboard.png`](results/training_dashboard.png) | Score, loss, and exploration curves |
| `results/untrained.gif`, `results/trained_best.gif` | Baseline and best trained gameplay |
| `results/episode_0025.gif` … `episode_0100.gif` | Intermediate gameplay every 25 episodes |
| [`results/lr_0.01_100ep/`](results/lr_0.01_100ep) | Full evidence, learning rate 0.01 |
| [`results/lr_0.0012_100ep/`](results/lr_0.0012_100ep) | Full evidence, learning rate 0.0012 |
| [`results/setup_check_5ep/`](results/setup_check_5ep) | Full evidence, 5-episode setup check |
| [`pacman_player.py`](pacman_player.py) | Optional floating gameplay window for local runs |
| [`requirements.txt`](requirements.txt) | Dependencies for a local run |

### Where the model checkpoints are

The playback checkpoints — `untrained.pt`, `trained.pt`, and `episode_0025/0050/0075/0100.pt`, 6.7 MB
each — are **not committed**. They are excluded by `.gitignore` along with the whole `pacman_runs/`
folder, and are kept in the run ZIPs stored locally: `20260910_060222_302005.zip` (headline),
`20260910_055110_655234.zip` (lr 0.01), `20260910_061316_548062.zip` (lr 0.0012), and
`20260910_052510_490439.zip` (setup check). They are needed only to replay a saved agent; every number
and image in this README comes from the JSON, CSV, PNG, and GIF files committed above.
