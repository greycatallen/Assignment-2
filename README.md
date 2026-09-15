# Assignment 2 — Training a Ms. Pac-Man Agent with a DQN

Class 3 assignment. A Deep Q-Network is trained on `ALE/MsPacman-v5` using the supplied notebook,
then evaluated against its own untrained starting point under identical settings. I ran the
experiment **eleven times**, sweeping the learning rate across five values and exploration across
three, and the sweep — not any single run — is the result.

Starter notebook adapted from [pepealonso95/pacman-dqn](https://github.com/pepealonso95/pacman-dqn).

> **Headline: my best run scored 494 against an untrained baseline of 492 — a 2-point gain that is
> not a real improvement.** Its median *fell* 210 points, three of five games got worse, and ten
> other configurations landed between 156 and 464. Across the sweep, training score and evaluation
> score move in **opposite** directions, and gameplay recordings show the trained networks
> repeatedly collapsing onto degenerate policies — in four runs reproducing the untrained network's
> gameplay frame-for-frame. See [The finding](#the-finding).

## Open and run

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/greycatallen/Assignment-2/blob/main/pacman_dqn.ipynb)

- **Colab:** click the badge, choose Runtime → Change runtime type → T4 GPU, then Runtime → Run all.
  The first cell installs every package. My three values are already set in section 1.
- **Local Jupyter / VS Code:** clone this repo, select a Python 3.11–3.13 kernel,
  `pip install -r requirements.txt`, and Run All.

## My three hyperparameters

Settings for the headline run and for `pacman_dqn.ipynb` in this repository.

| Setting | Value | Why I chose it |
|---|---|---|
| Exploration | `0.20` | Keeps one training move in five random after warm-up. Ms. Pac-Man rewards nothing for standing still, so the agent needs forced variety to keep meeting pellets and ghosts it would not choose itself. I also tested 0.25 and 0.40; 0.20 is the notebook's reference and the value my best run used. |
| Episodes | `100` | My first run used 5 episodes and produced only 388 learning updates — far too few to conclude anything. 100 episodes gives ~62,000 decisions and ~15,200 updates, and every run below except the setup check uses it, so runs are comparable. |
| Learning rate | `0.001` | Reached by sweeping. I began at `0.01`, which spiked the loss to 16.09 in episode 2, then worked downward through 0.002, 0.0012, 0.0005, 0.0002 and 0.0001. `0.001` produced the highest evaluation mean of the eleven. As the sweep below shows, that ranking is not stable, and I do not claim it is the right value. |

Every other setting is the notebook's fixed classroom default. **Evaluation settings are untouched
in all eleven runs**: the same five seeds (101, 202, 303, 404, 505), 5% exploration, and the same
3,000-decision cap, applied identically before and after training.

## What I expected

I expected training to improve the agent, and early on I read the rising training-score curve as
evidence that it was. After the first few runs came back below baseline I revised that: by the time
I ran this configuration I expected it to land in the same 200–460 band as everything else, and I
did not expect to beat the untrained network. Clearing 492 by 2 points was better than I predicted
and still, on inspection, not an improvement.

## What I observed

### All five before/after evaluation scores

Same five seeds, 5% exploration, same step cap, untrained network as baseline.
Full data: [`results/comparison.json`](results/comparison.json).

| Seed | Untrained | Trained | Change |
|---:|---:|---:|---:|
| 101 | 350 | 700 | +350 |
| 202 | 500 | 280 | −220 |
| 303 | 320 | **1000** | +680 |
| 404 | 800 | 240 | −560 |
| 505 | 490 | 250 | −240 |
| **Mean** | **492.0** | **494.0** | **+2.0** |
| Median | 490 | 280 | **−210** |

**The mean is the wrong summary here.** Three of five games got worse. Two large wins (seeds 303 and
101) offset three losses, and the median dropped 210 points. With individual scores spanning
240–1000, a 2-point difference in the mean is far inside the noise. This run is the best of eleven
and still does not demonstrate that training helped.

### Training plot

![Training dashboard](results/training_dashboard.png)

- **Raw training score:** the 25-game average sits around 700 for most of the run and drifts down to
  546 by episode 100. Individual games are wildly variable, including a **4,430-point game at
  episode 12** — my highest anywhere — and a 3,000-point game near episode 65. No trend.
- **Mean update loss:** *rises* steadily from 0.026 to about 0.08. No spike, no `NaN`.
- **Training exploration:** flat at 0.20 after warm-up ends in episode 2, as configured.

### Gameplay

Every clip is the first 20 seconds at 4× speed, playing twice before stopping. The trained clip is
the best of the five evaluation games, chosen on full-game score.

| Untrained (baseline) | Trained, best of five |
|---|---|
| ![Untrained](results/untrained.gif) | ![Trained](results/trained_best.gif) |

Intermediate samples every 25 episodes ([`results/demo_scores.json`](results/demo_scores.json)), all
on seed 101, whose untrained baseline was 350:

| Episode 25 | Episode 50 | Episode 75 | Episode 100 |
|---|---|---|---|
| ![ep25](results/episode_0025.gif) | ![ep50](results/episode_0050.gif) | ![ep75](results/episode_0075.gif) | ![ep100](results/episode_0100.gif) |
| 230 | 320 | 200 | 700 |

Three of the four sit below the 350 baseline before the final sample jumps to 700 — the same
volatility the evaluation table shows.

### Actual training budget

From [`results/training_summary.json`](results/training_summary.json) and [`results/config.json`](results/config.json):

| | |
|---|---|
| Status | `completed` — not interrupted |
| Episodes completed | 100 of 100 requested |
| Total decisions | 61,787 |
| Learning updates | **15,197** |
| Elapsed training time | 377.5 s (6.3 min), including periodic sample capture |
| Hardware | CPU (Colab, Linux x86-64), PyTorch 2.9.0+cpu, Python 3.13.15 |

Colab reported CPU rather than GPU. It made little practical difference — the Atari emulator is
single-threaded CPU work that dominates the loop — but the hardware is recorded honestly as CPU.
Per-episode data is in [`results/training.csv`](results/training.csv).

## The finding

### 1. Training score and evaluation score move in opposite directions

The agent trains at 20–40% exploration with weights updating, and is evaluated at 5% with weights
frozen. Those are different measurements, and across eleven runs they disagree:

| Learning rate | ε | Mean score *during* training | **Evaluation mean** | Median |
|---:|---:|---:|---:|---:|
| 0.001 | 0.20 | 662.6 | **494** ← headline | 280 |
| 0.0002 | 0.25 | 638.6 | 464 | 430 |
| 0.0001 | 0.40 | 563.2 | 458 | 410 |
| 0.0005 | 0.40 | 579.1 | 422 | 430 |
| 0.01 | 0.20 | 505.7 | 408 | 440 |
| 0.01 | 0.40 | 413.2 | 408 | 440 |
| 0.002 | 0.20 | 637.8 | 304 | 270 |
| 0.0012 | 0.20 | 638.6 | 218 | 240 |
| 0.002 | 0.40 | 499.0 | 198 | 180 |
| **0.0005** | **0.20** | **682.5** | **156** | **70** |
| 0.0001 (5 ep) | 0.20 | 316.0 | 730 | 440 |

Baseline evaluation mean is **492.0** in every run — `SEED = 42` fixes the untrained weights.

The last 100-episode row is the clearest case in the whole project: **the highest training mean of
any run, 682.5, produced the lowest evaluation mean, 156**, with three of its five games scoring 70.
A policy can post strong training numbers while random moves are carrying it, and lose everything
when evaluation removes that crutch. Nothing in the training curve warns you — only the held-out
evaluation does.

The 730 in the final row is not a real result either: that run had 388 updates, its gain came from a
single seed, and its median fell. It scores highest because it barely trained.

### 2. The trained networks collapse onto degenerate policies

Evaluation is fully deterministic — the environment is seeded per evaluation seed and the
exploration RNG alongside it — so two recordings of the same seed are byte-identical only if the
network chose the same action at every step. Hashing all 55 gameplay recordings across the eleven
published runs:

- **In four separate runs, a trained checkpoint reproduced the untrained network's gameplay exactly**
  (hash `5fc7e8`): `lr0.01_eps0.20` at ep 50, `lr0.002_eps0.20` at ep 75, `lr0.002_eps0.40` at ep 25,
  and `lr0.0005_eps0.20` at ep 25.
- **`lr0.0012_eps0.20` played an identical game at episodes 25, 50 and 75** (`d1d9c4`) — 50 episodes
  and ~7,000 weight updates changed its behaviour not at all.
- **`lr0.01_eps0.20` and `lr0.01_eps0.40` produced byte-identical final policies.** Doubling
  exploration changed the training data substantially (3,400 fewer decisions, 855 fewer updates,
  training mean 92 points lower) and the resulting agent was the same to the byte — identical scores
  `[250, 440, 840, 440, 70]`, identical step counts, identical recordings.
- Trajectories `ade1eb`, `1b48ad`, `44878d` and `c44157` each recur across runs that share no
  training history. Even the headline run repeats `ade1eb` at episode 75.

The networks are not learning bad policies so much as **falling into a small set of near-constant
ones**. That explains the tightly clustered low scores (170–240 at lr 0.0012; three 70s at
lr 0.0005) and why the 20% random moves mattered so much during training.

I infer collapse from identical trajectories rather than measuring it directly. The check that would
settle it is loading a checkpoint and printing the distribution of `argmax` actions over a batch of
states; if the network has collapsed onto one or two of the nine joystick moves, this is confirmed.

### 3. No hyperparameter I varied produced a systematic effect

At fixed exploration 0.20, sorted by learning rate:

| lr | 0.01 | 0.002 | 0.0012 | 0.001 | 0.0005 |
|---|---:|---:|---:|---:|---:|
| Evaluation mean | 408 | 304 | 218 | **494** | **156** |

`0.0012` → 218 and `0.001` → 494 are nearly the same learning rate and 276 points apart. Exploration
is no more orderly: lr 0.0005 gives 156 at ε 0.20 and 422 at ε 0.40, while lr 0.01 gives 408 at both.
Ten 100-episode runs produced one result above baseline, by 2 points. **Run-to-run variance dominates
any hyperparameter effect I was able to produce**, and I do not believe the ranking in the sweep table
would survive re-running with different seeds.

## How the agent works, in plain language

- **Observations.** The agent never sees the game's memory or a list of objects. It sees **four
  consecutive game screens**, shrunk to 84 × 84 grayscale images and stacked. Four frames rather than
  one is what makes motion visible: a single still image cannot tell you whether a ghost is
  approaching or retreating.
- **Actions.** It picks one of the **nine joystick moves** — eight directions plus no-op — and holds
  it for four emulator frames, so one decision covers about a fifteenth of a second.
- **Rewards.** The reward is the **game points** earned in that interval: pellets, power pellets,
  edible ghosts, fruit. During training these are clipped to [−1, 1] so no single event dominates an
  update, but every score in this README is the raw, unclipped game score.
- **Learning.** Each experience goes into a replay memory. Every four decisions the agent samples 32
  past experiences at random and nudges its value estimates toward the reward received plus the
  discounted value of the next state, judged by a slowly-updated target network. Random sampling
  breaks the correlation between consecutive frames; the target network stops the agent chasing its
  own moving estimate.

## One limitation

**Training score is not a valid measure of what the agent has learned, and I spent several runs
treating it as one.** The agent is trained at 20–40% exploration and scored at 5%, so a policy that
is bad on its own can post strong training numbers while random moves rescue it. My highest training
mean belongs to my worst-evaluating run. Nothing in the score or loss panel flags this — the loss in
that run was smooth and low — and it took hashing the gameplay recordings, which is not part of the
notebook's standard output, to see what was actually happening.

A second limitation bounds every conclusion above: **one run per configuration, one seed, five
evaluation games with scores spanning 70–1000.** That is enough to say no configuration reliably beat
the baseline — eleven runs agree on that. It is not enough to rank configurations against each other,
and section 3 shows the ranking is unstable. Properly separating a hyperparameter effect from noise
would need several seeds per configuration and many more than five evaluation games.

## My next experiment

**Change only `REPLAY_CAPACITY`, from 5,000 to 50,000.** Learning rate stays at 0.001, exploration at
0.20, budget at 100 episodes.

I am leaving the learning rate and exploration alone because I have now tested five values of one and
three of the other, eleven runs in total, and section 3 shows neither produces a systematic effect.
Testing a sixth learning rate would sample the same noise again.

Replay capacity is the one structural setting I have never varied, and it is the most plausible
remaining cause of the collapse. At 5,000 transitions the buffer holds roughly **ten episodes** of
experience. A collapsed policy generates homogeneous states, those states fill the buffer within ten
episodes, and the agent then trains almost entirely on its own degenerate behaviour — a feedback loop
that would explain identical trajectories 50 episodes apart. A 10× larger buffer retains early,
more varied experience and breaks that loop, and it is the one change that attacks the mechanism
rather than the symptom.

The prediction, stated so it can fail: the intermediate recordings should stop repeating — no hash
appearing twice within a run, and none matching the untrained `5fc7e8` — and the evaluation mean
should clear 492 with the median rising rather than falling. If recordings still repeat with a 50,000
buffer, the cause is in the optimisation itself rather than the data, and the argmax-distribution
check described in the finding becomes the next step.

## All eleven runs

Every run is published with its full evidence: config, comparison, baseline, per-episode CSV, summary,
dashboard, and all gameplay recordings.

| Learning rate | ε | Episodes | Updates | Evaluation mean | Evidence |
|---:|---:|---:|---:|---:|---|
| **0.001** | **0.20** | **100** | **15,197** | **494** | `results/` (headline) |
| 0.0002 | 0.25 | 100 | 14,427 | 464 | [`runs/lr0.0002_eps0.25`](results/runs/lr0.0002_eps0.25) |
| 0.0001 | 0.40 | 100 | 14,130 | 458 | [`runs/lr0.0001_eps0.40`](results/runs/lr0.0001_eps0.40) |
| 0.0005 | 0.40 | 100 | 14,371 | 422 | [`runs/lr0.0005_eps0.40`](results/runs/lr0.0005_eps0.40) |
| 0.01 | 0.20 | 100 | 13,488 | 408 | [`runs/lr0.01_eps0.20`](results/runs/lr0.01_eps0.20) |
| 0.01 | 0.40 | 100 | 12,633 | 408 | [`runs/lr0.01_eps0.40`](results/runs/lr0.01_eps0.40) |
| 0.002 | 0.20 | 100 | 14,133 | 304 | [`runs/lr0.002_eps0.20`](results/runs/lr0.002_eps0.20) |
| 0.0012 | 0.20 | 100 | 14,280 | 218 | [`runs/lr0.0012_eps0.20`](results/runs/lr0.0012_eps0.20) |
| 0.002 | 0.40 | 100 | 13,702 | 198 | [`runs/lr0.002_eps0.40`](results/runs/lr0.002_eps0.40) |
| 0.0005 | 0.20 | 100 | 15,008 | 156 | [`runs/lr0.0005_eps0.20`](results/runs/lr0.0005_eps0.20) |
| 0.0001 | 0.20 | 5 | 388 | 730 | [`runs/lr0.0001_eps0.20_5ep`](results/runs/lr0.0001_eps0.20_5ep) |

All eleven completed without interruption, and every one recorded a non-zero learning-update count.

## Repository contents

| Path | What it is |
|---|---|
| [`pacman_dqn.ipynb`](pacman_dqn.ipynb) | The executed notebook for the headline run |
| [`results/`](results) | Headline run evidence: comparison, config, baseline, training CSV, summary, demo scores, dashboard, and all six gameplay GIFs |
| [`results/runs/`](results/runs) | The other ten runs, one folder each, same file set |
| [`pacman_dqn_annealed.ipynb`](pacman_dqn_annealed.ipynb) | A variant I prepared that anneals exploration from 0.50 to 0.10 over 40,000 decisions instead of holding it constant. **Never run** — kept because it is the modification I would pair with a larger replay buffer. |
| [`pacman_player.py`](pacman_player.py) | Optional floating gameplay window for local runs |
| [`requirements.txt`](requirements.txt) | Dependencies for a local run |
| [`tests/verify_notebook.py`](tests/verify_notebook.py) | Optional end-to-end execution check |

### Where the model checkpoints are

The playback checkpoints — `untrained.pt`, `trained.pt`, and `episode_0025/0050/0075/0100.pt`,
6.7 MB each, roughly 37 MB per run and over 400 MB across eleven runs — are **not committed**. They
are excluded by `.gitignore` along with the whole `pacman_runs/` folder, and are kept in the original
run ZIPs stored locally; the headline run is `20260915_033932_047120.zip`. They are needed only to
replay a saved agent. Every number and image in this README comes from the JSON, CSV, PNG and GIF
files committed above.
