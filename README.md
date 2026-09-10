# Assignment 2 — Training a Ms. Pac-Man Agent with a DQN

Class 3 assignment. A Deep Q-Network is trained on `ALE/MsPacman-v5` using the supplied
notebook, then evaluated against its own untrained starting point under identical settings.

Starter notebook adapted from [pepealonso95/pacman-dqn](https://github.com/pepealonso95/pacman-dqn).

**Headline result: this run did not produce reliable improvement.** The mean score rose from
492 to 730, but that gain comes almost entirely from a single lucky game, and the median score
fell. Details and reasoning are in [What I observed](#what-i-observed) below.

## Open and run

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/greycatallen/Assignment-2/blob/main/pacman_dqn.ipynb)

- **Colab:** click the badge, choose Runtime → Change runtime type → T4 GPU, set the three
  values in section 1, then Runtime → Run all. The first cell installs all packages.
- **Local Jupyter / VS Code:** clone this repo, select a Python 3.11–3.13 kernel, and Run All.
  `pip install -r requirements.txt` covers the dependencies. Keep `pacman_player.py` beside the
  notebook for the local floating gameplay window.

The notebook detects CUDA, Apple Silicon MPS, or CPU automatically.

## My three hyperparameters

| Setting | Value | Why I chose it |
|---|---|---|
| Exploration | `0.20` | Keeps one move in five random after warm-up. Ms. Pac-Man gives no reward for standing still, so the agent needs steady forced variety to encounter pellets and ghosts it would not choose on its own. I kept the notebook's reference value so that episodes were the only variable I changed. |
| Episodes | `5` | The assignment describes a five-episode run as a setup check. I used it to confirm the full pipeline — install, baseline evaluation, training, GIF capture, checkpointing, ZIP export — before committing to a long run. |
| Learning rate | `0.0001` | The standard Adam step size for DQN and the notebook's reference point. Larger values destabilise Q-learning because the target network moves under the learner; I had no evidence yet that would justify departing from it. |

All other settings are the notebook's fixed classroom defaults, unchanged. Evaluation settings
in particular are untouched: the same five seeds (101, 202, 303, 404, 505), 5% exploration, and
the same 3,000-decision cap, applied identically before and after training.

## What I expected

Five episodes is roughly 2,500 agent decisions, and the first 1,000 are a random warm-up that
produces no weight updates at all. I therefore expected **no meaningful improvement** — at best a
small change indistinguishable from seed noise. I expected the training-loss curve to be
non-zero but flat, since the network cannot converge on so few updates, and I expected the
trained GIF to look much like the untrained one: movement without pellet-seeking or
ghost-avoidance.

## What I observed

That is essentially what happened.

### All five before/after evaluation scores

Same five seeds, 5% exploration, same step cap. Full data: [`results/comparison.json`](results/comparison.json).

| Seed | Untrained | Trained | Change |
|---:|---:|---:|---:|
| 101 | 350 | 440 | +90 |
| 202 | 500 | 240 | −260 |
| 303 | 320 | **1640** | +1320 |
| 404 | 800 | 440 | −360 |
| 505 | 490 | 890 | +400 |
| **Mean** | **492.0** | **730.0** | +238.0 |
| Median | 490 | 440 | −50 |

**The mean is misleading here.** Three of five games improved and two got worse. Seed 303 alone
contributes +1320 of the +1190 net change — remove it and the trained agent is *behind* its
starting point. The median score actually fell, from 490 to 440. With 388 weight updates, the
honest conclusion is that the difference is variance between game seeds, not learned skill.
The baseline is an untrained network, not a random-action agent, which is why its scores are
already in the 320–800 range.

### Training plot

![Training dashboard](results/training_dashboard.png)

Read left to right:

- **Raw training score** rises to 460 at episode 2 and then falls back to 180 by episode 5. No trend.
- **Mean update loss** oscillates between 0.026 and 0.028 with no downward direction. Episode 1
  is absent because warm-up produces no updates.
- **Training exploration** drops from 1.0 to the chosen 0.20 once the 1,000-decision warm-up ends,
  then stays flat, as the notebook specifies.

### Gameplay

Both clips are the first 20 seconds of a game at 4× speed. Each plays twice, then stops on its
final frame. The trained clip is the best of the five evaluation games, selected on full-game score.

| Untrained (baseline) | Trained (best of five) |
|---|---|
| ![Untrained gameplay](results/untrained.gif) | ![Trained gameplay](results/trained_best.gif) |

Watching them side by side, I could not distinguish learned behaviour from the baseline. Neither
clip shows the agent tracking pellet corridors or backing away from a ghost. Both spend time
pressed against walls. The higher score on seed 303 came from a run of pellets the agent happened
to be moving through, not from visible pursuit of them.

**No intermediate GIFs or periodic checkpoints exist for this run.** The notebook samples every
25 episodes (`DEMO_EVERY = 25`), and this run stopped at 5.

### Actual training budget

From [`results/training_summary.json`](results/training_summary.json) and [`results/config.json`](results/config.json):

| | |
|---|---|
| Status | `completed` — not interrupted |
| Episodes completed | 5 of 5 requested |
| Total decisions | 2,550 |
| Learning updates | **388** |
| Elapsed training time | 12.5 s (including periodic demo capture) |
| Hardware | CPU (Colab, Linux x86-64), PyTorch 2.9.0+cpu, Python 3.13.15 |

The low update count is the whole story: warm-up consumes the first 1,000 decisions, and the
notebook then applies one update every four decisions, so 2,550 decisions yield only 388 updates.

I ran this configuration twice, once on a Colab **T4 GPU** and once on **CPU**. Because `SEED = 42`
is fixed, both produced byte-identical scores, decision counts, and update counts. The run
published here is the CPU run; the GPU run is retained locally. Per-episode data for the published
run is in [`results/training.csv`](results/training.csv).

## How the agent works, in plain language

- **Observations.** The agent never sees the game's memory or a list of objects. It sees
  **four consecutive game screens**, cropped and shrunk to 84 × 84 grayscale images and stacked
  together. Four frames rather than one is what makes direction and speed visible: a single still
  image cannot tell you whether a ghost is approaching or retreating.
- **Actions.** The agent picks one of the **nine joystick moves** — the eight directions plus no-op.
  It repeats that choice for four emulator frames, so one decision covers about a fifteenth of a second.
- **Rewards.** The reward is simply the **game points** scored in that interval: pellets, power
  pellets, edible ghosts, fruit. During training these are clipped to the range [−1, 1] so no single
  event dominates a weight update, but every score reported in this README is the raw, unclipped
  game score.
- **Learning.** Each experience is stored in a replay memory. Every four decisions the agent samples
  32 past experiences at random and nudges its value estimates toward the reward it received plus the
  discounted value of the next state, judged by a slowly-updated target network. Random sampling
  breaks the correlation between consecutive frames; the separate target network stops the agent from
  chasing its own moving estimate.

## One limitation

**The training budget is far too small for the conclusion to mean anything.** 388 gradient updates
cannot shape a convolutional network with tens of thousands of parameters into a Pac-Man player, so
this experiment cannot distinguish "DQN does not work here" from "DQN has not started yet." Reporting
the mean improvement of +238 as a success would have been the specific error the assignment warns
against — the per-seed table and the flat loss curve are what reveal it as noise.

A second limitation worth naming: with only five evaluation seeds and score variance of this size,
even a genuinely better agent would be hard to certify. The seed-303 outlier makes that concrete.

## My next experiment

**Change only `EPISODES`, from 5 to 100.** Exploration and learning rate stay at 0.20 and 0.0001.

This is the right single variable because nothing in this run indicates the other two are wrong —
the loss was stable rather than diverging, which is what a bad learning rate looks like, and the
agent was clearly still in the exploratory phase the 0.20 setting is meant to sustain. What the
evidence does show is starvation: 388 updates. Raising the episode budget to 100 yields roughly
50,000 decisions and about 12,000 updates, a thirty-fold increase, which is the smallest change
that could make the before/after comparison informative.

The concrete thing I would look for is a **downward trend in the loss panel** combined with the
per-seed table moving together rather than one seed carrying the mean. Tuning exploration or the
learning rate before that point would be tuning against noise.

## Repository contents

| Path | What it is |
|---|---|
| [`pacman_dqn.ipynb`](pacman_dqn.ipynb) | The executed notebook for the run described above |
| [`results/comparison.json`](results/comparison.json) | All five before/after evaluation scores and means |
| [`results/config.json`](results/config.json) | Hyperparameters, hardware, and exact package versions |
| [`results/training.csv`](results/training.csv) | One row per training episode |
| [`results/training_summary.json`](results/training_summary.json) | Episodes, decisions, updates, elapsed time, status |
| [`results/baseline.json`](results/baseline.json) | Untrained evaluation detail |
| [`results/training_dashboard.png`](results/training_dashboard.png) | Score, loss, and exploration curves |
| [`results/untrained.gif`](results/untrained.gif) | Baseline gameplay |
| [`results/trained_best.gif`](results/trained_best.gif) | Best trained gameplay of five games |
| [`pacman_player.py`](pacman_player.py) | Optional floating gameplay window for local runs |
| [`requirements.txt`](requirements.txt) | Dependencies for a local run |
| [`tests/verify_notebook.py`](tests/verify_notebook.py) | Optional end-to-end execution check |

### Where the model checkpoints are

`untrained.pt` and `trained.pt` (6.7 MB each) are **not committed**. They are excluded by
`.gitignore` along with the whole `pacman_runs/` folder, and are kept in the run ZIP
`20260910_052510_490439.zip` stored locally. They are needed only to replay a saved agent; every
number and image in this README comes from the JSON, CSV, PNG, and GIF files committed above.
