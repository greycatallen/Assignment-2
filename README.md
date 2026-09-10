# Assignment 2 — Training a Ms. Pac-Man Agent with a DQN

Class 3 assignment. A Deep Q-Network is trained on `ALE/MsPacman-v5` with the supplied notebook,
then evaluated against its own untrained starting point under identical settings.

Starter notebook adapted from [pepealonso95/pacman-dqn](https://github.com/pepealonso95/pacman-dqn).

> **Headline result: training made the agent worse.** Across the five fixed evaluation seeds the
> mean score fell from **492 to 408**, with four of the five games regressing. The run completed
> normally — 100 episodes, 13,488 learning updates, no crash and no `NaN` — so this is a real
> negative result caused by the learning rate I chose, not a broken run. The reasoning is in
> [What I observed](#what-i-observed).

## Open and run

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/greycatallen/Assignment-2/blob/main/pacman_dqn.ipynb)

- **Colab:** click the badge, choose Runtime → Change runtime type → T4 GPU, then Runtime → Run all.
  The first cell installs every package. My three values are already set in section 1.
- **Local Jupyter / VS Code:** clone this repo, select a Python 3.11–3.13 kernel, `pip install -r
  requirements.txt`, and Run All. Keep `pacman_player.py` beside the notebook for the local
  floating gameplay window.

The notebook detects CUDA, Apple Silicon MPS, or CPU automatically.

## My three hyperparameters

| Setting | Value | Why I chose it |
|---|---|---|
| Exploration | `0.20` | Keeps one training move in five random after warm-up. Ms. Pac-Man rewards nothing for standing still, so the agent needs steady forced variety to keep meeting pellets and ghosts it would not choose on its own. I held this at the notebook's reference value so the learning rate was the only thing I changed from my first run. |
| Episodes | `100` | My earlier five-episode run produced only 388 learning updates, far too few to conclude anything. 100 episodes gives roughly 55,000 decisions and about 13,500 updates — a thirty-fold increase, and enough for a genuine before/after comparison. |
| Learning rate | `0.01` | A deliberate 100× departure from the 0.0001 reference. I wanted to find out how large a step DQN tolerates rather than accept the default on trust, and to see the failure mode directly if it was too large. |

Every other setting is the notebook's fixed classroom default. The evaluation settings in
particular are untouched: the same five seeds (101, 202, 303, 404, 505), 5% exploration, and the
same 3,000-decision cap, applied identically before and after training.

## What I expected

I predicted the large learning rate would produce **significant improvement early on** — a mean
score rising sharply in the first stretch of training — and then a **long plateau**, on the
reasoning that big steps make fast initial progress but then overshoot and bounce around a good
solution instead of settling into it. I expected the plateau to be the visible cost of the 100×
step size.

## What I observed

Half of that held. The plateau arrived; the improvement did not.

### All five before/after evaluation scores

Same five seeds, 5% exploration, same step cap, untrained network as the baseline.
Full data: [`results/comparison.json`](results/comparison.json).

| Seed | Untrained | Trained | Change |
|---:|---:|---:|---:|
| 101 | 350 | 250 | −100 |
| 202 | 500 | 440 | −60 |
| 303 | 320 | **840** | +520 |
| 404 | 800 | 440 | −360 |
| 505 | 490 | **70** | −420 |
| **Mean** | **492.0** | **408.0** | **−84.0** |
| Median | 490 | 440 | −50 |

Four of the five games got worse, and seed 505 collapsed to 70 points — barely more than the
opening pellets. Only seed 303 improved. Both the mean and the median fell, so unlike my earlier
run this is not a case of one outlier disguising the picture: the result is consistently negative.

### Training plot

![Training dashboard](results/training_dashboard.png)

The middle panel is the important one. **Mean update loss hits 16.09 at episode 2** — roughly 600×
the ~0.026 my 0.0001 run showed — then crashes to about 0.5 by episode 5 and sits near 0.2 for the
remaining 95 episodes. It never becomes `NaN`. So the network did not blow up; it took one very
large, destabilising jump early and then converged smoothly onto a policy that plays worse than
the random-weight network it started from.

The left panel is where my "rose then plateaued" prediction appeared to come true, and it is worth
being precise about why that reading is misleading:

| Episode | 25-game average training score |
|---:|---:|
| 1 | 130.0 |
| 5 | 602.0 |
| 25 | 460.8 |
| 50 | 455.2 |
| 75 | 582.8 |
| 100 | 524.0 |

The jump from 130 to 602 in the first five episodes looks like rapid early learning, but it is an
artefact of the warm-up schedule. Episode 1 runs at exploration 1.0 — every move random — and
warm-up ends after 1,000 decisions, at which point exploration drops to my chosen 0.20. The score
rises simply because the agent stops playing randomly, before any meaningful learning has occurred.
After that the curve is flat noise between 455 and 583 for 95 episodes, with no trend. **The
plateau is real; the improvement preceding it is not.**

### Gameplay

Every clip is the first 20 seconds of a game at 4× speed, playing twice before stopping. The
trained clip is the best of the five evaluation games, chosen on full-game score.

| Untrained (baseline) | Trained, best of five |
|---|---|
| ![Untrained](results/untrained.gif) | ![Trained](results/trained_best.gif) |

Intermediate samples captured every 25 episodes ([`results/demo_scores.json`](results/demo_scores.json)):

| Episode 25 | Episode 50 | Episode 75 | Episode 100 |
|---|---|---|---|
| ![ep25](results/episode_0025.gif) | ![ep50](results/episode_0050.gif) | ![ep75](results/episode_0075.gif) | ![ep100](results/episode_0100.gif) |
| 380 | 350 | 200 | 250 |

Those four demonstration games all use seed 101, whose untrained baseline was 350. The sequence
380 → 350 → 200 → 250 tracks *downward* across training and matches the final evaluation score of
250 on that seed. Watching the clips, the later agent moves in shorter, more repetitive paths and
gets cornered sooner than the untrained one. It has learned something — it simply learned a bad
policy, and it learned it consistently.

### Actual training budget

From [`results/training_summary.json`](results/training_summary.json) and [`results/config.json`](results/config.json):

| | |
|---|---|
| Status | `completed` — not interrupted |
| Episodes completed | 100 of 100 requested |
| Total decisions | 54,949 |
| Learning updates | **13,488** |
| Elapsed training time | 366.4 s (6.1 min), including periodic sample capture |
| Hardware | CPU (Colab, Linux x86-64), PyTorch 2.9.0+cpu, Python 3.13.15 |

The run reported CPU rather than GPU. It made little practical difference here — the Atari
emulator is single-threaded CPU work that dominates the loop, and an identical earlier
configuration finished in comparable time on a T4 — but the hardware is recorded honestly as CPU.
Per-episode data is in [`results/training.csv`](results/training.csv).

## How the agent works, in plain language

- **Observations.** The agent never sees the game's memory or a list of objects. It sees
  **four consecutive game screens**, shrunk to 84 × 84 grayscale images and stacked. Four frames
  rather than one is what makes motion visible: a single still image cannot tell you whether a
  ghost is approaching or retreating.
- **Actions.** It picks one of the **nine joystick moves** — eight directions plus no-op — and
  holds it for four emulator frames, so one decision covers about a fifteenth of a second.
- **Rewards.** The reward is the **game points** earned in that interval: pellets, power pellets,
  edible ghosts, fruit. During training these are clipped to [−1, 1] so no single event dominates
  an update, but every score in this README is the raw, unclipped game score.
- **Learning.** Each experience goes into a replay memory. Every four decisions the agent samples
  32 past experiences at random and nudges its value estimates toward the reward received plus the
  discounted value of the next state, judged by a slowly-updated target network. Random sampling
  breaks the correlation between consecutive frames; the separate target network stops the agent
  chasing its own moving estimate.

## One limitation

**The learning rate was large enough to make the agent worse, and a single run cannot separate a
bad step size from bad luck.** The 16.09 loss spike at episode 2 shows the first updates moved the
weights very far very fast. With Adam, a moving target network, and rewards clipped to [−1, 1],
a step that size lets early, barely-informed estimates dominate the network before the replay
memory holds anything worth learning from — and the agent then spends 95 episodes converging
neatly onto that damaged starting point. Low, stable loss is exactly what that looks like from the
inside, which is why the loss curve alone would have suggested a healthy run.

The honest limit on my conclusion is that I ran this configuration once, with one seed. Five
evaluation games with scores ranging 70–840 carry enough variance that I can say this run was
worse, but not precisely how much of the −84 is the learning rate and how much is noise.

## My next experiment

**Change only the learning rate, from `0.01` to `0.002`.** Exploration stays at 0.20 and the
budget stays at 100 episodes.

That is the right single variable because the evidence points at it directly and exonerates the
other two. The episode budget is no longer the constraint — 13,488 updates is thirty times my
earlier run and the curve had clearly stopped moving, so more episodes would buy nothing. The
exploration setting behaved exactly as intended, holding flat at 0.20 for the whole run. What
changed between a run that merely failed to learn and a run that actively degraded was the step
size, and the loss panel shows the damage happening in the first two episodes.

I chose 0.002 rather than returning to 0.0001 because it keeps the experiment informative: it is
five times the reference and still a deliberate departure from the default, but a fifth of the
value that broke this run. If the early loss spike disappears and the evaluation mean lands above
492, the step size was the whole story. If the mean still falls, the problem is elsewhere and I
would look at the replay memory size next.

## Repository contents

| Path | What it is |
|---|---|
| [`pacman_dqn.ipynb`](pacman_dqn.ipynb) | The executed notebook for the run above |
| [`results/comparison.json`](results/comparison.json) | All five before/after scores and means |
| [`results/config.json`](results/config.json) | Hyperparameters, hardware, exact package versions |
| [`results/training.csv`](results/training.csv) | One row per training episode |
| [`results/training_summary.json`](results/training_summary.json) | Episodes, decisions, updates, elapsed time, status |
| [`results/baseline.json`](results/baseline.json) | Untrained evaluation detail |
| [`results/demo_scores.json`](results/demo_scores.json) | Scores for the four intermediate samples |
| [`results/training_dashboard.png`](results/training_dashboard.png) | Score, loss, and exploration curves |
| `results/untrained.gif`, `results/trained_best.gif` | Baseline and best trained gameplay |
| `results/episode_0025.gif` … `episode_0100.gif` | Intermediate gameplay every 25 episodes |
| [`results/setup_check_5ep/`](results/setup_check_5ep) | My earlier five-episode run, kept for comparison |
| [`pacman_player.py`](pacman_player.py) | Optional floating gameplay window for local runs |
| [`requirements.txt`](requirements.txt) | Dependencies for a local run |

### Earlier run, for comparison

Before this experiment I ran the setup check the assignment suggests: **5 episodes, learning rate
0.0001**, everything else identical. It produced 388 learning updates and a mean of 492 → 730 —
but that apparent gain came entirely from one seed while the median fell, so it showed no reliable
improvement either. Its evidence is preserved under
[`results/setup_check_5ep/`](results/setup_check_5ep). Taken together the two runs bracket the
problem: 0.0001 with too few updates learned nothing, and 0.01 with enough updates learned the
wrong thing.

### Where the model checkpoints are

The playback checkpoints — `untrained.pt`, `trained.pt`, and `episode_0025/0050/0075/0100.pt`,
6.7 MB each — are **not committed**. They are excluded by `.gitignore` along with the whole
`pacman_runs/` folder, and are kept in the run ZIP `20260910_055110_655234.zip` stored locally.
They are needed only to replay a saved agent; every number and image in this README comes from the
JSON, CSV, PNG, and GIF files committed above.
