# Assignment 2 — train an agent to play Ms. Pac-Man

My submission repository for the Ms. Pac-Man DQN assignment.
Starter notebook and materials adapted from [pepealonso95/pacman-dqn](https://github.com/pepealonso95/pacman-dqn).

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/greycatallen/Assignment-2/blob/main/pacman_dqn.ipynb)

**[Open the notebook in Google Colab](https://colab.research.google.com/github/greycatallen/Assignment-2/blob/main/pacman_dqn.ipynb)**

Open **[pacman_dqn.ipynb](pacman_dqn.ipynb)**, choose three numbers, and run all cells.
The complete DQN is in the notebook, split into short code cells with plain-language explanations.
No coding is needed. Keep `pacman_player.py` beside the notebook for the local floating gameplay player.

| Your choice | What it controls | Starting point |
|---|---|---|
| Exploration | Fraction of random training moves after warm-up | `0.20` |
| Episodes | Number of training games | `100` |
| Learning rate | Size of each learning update | `0.0001` |

Try five episodes to check your setup. Useful Atari learning may require much longer runs.
Exploration stays constant after 1,000 random warm-up decisions. The code includes an instruction for Claude Code or Codex to ask the student for these three choices before training.

## Open and run

**Google Colab:** use the Colab button above, select Runtime → Change runtime type → T4 GPU if available, edit the three values in section 1, and choose Runtime → Run all. The setup cell installs packages automatically.

**Local Jupyter or VS Code:** clone or download this repository, open the notebook, select a Python 3.11–3.13 kernel, edit the three choices, and choose Run All. CUDA, Apple Silicon MPS, and CPU are detected automatically; a real training batch checks the selected device before the experiment starts.

**VS Code with `py313`:** select **Select Kernel → Python Environments → py313 (Python 3.13)**.
Use the Python and Jupyter extensions. The local Conda `py313` environment supports Tk for popup playback.
The notebook still runs if Tk is unavailable, but samples appear inline only.

### Fast floating gameplay samples

The before-training sample, every-25-game progress samples, and final sample play at **4× speed**.
A 20-second excerpt takes about five seconds to watch. Each local sample opens automatically in a separate
**always-on-top window**, with Pause, Replay, and a “Keep above other windows” toggle. Press Escape or close
the window to dismiss it. Each sample plays twice and stops on its final frame. Replay starts two more plays.
A new sample replaces the previous popup, and training continues while it plays.

The popup plays a recorded evaluation excerpt once that evaluation finishes. Training and evaluation already
run as fast as the hardware allows, without real-time delays. Faster preview playback does not change the
agent's decisions, learning settings, or full-game scores. The saved GIFs also use accelerated playback.

Set `SHOW_POPUPS = False` in the preview settings for inline playback only. Colab, remote kernels without
a desktop, and Python installations without Tk use the inline GIF. The popup helper is optional, so the
notebook still runs by itself in Colab.

### Read the learning process one piece at a time

Section 3 separates screen preparation, the network, memory, move selection, and a learning update.
Section 4 separates evaluation, playback, checkpoints, logging, and plots. Section 5 separates experiment
setup, baseline evaluation, one training game, progress samples, saving, and the final experiment loop.
Each code cell contains at most 33 lines and has an explanation immediately before it.

If you need to install Jupyter first:

```sh
python -m venv .venv
# macOS / Linux:
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m jupyter lab pacman_dqn.ipynb
```

## What you get

Each experiment saves a separate folder under `pacman_runs/`:

- Settings, hardware, and actual package versions in `config.json`.
- Untrained gameplay and new GIFs every 25 episodes.
- A training dashboard with raw score, loss, and exploration.
- All five untrained and trained evaluation scores in `comparison.json`.
- Untrained, periodic, and final model checkpoints for playback.
- CSV training history, elapsed time, decision count, and number of learning updates.
- A ZIP download with all results.

Use Interrupt / Stop once to end training early and save progress, then run section 6 onward for evaluation and download. The saved model includes any updates from the interrupted episode; the episode log contains completed episodes only. Checkpoints support playback, not exact training resumption. To start again, rerun from section 5a or choose Run All; the training cell guards against accidentally reusing an old experiment.

In Colab, download the ZIP before the session ends. Commit the notebook, selected GIFs, plots, comparison, and your written reflection to your own repository. Generated run folders and checkpoints are ignored by default; copy the selected evidence into a `results/` folder to publish it. Keep large checkpoints locally or attach them to a release.

## How it works

The environment is `ALE/MsPacman-v5`. Four grayscale 84 × 84 frames feed a convolutional Q-network, which predicts action values. The agent uses experience replay, a target network, Adam, Huber loss, and a discounted reward target. Game over removes the future-value term; time limits do not.

Frame skipping happens only in `AtariPreprocessing` (four frames per decision); the base environment uses `frameskip=1`. Reset fills the stack with the new game's initial screen. Sticky-action probability is 0.25, no-op resets use up to 30 actions, and losing one life does not end the episode. Each game is capped at 3,000 decisions, about 200 seconds of game time.

Replay stores each current stack plus one new frame as `uint8`: about 168 MiB at 5,000 transitions, plus Python, network, and batch overhead. Training samples 32 transitions every four decisions after warm-up. The target network syncs every 1,000 decisions; gamma is 0.99. Training rewards are clipped to [-1, 1], while all reported game scores are raw.

Before/after evaluation uses the same five seeds, 5% exploration, and step cap. The baseline is an untrained network. Evaluation uses a separate environment and never updates replay or weights. The best GIF is selected by full-game score, but only its first 20 seconds are recorded. Inspect all five scores before claiming improvement. The small replay memory and fixed exploration simplify the classroom exercise; this is not a benchmark-scale DQN reproduction.

## Submission checklist

- [ ] Pick the three values in section 1 of the notebook (exploration, episodes, learning rate).
- [ ] Run all cells in Colab (Runtime -> Change runtime type -> T4 GPU, then Runtime -> Run all).
- [ ] Download the run ZIP before the Colab session ends.
- [ ] Copy the selected evidence (GIFs, dashboard plot, `comparison.json`) into `results/` and commit it.
- [ ] Fill in the written reflection in the last cell of the notebook, then commit the notebook.

`pacman_runs/` and `*.pt` checkpoints are gitignored, so publish evidence by copying it into `results/`.

## Verification

Executed every revised notebook cell in order through the real `py313` Jupyter kernel on macOS Apple Silicon with MPS, Python 3.13.9, PyTorch 2.10.0, Gymnasium 1.3.0, ALE 0.11.2, OpenCV headless 4.14.0.94, NumPy 2.3.4, Matplotlib 3.10.6, and Pillow 11.3.0.

The verification copy used 20% exploration, **five episodes**, learning rate 0.0001, and samples every two games to exercise the periodic popup/checkpoint path: 3,306 decisions and 577 learning updates. All five before/after evaluation games completed. Checks verified changed and finite model weights, finite losses after warm-up, periodic checkpoints and samples at games 2 and 4, accelerated GIF frame counts and duration, the dashboard, and ZIP contents.

The native Tk player was also tested on the local desktop: window mapping, loaded gameplay image, exactly two plays, window dimensions, always-on-top state, Pause/Play, Replay, and unpin/repin. The no-Tk inline fallback was checked separately. The updated notebook was opened in VS Code with `py313`; the ordered execution test used Jupyter programmatically, not a VS Code Run All click.

To repeat the five-game verification without editing the classroom notebook:

```sh
python tests/verify_notebook.py --kernel py313
# For a kernel without a local desktop:
python tests/verify_notebook.py --kernel py313 --no-popups
```

Run that command with the same Python environment used by the notebook. Its executed test notebook and all artifacts are saved under `pacman_runs/`.

This verifies execution, not strong Pac-Man performance. A full 100-episode run of this revision, CUDA, Windows, and hosted Colab have not been tested here. The distributed notebook has no saved outputs and retains its 100-episode starting value and every-25-game sample interval.

## Sources

- [ALE installation](https://ale.farama.org/getting-started/)
- [Gymnasium Atari preprocessing](https://gymnasium.farama.org/api/wrappers/misc_wrappers/#gymnasium.wrappers.AtariPreprocessing)
- [Gymnasium frame stacking](https://gymnasium.farama.org/api/wrappers/observation_wrappers/#gymnasium.wrappers.FrameStackObservation)
- [DQN paper](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
