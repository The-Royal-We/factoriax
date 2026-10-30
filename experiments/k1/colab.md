# K=1 OreSplit on Colab

## Cells

1. Mount Drive

   ```python
   from google.colab import drive

   drive.mount("/content/drive")
   ```

2. Install

   ```bash
   !git clone https://github.com/The-Royal-We/factoriax.git && cd factoriax && git checkout k1-ore-split
   !pip install "./factoriax[cuda]" -r factoriax/experiments/k1/requirements.txt
   !pip uninstall -y torchao
   ```

   Colab preinstalls torchao 0.10. That version is too old for current peft, and peft raises an error at LoRA setup if it finds it. Nothing in this repo uses torchao.

3. Check JAX sees the GPU

   ```python
   import jax

   print(jax.devices())  # expect [CudaDevice(id=0)]
   ```

4. Pilot (Task 10 Steps 1–2; done). Rerun this cell after a disconnect, and it resumes from `latest/`.

   ```bash
   !cd factoriax && python experiments/k1/train.py \
       --model Qwen/Qwen2.5-1.5B-Instruct \
       --ckpt-dir /content/drive/MyDrive/k1/seed0 \
       --save-every 10 --max-steps-this-session 10 --seed 0
   ```

5. Phase 1 gate: 3 seeds × 150 steps (about 40 min on an A100). Each seed
   tops up to 150 steps from its own checkpoint, so after a disconnect you just
   rerun the cell. `k1/seed0` is the pilot on the old geometry, so don't reuse it.

   ```python
   from pathlib import Path

   import torch

   ROOT = Path("/content/drive/MyDrive/k1/w48h100")
   STEPS = 150


   def steps_done(ckpt: Path) -> int:
       for name in ("latest", "latest.old"):
           if (ckpt / name / "state.pt").exists():
               return torch.load(ckpt / name / "state.pt", weights_only=False)["global_step"]
       return 0


   for seed in range(3):
       ckpt = ROOT / f"seed{seed}"
       remaining = STEPS - steps_done(ckpt)
       if remaining > 0:
           !cd factoriax && python experiments/k1/train.py \
               --model Qwen/Qwen2.5-1.5B-Instruct --ckpt-dir {ckpt} \
               --save-every 10 --max-steps-this-session {remaining} --seed {seed}
   ```

6. Gate check. Learning signal: every seed's last-30-step mean normalised score
   is above 0.198 (zero-shot). Coordination: every seed's score is also above
   0.74 (the fixed rule), and the near-patch rate is near 1. The fixed rule gives
   a near-patch rate of 0.5, because it is right only on the variants where agent
   0 spawns next to iron.

   ```python
   import json
   import sys

   import matplotlib.pyplot as plt
   import numpy as np

   sys.path.insert(0, "factoriax/experiments/k1")
   from metrics import choice, last_per_step, near_patch_rate, nmi

   from factoriax.marl.macros import Macro

   EPISODES_PER_STEP = 8
   BIN = 10
   SEED_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a"]


   def load(seed):
       lines = (ROOT / f"seed{seed}" / "episodes.jsonl").read_text().splitlines()
       return last_per_step([json.loads(line) for line in lines], EPISODES_PER_STEP)


   def nmi_of(rows):
       picks = [choice([Macro(*m) for m in plan]) for r in rows for plan in r["plans"]]
       return nmi([0, 1] * len(rows), picks)


   panels = {
       "normalised score": (lambda rows: np.mean([r["normalised"] for r in rows]),
                            {"zero-shot": 0.198, "fixed rule": 0.74}),
       "NMI (agent × patch)": (nmi_of, {}),
       "near-patch rate": (near_patch_rate, {"fixed rule": 0.5}),
   }
   fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True)
   final = {}
   for seed, colour in zip(range(3), SEED_COLOURS):
       rows = load(seed)
       bins = [[r for r in rows if r["global_step"] // BIN == b]
               for b in range(STEPS // BIN)]
       bins = [b for b in bins if b]
       x = [b[0]["global_step"] // BIN * BIN for b in bins]
       for ax, (title, (fn, _)) in zip(axes, panels.items()):
           ax.plot(x, [fn(b) for b in bins], color=colour, lw=2, label=f"seed {seed}")
       tail = [r for r in rows if r["global_step"] >= STEPS - 30]
       final[seed] = (np.mean([r["normalised"] for r in tail]), near_patch_rate(tail))
   for ax, (title, (_, refs)) in zip(axes, panels.items()):
       for name, y in refs.items():
           ax.axhline(y, color="#8a8a85", lw=1, ls="--")
           ax.annotate(name, (0, y), xytext=(2, 3), textcoords="offset points",
                       color="#5f5f5a", fontsize=9)
       ax.set_ylim(0, 1.05)
       ax.set_title(title, loc="left")
       ax.set_xlabel("step")
       ax.spines[["top", "right"]].set_visible(False)
       ax.grid(axis="y", color="#e6e6e3", lw=0.8)
   axes[0].legend(frameon=False)
   plt.tight_layout()
   plt.show()

   for seed, (score, near) in final.items():
       print(f"seed {seed}: last-30 normalised {score:.3f}, near-patch {near:.2f}")
   worst = min(score for score, _ in final.values())
   print("learning signal:", "PASS" if worst > 0.198 else "FAIL")
   print("coordination:   ", "PASS" if worst > 0.74 else "FAIL",
         "(check near-patch rate too)")
   ```

## Measurements

### Task 5: `score_plans` benchmark (CPU, M-series Mac, JAX 0.10.2)

Median of 20 calls after warm-up: **0.064 s**. That is well under the 0.3 s
gate, so JAX on CPU is fine.

Every pallet saturates at 256 ore well before tick 400. So the oracle scores
1024 on every variant, and any 4/4 split scores the same 1024, whichever agent
walks to the far patch. The reward sees how many sites get built, not how fast.

### Horizon/geometry sweep (fixes the saturation above)

The pallets fill by t=120 at every map width. On the original 16-wide map, the
swapped split trails the oracle by only 96 ore (about 8 ticks of walking), so a
fixed "agent 0 iron, agent 1 copper" rule scored 1.0 at H=400. Ratios to the
oracle on variant 0:

| width | H | swapped split | fixed rule (mean over variants) | hedge 2/2 |
|---|---|---|---|---|
| 16 | 400 | 1.00 | 1.00 | 0.50 |
| 16 | 100 | 0.90 | 0.95 | 0.42 |
| 32 | 100 | 0.69 | 0.85 | 0.32 |
| **48** | **100** | **0.48** | **0.74** | **0.21** |

Now using width 48 (patches at x 2–5 and 42–45, spawns at x 3 and 44) and
`HORIZON = 100`. The oracle scores 913 on every variant, below saturation, so
the reward now depends on reading the spawn positions in the prompt.
`score_plans` median: **0.019 s** (was 0.064 s). The Task 9 table below was
measured on the old geometry. Rerun on the new one (same settings) in
`results/zero_shot_Qwen2.5-1.5B-Instruct_w48h100.json`:

| Model | parse rate | mean normalised | NMI | choice entropy (bits) |
|---|---|---|---|---|
| Qwen2.5-1.5B-Instruct (w48, H100) | 1.00 | **0.198** | 0.044 | 0.76 |

The model still hedges, so this is the Phase 1 baseline. The fixed rule (0.74)
and the oracle (1.0) are both well above it.

### Task 8: CoMLRL + PEFT spike (CPU)

Passed. `MAGRPOTrainer(agents=[peft0, peft1])` ran one step with
`trl-internal-testing/tiny-Qwen2ForCausalLM-2.5`, and the optimiser held state
for the LoRA tensors only. The plan's `hf-internal-testing/tiny-random-Qwen2ForCausalLM`
does not exist on the Hub (it returns 401). The tiny-model checkpoint (2 LoRA
r=4 adapters + optimiser + RNG) is 112 KB.

CoMLRL dispatches the reward function by counting its positional parameters,
so `batch_items` must be keyword-only (see `reward.py`).

### Task 9: zero-shot gate

Settings: CPU Mac, chat template, temperature 0.7, 32 episodes (8 per variant),
`max_new_tokens=64`. Full JSON is in `results/zero_shot_<model>.json`.

| Model | parse rate | mean normalised (ICL-none baseline) | NMI | choice entropy (bits) |
|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | 0.77 (fails the 0.8 gate) | 0.195 | 0.054 | 1.55 |
| Qwen2.5-1.5B-Instruct | 1.00 | 0.258 | 0.007 | 0.63 |

The 1.5B model nearly always hedges: `BUILD iron 2 / BUILD copper 2` or
`BUILD iron 4 / BUILD copper 4`. Both agents do it, so they collide, and NMI is
about 0. That leaves plenty of headroom for MAGRPO. 0.5B fails mostly on prose
answers and over-long lists, so the prompt was not changed. Changing it would
also move the 1.5B baseline.

### Task 10: A100 pilot

Seconds/step, from the `t` timestamps in `episodes.jsonl` (steps 0–9, 8
episodes per step, Qwen2.5-1.5B-Instruct):

| | value |
|---|---|
| step-to-step gap | alternates 2.1–2.3 s and 7.2–7.4 s |
| mean seconds/step | **~4.7 s** (9.4 s per pair of steps) |
| reward scoring per step | 0.77 s (about 0.11 s per episode, run one after another) |
| first to last episode, 10 steps | 40.7 s |

Every other step takes about 5 s longer. Reward scoring stays the same on every
step, so the extra time is spent in the trainer.

Peak GPU memory: about 15 GB. Checkpoint size: about 67 MB. Kill-and-resume
works: after a kill at step 10, the resumed run reached `global_step` 20.
