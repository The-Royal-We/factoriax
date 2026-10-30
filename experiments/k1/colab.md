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
   ```

3. Check JAX sees the GPU

   ```python
   import jax

   print(jax.devices())  # expect [CudaDevice(id=0)]
   ```

4. Train (rerun this cell after a disconnect, and it resumes from `latest/`)

   ```bash
   !cd factoriax && python experiments/k1/train.py \
       --model Qwen/Qwen2.5-1.5B-Instruct \
       --ckpt-dir /content/drive/MyDrive/k1/seed0 \
       --save-every 10 --max-steps-this-session 10 --seed 0
   ```

## Measurements

### Task 5: `score_plans` benchmark (CPU, M-series Mac, JAX 0.10.2)

Median of 20 calls after warm-up: **0.064 s**. That is well under the 0.3 s
gate, so JAX on CPU is fine.

Every pallet saturates at 256 ore well before tick 400. So the oracle scores
1024 on every variant, and any 4/4 split scores the same 1024, whichever agent
walks to the far patch. The reward sees how many sites get built, not how fast.

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

Not run yet. Record here: seconds/step, peak GPU memory, checkpoint size, and
whether `global_step` reads 20 after a kill-and-resume at step 10.
