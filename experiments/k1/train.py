"""Train two LoRA agents with MAGRPO on OreSplit, resuming from ``--ckpt-dir``."""

import os

# JAX preallocates most of the GPU by default, starving torch on a shared A100.
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.2")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import subprocess  # noqa: E402
from collections.abc import Callable  # noqa: E402
from importlib.metadata import distribution  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from comlrl.trainers.reinforce.magrpo import MAGRPOConfig  # noqa: E402
from peft import LoraConfig  # noqa: E402
from resumable import ResumableMAGRPO, load_agents  # noqa: E402
from reward import make_dataset, make_reward_func  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from factoriax.marl.ore_split import NUM_PLAYERS  # noqa: E402


def _git_sha() -> str:
    repo = Path(__file__).resolve().parent
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True
    ).stdout.strip()


def _comlrl_sha() -> str:
    direct_url = distribution("comlrl").read_text("direct_url.json") or "{}"
    return json.loads(direct_url).get("vcs_info", {}).get("commit_id", "unknown")


def _record_session(ckpt_dir: Path, args: argparse.Namespace, start_step: int) -> None:
    path = ckpt_dir / "config.json"
    config = json.loads(path.read_text()) if path.exists() else {"sessions": []}
    config["sessions"].append(
        {
            "args": {k: str(v) for k, v in vars(args).items()},
            "start_step": start_step,
            "factoriax_sha": _git_sha(),
            "comlrl_sha": _comlrl_sha(),
        }
    )
    path.write_text(json.dumps(config, indent=2))


def main() -> None:
    """Parse args, build or restore the trainer, and train for one session."""
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--ckpt-dir", type=Path, required=True)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--max-steps-this-session", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--dataset-rows", type=int, default=400)
    p.add_argument("--wandb-project", default=None)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    wandb_config = None
    if args.wandb_project:
        # CoMLRL's wandb.init takes no run id, so pin it via env to resume one run across sessions.
        ckpt_key = str(args.ckpt_dir.resolve()).encode()
        os.environ["WANDB_RUN_ID"] = hashlib.sha1(ckpt_key).hexdigest()[:12]
        os.environ["WANDB_RESUME"] = "allow"
        wandb_config = {
            "project": args.wandb_project,
            "name": f"{args.ckpt_dir.parent.name}/{args.ckpt_dir.name}",
        }

    tok = AutoTokenizer.from_pretrained(args.model)
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=2 * args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )

    def chat_formatter(i: int) -> Callable[[dict[str, Any]], str]:
        def fmt(item: dict[str, Any]) -> str:
            return tok.apply_chat_template(
                [{"role": "user", "content": item[f"prompt_{i}"]}],
                tokenize=False,
                add_generation_prompt=True,
            )

        return fmt

    trainer = ResumableMAGRPO(
        agents=load_agents(args.model, num_agents=NUM_PLAYERS, lora_cfg=lora),
        tokenizer=tok,
        train_dataset=make_dataset(args.dataset_rows),
        reward_func=make_reward_func(
            args.ckpt_dir / "episodes.jsonl", step_fn=lambda: trainer.global_step
        ),
        formatters=[chat_formatter(i) for i in range(NUM_PLAYERS)],
        args=MAGRPOConfig(
            num_turns=1,
            num_generations=8,
            max_new_tokens=64,
            # Large on purpose: --max-steps-this-session is what ends a run.
            num_train_epochs=1000,
            eval_interval=0,
            agent_learning_rate=args.lr,
            logging_steps=1,
        ),
        wandb_config=wandb_config,
        ckpt_dir=str(args.ckpt_dir),
        save_every=args.save_every,
        max_steps_this_session=args.max_steps_this_session,
    )
    trainer.restore()
    _record_session(args.ckpt_dir, args, trainer.global_step)
    trainer.train()
    print(f"global_step={trainer.global_step}")


if __name__ == "__main__":
    main()
