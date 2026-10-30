"""MAGRPO that checkpoints LoRA adapters + optimiser + RNG and resumes across sessions.

Relies on private MAGRPOTrainer methods (``_train_step_returns``,
``_drain_ready_buffers``), which is why CoMLRL is pinned in requirements.txt.
"""

import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from comlrl.trainers.reinforce.magrpo import MAGRPOTrainer
from peft import LoraConfig, PeftModel, get_peft_model, set_peft_model_state_dict
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM


def load_agents(
    base_name: str, num_agents: int, lora_cfg: LoraConfig
) -> list[PeftModel]:
    """Build fresh LoRA agents; ``ResumableMAGRPO.restore`` loads saved adapters."""
    return [
        get_peft_model(AutoModelForCausalLM.from_pretrained(base_name), lora_cfg)
        for _ in range(num_agents)
    ]


def _rng_state() -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


class ResumableMAGRPO(MAGRPOTrainer):
    """One step is one dataset row, visited in order, so resume skips ahead exactly.

    Each save drains the rollout buffers. A session cap that is not a multiple of
    ``save_every`` therefore applies one partial update that an uninterrupted
    run would not.
    """

    def __init__(
        self,
        *args: Any,
        ckpt_dir: str,
        save_every: int = 10,
        max_steps_this_session: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.ckpt_dir = Path(ckpt_dir)
        self.save_every = save_every
        self.max_steps_this_session = max_steps_this_session
        self.global_step = 0

    def save_checkpoint(self) -> None:
        """Drain pending updates, write to ``tmp/``, then swap it in as ``latest/``."""
        self._drain_ready_buffers([i for i, b in enumerate(self.rollout_buffers) if b])
        tmp, latest, old = (self.ckpt_dir / n for n in ("tmp", "latest", "latest.old"))
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        for i, agent in enumerate(self.agents):
            agent.save_pretrained(tmp / f"agent_{i}")
        torch.save(
            {
                "global_step": self.global_step,
                "env_step": self.env_step,
                "optimizers": [o.state_dict() for o in self.optimizers],
                "rng": _rng_state(),
            },
            tmp / "state.pt",
        )
        if latest.exists():
            shutil.rmtree(old, ignore_errors=True)
            latest.rename(old)
        tmp.rename(latest)

    def restore(self) -> bool:
        """Load ``latest/``, else ``latest.old/``; return False if neither exists."""
        for name in ("latest", "latest.old"):
            src = self.ckpt_dir / name
            if (src / "state.pt").exists():
                break
        else:
            return False
        for i, agent in enumerate(self.agents):
            weights = load_file(src / f"agent_{i}" / "adapter_model.safetensors")
            set_peft_model_state_dict(agent, weights)
        # Our own file, and it holds numpy RNG state, which weights_only rejects.
        state = torch.load(src / "state.pt", weights_only=False)
        for opt, opt_state in zip(self.optimizers, state["optimizers"], strict=True):
            opt.load_state_dict(opt_state)
        _set_rng_state(state["rng"])
        self.global_step = state["global_step"]
        self.env_step = state["env_step"]
        return True

    def train(self, **kwargs: Any) -> None:
        """Train from ``global_step`` until the session cap or the last epoch."""
        for i, agent in enumerate(self.agents):
            agent.to(self.agent_devices[i])
            agent.train()
        n_rows = len(self.train_dataset)
        end = int(self.args.num_train_epochs) * n_rows
        if self.max_steps_this_session is not None:
            end = min(end, self.global_step + self.max_steps_this_session)
        turn_rewards: list[list[float]] = [[] for _ in range(self.args.num_turns)]
        turn_returns: list[list[float]] = [[] for _ in range(self.args.num_turns)]
        while self.global_step < end:
            item = self.train_dataset[self.global_step % n_rows]
            self._train_step_returns(item, turn_rewards, turn_returns, **kwargs)
            self.global_step += 1
            if self.global_step % self.save_every == 0:
                self.save_checkpoint()
        if self.global_step % self.save_every:
            self.save_checkpoint()
