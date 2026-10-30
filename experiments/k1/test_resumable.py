import shutil

import pytest

pytest.importorskip("comlrl")

import torch  # noqa: E402
from comlrl.trainers.reinforce.magrpo import MAGRPOConfig  # noqa: E402
from datasets import Dataset  # noqa: E402
from peft import LoraConfig  # noqa: E402
from resumable import ResumableMAGRPO, load_agents  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

TINY = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"


def length_reward(c0, c1, batch_items=None):
    return [len(a) / 10 for a in c0]


def build_tiny_trainer(tmp_path, max_steps_this_session):
    torch.manual_seed(0)
    lora = LoraConfig(r=4, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
    rows = [{"variant": v, "prompt_0": f"a{v}", "prompt_1": f"b{v}"} for v in range(4)]
    return ResumableMAGRPO(
        agents=load_agents(TINY, num_agents=2, lora_cfg=lora),
        tokenizer=AutoTokenizer.from_pretrained(TINY),
        train_dataset=Dataset.from_list(rows),
        reward_func=length_reward,
        formatters=[lambda x: x["prompt_0"], lambda x: x["prompt_1"]],
        args=MAGRPOConfig(
            num_train_epochs=10,
            num_generations=2,
            max_new_tokens=8,
            num_turns=1,
            eval_interval=0,
        ),
        ckpt_dir=str(tmp_path),
        save_every=1,
        max_steps_this_session=max_steps_this_session,
    )


def lora_params(model):
    return {k: v.detach().clone() for k, v in model.named_parameters() if "lora" in k}


def test_stop_and_resume(tmp_path):
    t1 = build_tiny_trainer(tmp_path, max_steps_this_session=2)
    t1.train()
    assert t1.global_step == 2
    saved = lora_params(t1.agents[0])

    t2 = build_tiny_trainer(tmp_path, max_steps_this_session=2)
    fresh = lora_params(t2.agents[0])
    assert any(not torch.equal(fresh[k], saved[k]) for k in saved)
    t2.restore()
    assert t2.global_step == 2
    restored = lora_params(t2.agents[0])
    assert saved.keys() == restored.keys()
    for k, v in restored.items():
        assert torch.equal(v, saved[k])
    assert t2.optimizers[0].state_dict()["state"]
    t2.train()
    assert t2.global_step == 4


def test_partial_save_does_not_corrupt_latest(tmp_path):
    t1 = build_tiny_trainer(tmp_path, max_steps_this_session=2)
    t1.train()
    (tmp_path / "tmp").mkdir()
    (tmp_path / "tmp" / "state.pt").write_bytes(b"half-written")

    t2 = build_tiny_trainer(tmp_path, max_steps_this_session=2)
    t2.restore()
    assert t2.global_step == 2


def test_crash_between_swaps_falls_back_to_old(tmp_path):
    t1 = build_tiny_trainer(tmp_path, max_steps_this_session=2)
    t1.train()
    shutil.rmtree(tmp_path / "latest.old")
    (tmp_path / "latest").rename(tmp_path / "latest.old")

    t2 = build_tiny_trainer(tmp_path, max_steps_this_session=2)
    t2.restore()
    assert t2.global_step == 2


def test_restore_without_checkpoint_starts_fresh(tmp_path):
    t = build_tiny_trainer(tmp_path, max_steps_this_session=1)
    t.restore()
    assert t.global_step == 0
