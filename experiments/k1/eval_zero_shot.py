"""Zero-shot baseline: sample both agents' plans from a base model and score them."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from metrics import choice, entropy, nmi
from reward import NUM_VARIANTS
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from factoriax.marl.episode import score_plans
from factoriax.marl.ore_split import NUM_PLAYERS
from factoriax.marl.text import render_prompt


def sample(
    model: PreTrainedModel,
    tok: PreTrainedTokenizerBase,
    prompt: str,
    n: int,
    args: argparse.Namespace,
) -> list[str]:
    """Draw ``n`` chat-templated completions for one prompt."""
    enc = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)
    out = model.generate(
        **enc,
        do_sample=True,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        num_return_sequences=n,
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
    )
    prompt_len = enc["input_ids"].shape[1]
    return tok.batch_decode(out[:, prompt_len:], skip_special_tokens=True)


def main() -> None:
    """Run the zero-shot evaluation and write a JSON summary."""
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--samples", type=int, default=32, help="episodes in total")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device).eval()

    per_variant = {}
    all_ok, all_norm, agent_ids, choices = [], [], [], []
    for v in range(NUM_VARIANTS):
        n = args.samples // NUM_VARIANTS
        texts = [
            sample(model, tok, render_prompt(i, v), n, args) for i in range(NUM_PLAYERS)
        ]
        results = [score_plans(pair, v) for pair in zip(*texts, strict=True)]
        ok = [o for r in results for o in r.parsed_ok]
        norm = [r.normalised for r in results]
        per_variant[v] = {
            "parse_rate": float(np.mean(ok)),
            "mean_normalised": float(np.mean(norm)),
            "examples": [list(pair) for pair in zip(*texts, strict=True)][:3],
        }
        all_ok += ok
        all_norm += norm
        for r in results:
            for i, plan in enumerate(r.plans):
                agent_ids.append(i)
                choices.append(choice(plan))

    summary = {
        "model": args.model,
        "parse_rate": float(np.mean(all_ok)),
        "mean_normalised": float(np.mean(all_norm)),
        "nmi": nmi(agent_ids, choices),
        "choice_entropy": entropy(choices),
        "per_variant": per_variant,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_variant"}))


if __name__ == "__main__":
    main()
