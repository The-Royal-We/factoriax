# Research Plan: How LLM Agent Teams Organise to Build Production Systems

**Status:** Draft, 2026-10-28. Needs supervisor review before Phase 2.
**Replaces in part:** `thesis/factorio-learning-environment/project_proposal_v2.md` (Feb 2026) and the RQs in `thesis/research_report/literature-review-draft.md` (May 2026).
**First implementation step:** `plans/2026-10-28-k1-ore-split.md`.

## Motivation: von Neumann probes as the long-run target

A von Neumann probe is a team of autonomous agents that arrives somewhere with raw resources and no human help. It builds an industry there, including more copies of its own builders. That breaks down into four problems:

| # | Problem | Question | In this thesis |
|---|---|---|---|
| 1 | Design | How is a goal turned into a production system? | Partly. Agents choose what to build. |
| 2 | **Organisation** | How is the work split and coordinated across agents? | **Core** |
| 3 | **Scaling** | Does adding agents increase output, or does coordination overhead eat the gain? | **Core** |
| 4 | Closure | Can the system produce its own builders? | Stretch goal |

Factorio stands in for all four: going from ore to rocket bootstraps an industry from nothing. Factoriax provides that world as a deterministic JAX simulator. It's fast enough that the environment costs almost nothing, so the compute budget goes to the LLMs.

## Research questions

- **RQ1 — Organisation.** On the same production tasks, how do organisational structures compare on team throughput, specialisation and robustness? The structures are flat/peer, hierarchical (planner + workers), market (task auction) and stigmergic (no messages; agents read the shared factory).
- **RQ2 — Scaling.** How does each structure's throughput per agent change from 2 → 4 → 8 agents? Where does coordination overhead outweigh the gain from extra workers?
- **RQ3 — Learning vs design.** Does reward-driven learning (MAGRPO, rejection-sampling fine-tuning) improve on the best *designed* structure? Is the gain new behaviour (Discovery) or a sharper version of what the frozen model already does (Sharpening)? This keeps the interpretive frame from proposal v2, applied to organisation.
- **RQ4 — Robustness.** Does the organisation survive agent-index permutation and the removal of an agent mid-episode? (This was RQ5 in proposal v2.)
- **Stretch RQ5 — Closure.** If agents can spend resources on new builders, do they choose more builders or more output, and does that choice depend on the structure?

## Method

**Environment.** Factoriax plus a macro-action layer: LLMs output text plans, scripted controllers execute them, and rollouts are jitted. There's one decision point per episode at first (K=1), then periodic re-planning (K>1, using CoMLRL's `external_transition`).

**Task ladder.** Each rung adds one kind of interdependence:

1. **OreSplit:** two resources and two agents, scored as `min(iron, copper)`. Division of labour is required. This is the K=1 plan.
2. **TriSplit:** three resources, 2–4 agents, with unequal patch sizes. Here the optimal allocation isn't one agent per patch.
3. **SmeltChain:** ore → furnace → plate, so one agent's output is another's input (sequential dependency).
4. **EasyRocket-lite:** a cut-down version of the existing `EasyRocket-v1` with a macro vocabulary. This is the bootstrapping task.

**Model.** Qwen2.5-1.5B-Instruct is the main model, with 0.5B as a scale check. The same base model is used in every condition.

**Conditions.**

| Condition | Role |
|---|---|
| Random plans | Floor |
| Oracle plans (hand-written optimum) | Ceiling; the score normaliser |
| IPPO from scratch on primitive actions | The pure-discovery, no-prior reference (cheap in JAX) |
| Frozen LLM × {flat, hierarchical, market, stigmergic} | RQ1/RQ2. Inference only. |
| Frozen LLM + episode summaries (ICL) | Context only, with no weight updates |
| Rejection-sampling fine-tuning on the best structure | Cheap reward-driven learning |
| MAGRPO on the best structure | Full multi-agent RL |

**Metrics.**
- Team score / oracle score
- NMI (agent × choice)
- Per-agent entropy
- JS divergence between agents over training
- Interference (failed placements, duplicated sites)
- Scaling efficiency (score per agent vs N)
- Communication tokens per episode
- Seconds and dollars per condition

**Statistics.** Use ≥3 seeds per learning condition and ≥32 episodes per frozen condition. Report bootstrapped 95% CIs, use Kruskal-Wallis + pairwise tests with Holm correction, and give effect sizes (carried over from proposal v2 §3.7).

## Phases and gates

| Phase | Work | Gate to continue | If the gate fails |
|---|---|---|---|
| **0. Pipeline** | The K=1 OreSplit plan (Tasks 1–10) | Zero-shot parse rate ≥0.8; checkpoint resume works on the A100; seconds/step measured | Fix the prompt or the trainer; no research yet |
| **1. Learning signal** | MAGRPO on OreSplit, 3 seeds × 150 steps | Normalised score beats zero-shot beyond seed spread | Try 0.5B vs 1.5B, the reward scale and the group size; if there's still no signal, RQ3 becomes "learning didn't help at this scale", which is still a result |
| **2. Structures** | Frozen-LLM structures on OreSplit + TriSplit, N = 2, 4 | Structures differ measurably on at least one metric | Make the tasks harder before going on to scaling |
| **3. Harder tasks + K>1** | SmeltChain, re-planning, N up to 8 | Oracle and IPPO baselines exist for each task | Drop to the tasks that work |
| **4. Learning vs design** | Rejection-sampling fine-tuning + MAGRPO on the best structure from Phase 2 | — | — |
| **5. Stretch: closure** | "Builder" item that activates a reserved player slot | Only if Phases 0–4 finish early | Becomes future work |
| **6. Writing** | — | — | — |

## Compute budget (estimates; Phase 0 replaces them)

| Item | Estimate |
|---|---|
| Factoriax on CPU (measured 2026-10-28) | ~2,000 ticks/s for 1 env; ~5,200 ticks/s for 128 envs |
| MAGRPO step, 1.5B, A100 | ~1 min |
| Phase 1: 3 seeds × 150 steps | ~8 A100-hours |
| Phase 2: frozen structures | Inference only; a few A100-hours, or Kaggle/Mac |
| Phase 4: 2 methods × 3 seeds | ~15–25 A100-hours |
| **Total** | **~30–40 A100-hours**, before reruns |

Each paid session runs with `--max-steps-this-session`, so spending is capped and a session can stop at any point and resume.

## Risks

| Risk | Mitigation |
|---|---|
| A small model can't follow the plan format | Zero-shot gate in Phase 0; constrained prompt; format penalty |
| No learning signal in the budget | Phase 1 gate before spending more; a null result is reportable |
| The task is too easy (one fixed split solves it) | Variants swap resources and spawns; the task ladder adds asymmetry |
| Reward hacking through quirks of the macro layer | The oracle defines the ceiling; inspect episode logs, not just scores |
| CoMLRL private API changes | Pin the commit; `ResumableMAGRPO` test |
| Colab disconnects or credit runs out | Atomic checkpoints every 10 steps; per-session step cap |
| Scope creep (closure, more structures) | Closure is stretch-only; at most 4 structures |
| Examiner questions using a fork of the environment | All changes are additive (`factoriax/marl/`); pin the upstream commit and cite the EWRL 2026 paper |

## Decisions for the supervisor

1. Move the main question from "which learning mechanism" to "which organisation, and does learning beat design" (RQ1–RQ3 above).
2. Switch the environment from FLE to Factoriax plus a macro layer, for compute reasons.
3. Re-scope the literature review. Add work on multi-agent organisation paradigms and the LLM topology/role literature, and keep the Sharpening vs Discovery thread for RQ3.

## Reading to add (verify before citing)

- Horling & Lesser (2004), a survey of multi-agent organisational paradigms
- Qian et al. (2024), MacNet: scaling LLM multi-agent collaboration across network topologies
- MetaGPT; ChatDev (fixed role hierarchies)
- Smith (1980), Contract Net Protocol (the market structure)
- NASA (1980), *Advanced Automation for Space Missions*; Freitas & Merkle (2004), *Kinematic Self-Replicating Machines* (probe motivation)
- Liu et al. (2026, AAAI), LLM Collaboration with MARL (MAGRPO / CoMLRL)
- Beurskens, Tomilin & Simão (2026, EWRL), Factoriax

## Open questions

- What is the thesis deadline? The phase lengths depend on it.
- How do messages fit into K=1? One message round before the plans, or messages only in the K>1 phases?
- Which structures make the final cut? Stigmergic is the cheapest and the most probe-like (it needs no communication bandwidth).
