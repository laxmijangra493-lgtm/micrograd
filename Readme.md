# 30-Day Aggressive AI/ML Engineering Plan
**Track:** ML/DL Engineering (training & fine-tuning) · **Starting point:** Some coding, new to AI/ML · **Time:** 3–5 hrs/day

**Core principle:** Build everything from scratch first, use libraries second, theory just-in-time, ship publicly every week.

**Tools you'll need:** Python, PyTorch, a GitHub account, a RunPod or Vast.ai account (credit card, ~$50 budget for the month), Hugging Face account.

---

## WEEK 1 — Build the Machinery by Hand
**Goal:** Understand backprop and neural nets at the code level so you're never confused by what a "gradient," "layer," or "loss" actually is.

- **Day 1 — Setup + micrograd (part 1).** Install Python/PyTorch, set up a GitHub repo as your public build log. Watch/code along with Karpathy's "micrograd" video — build a tiny scalar autograd engine from raw Python.
- **Day 2 — micrograd (part 2).** Finish micrograd. Build a tiny MLP classifier using only your own engine — no PyTorch yet.
- **Day 3 — Real PyTorch.** Rebuild yesterday's MLP using actual `torch.Tensor` and `autograd`. Learn `.backward()`, optimizers, `nn.Module`.
- **Day 4 — makemore (part 1-2).** Build a bigram and then MLP character-level language model. Focus entirely on the training loop: forward → loss → backward → step.
- **Day 5 — makemore (part 3-4).** Add batchnorm, learning rate scheduling, train/val splits. Deliberately debug a "why isn't my loss dropping" problem yourself.
- **Day 6 — Backfill the vocabulary.** Read the PyTorch 60-min blitz + `nn.Module` docs cover to cover once. Rewrite one earlier model cleanly using proper `nn.Module` structure.
- **Day 7 — Review + ship.** Write a short public post/README explaining backprop and your MLP in your own words. Push everything to GitHub.

## WEEK 2 — Build a GPT From Scratch
**Goal:** Understand transformers/attention at the code level, and get comfortable renting and using cloud GPUs.

- **Day 8 — Self-attention.** Karpathy's "Let's Build GPT" part 1 — implement single-head self-attention from scratch.
- **Day 9 — Multi-head + blocks.** Multi-head attention, transformer block, stacking layers.
- **Day 10 — Train it.** Finish your nanoGPT-style model, train a tiny character-level Shakespeare GPT on a laptop/free Colab GPU.
- **Day 11 — Rent your first GPU.** Set up RunPod or Vast.ai, rent a cheap RTX 4090 (~$0.35–0.50/hr). Re-run and scale up your training there. Practice SSH-in, monitor usage, checkpoint, and shut the instance down — this cost-discipline habit matters for the rest of the month.
- **Day 12 — Tokenizer from scratch.** Karpathy's BPE tokenizer video — build your own tokenizer and swap it into your model.
- **Day 13 — Experiment day.** Vary model size, context length, learning rate. Start a run log (spreadsheet: hyperparams → loss) — your first experiment-tracking habit.
- **Day 14 — Ship.** Write up "I built a GPT from scratch" with your loss curves and what broke. Post it publicly.

## WEEK 3 — Real Fine-Tuning on Real Models
**Goal:** Move from toy models to fine-tuning actual open-weight LLMs — the job-relevant skill.

- **Day 15 — LoRA/QLoRA theory (light).** Install `transformers` + `peft`. Skim the LoRA and QLoRA papers just enough to know why they work (freeze base weights, train small low-rank adapters).
- **Day 16 — First real fine-tune.** Install Unsloth. Run their quickstart to fine-tune a small open model (Qwen2.5-7B or Llama-3-8B class) with QLoRA on your rented GPU. Get one successful end-to-end run.
- **Day 17 — Pick your project.** Choose a narrow, real fine-tuning project you actually care about (a writing style, a support-ticket assistant, a coding-style model, a niche Q&A bot). Curate/clean 200–1000 examples.
- **Day 18 — Your own data.** Format your dataset as instruction/output pairs. Run your first fine-tune on your own data.
- **Day 19 — Evaluate.** Build a small eval set. Compare base model vs. fine-tuned outputs side by side. Learn to spot overfitting and catastrophic forgetting.
- **Day 20 — Iterate.** Adjust learning rate, LoRA rank, epochs. Try mixing in general examples to prevent forgetting. Log every run.
- **Day 21 — Quantize + export.** Quantize your fine-tuned model to 4-bit, test the speed/quality trade-off. Write up what changed.

## WEEK 4 — Depth, Breadth, and Shipping
**Goal:** Turn three weeks of reps into a real portfolio piece and a clear next direction.

- **Day 22 — Read one paper properly.** Pick something relevant (LoRA, DPO, or a GRPO/RLHF overview) and actually work through the method section, not just the abstract.
- **Day 23 — Second technique.** Try something new: DPO/preference tuning on a small preference dataset, or reasoning-style SFT with step-by-step outputs.
- **Day 24 — Build an eval harness.** Write a script that runs a batch of prompts through both models and scores/compares outputs automatically.
- **Day 25 — Package the project.** Clean up your best repo: clear README (problem → approach → results → loss curves), reproducible script.
- **Day 26 — Write it up.** A proper technical blog post on your 30-day project — this is portfolio piece #1.
- **Day 27 — Cost retrospective.** Total up your month's compute spend, note what was wasteful, tighten your workflow.
- **Day 28 — Choose your specialization.** Based on what you enjoyed most: research-leaning (bigger models, pretraining) or applied (production fine-tuning, MLOps). Scope a 2-week deeper project for month 2.
- **Day 29 — Buffer day.** Catch up, fix bugs, polish anything unfinished.
- **Day 30 — Ship day.** Publish your project properly (GitHub, personal site, or Hugging Face). Post in a relevant community, apply somewhere, or pitch it — even before you feel "ready."

---

## Budget Reference
- Total compute for the month: roughly **$30–60** if you follow the plan (mostly RTX 4090-class spot/on-demand rentals for hours at a time, not days).
- Skip A100/H100 rentals entirely for this month — LoRA/QLoRA on a 24GB card covers everything above.

## Notes
- This is intentionally aggressive. If a day runs long, don't skip the *shipping* step (writing up / pushing to GitHub) to save time elsewhere — that's the step that actually compounds.
- Days 7, 14, 21, 30 are your checkpoints. If you're behind, catch up there rather than mid-week.
- Communities worth lurking in while you build: r/LocalLLaMA, the Hugging Face forums, and Karpathy's Discord (linked from his course page).