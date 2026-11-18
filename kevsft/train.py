"""Fine-tune a released Kev choice router on local, group-disjoint examples."""

import argparse
import gc
import json
import math
import random
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from .data import INSTRUCTIONS, load_dataset, load_labels, make_question, render_state


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Dataset directory")
    parser.add_argument("--output", type=Path, required=True, help="New local run directory")
    parser.add_argument("--model", default="jaredpalmer/kev-0.8b", help="Released LoRA checkpoint or its local copy")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--max-row-tokens", type=int, default=2048,
                        help="Reject training or evaluation rows exceeding this token budget")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.max_row_tokens) < 1:
        parser.error("epochs, batch size and row budget must be positive")
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("learning rate must be positive and finite")
    return args


def score_matrix(counts, keys):
    """Rows are reference classes, columns are predictions; 'other' comes last."""
    support = [sum(row) for row in counts]
    correct = sum(row[i] for i, row in enumerate(counts))
    samples = sum(support)
    recalls = {key: counts[i][i] / support[i] if support[i] else None
               for i, key in enumerate(keys)}
    known = samples - support[-1]
    return {
        "samples": samples,
        "correct": correct,
        "accuracy": correct / samples,
        "support": dict(zip(keys, support)),
        "recall": recalls,
        "fallback_recall": recalls[keys[-1]],
        "fallback_false_positive_rate": sum(row[-1] for row in counts[:-1]) / known if known else None,
        "confusion_matrix": counts,
    }


def encode(model, tokenizer, examples, options, token_limit, rng=None):
    encodings = []
    targets = []
    for example in examples:
        order = list(range(len(options)))
        if rng is not None:
            rng.shuffle(order)
        label = order.index(example.label)
        record = {
            "state": example.state,
            "questions": [{
                "instr": INSTRUCTIONS,
                "options": [options[i] for i in order],
                "label": label,
            }],
        }
        encodings.append(model.encode(tokenizer, record, strict=True,
                                      max_state=token_limit, max_branch=token_limit))
        targets.append(label)
    return encodings, targets


def batches(rows, size):
    for start in range(0, len(rows), size):
        yield rows[start:start + size]


def evaluate(model, tokenizer, examples, options, keys, args, autocast):
    import torch

    counts = [[0] * len(keys) for _ in keys]
    ce_sum = 0.0
    model.eval()
    with torch.inference_mode(), autocast():
        for batch in batches(examples, args.batch_size):
            packed, targets = encode(model, tokenizer, batch, options, args.max_row_tokens)
            logits = torch.stack([out[0].float() for out in model.forward_batch(packed)])
            target = torch.tensor(targets, device=args.device)
            ce_sum += torch.nn.functional.cross_entropy(logits, target, reduction="sum").item()
            for gold, pred in zip(targets, logits.argmax(dim=1).tolist()):
                counts[gold][pred] += 1
    metrics = score_matrix(counts, keys)
    metrics["raw_cross_entropy"] = ce_sum / len(examples)
    return metrics


def train_epoch(model, tokenizer, examples, options, args, optimizer, params, rng, autocast):
    import torch

    shuffled = list(examples)
    rng.shuffle(shuffled)
    ce_sum = 0.0
    model.train()
    for batch in batches(shuffled, args.batch_size):
        packed, targets = encode(model, tokenizer, batch, options, args.max_row_tokens, rng)
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            logits = torch.stack([out[0].float() for out in model.forward_batch(packed)])
            loss = torch.nn.functional.cross_entropy(logits, torch.tensor(targets, device=args.device))
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
        optimizer.step()
        ce_sum += loss.item() * len(batch)
    return ce_sum / len(shuffled)


def save_candidate(directory, model, tokenizer, meta):
    from kev.checkpoint import write_meta

    directory.mkdir(exist_ok=True)
    model.lm.save_pretrained(directory)
    tokenizer.save_pretrained(directory)
    head = {name: tensor.detach().cpu() for name, tensor in model.head.state_dict().items()}
    # A changed classifier needs its own calibration. Store raw T=1 scores.
    write_meta(directory, replace(meta, head=head, temperature=1.0, extra={}, holdout=[]))


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def run(args):
    from contextlib import nullcontext

    import torch
    from kev.checkpoint import Checkpoint, LoadOptions
    from kev.model import DecisionModel, load_tokenizer

    options, splits = load_dataset(args.data)
    labels = load_labels(args.data)
    keys = list(labels)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu for a small smoke run")
    if args.output.exists():
        raise FileExistsError("Choose a new output directory for each run")
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    use_bf16 = args.device == "cuda" and torch.cuda.is_bf16_supported()
    amp = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if use_bf16 else nullcontext
    checkpoint = Checkpoint(args.model)
    meta = checkpoint.meta
    if checkpoint.full or not meta.lora or meta.weights_dtype != "fp32" or meta.special_embeddings:
        raise ValueError("Use a released FP32 LoRA + pointer-head checkpoint, such as Kev-0.8B")
    tokenizer = load_tokenizer(meta.base, revision=meta.base_revision)
    model = DecisionModel(meta.base, tokenizer, args.device, revision=meta.base_revision,
                          lora=meta.lora, head_dim=meta.head_dim, option_isolation=meta.option_isolation,
                          dtype=torch.float32)
    checkpoint.warm_start(model, meta)
    model.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.lm.config.use_cache = False
    named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not named or not all("lora_" in name or name.startswith("head.") for name, p in named):
        raise RuntimeError("Expected a frozen backbone with trainable LoRA and pointer head")
    params = [p for _, p in named]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    args.output.mkdir(parents=True, exist_ok=False)

    with TemporaryDirectory(prefix=".in-progress-", dir=args.output) as staging:
        stage = Path(staging)
        best = stage / "best"
        history = []
        best_score = None
        for epoch in range(1, args.epochs + 1):
            ce = train_epoch(model, tokenizer, splits["train"], options, args, optimizer, params, rng, amp)
            valid = evaluate(model, tokenizer, splits["validation"], options, keys, args, amp)
            entry = {"epoch": epoch, "train_ce": ce, "validation": valid}
            print(json.dumps({"epoch": epoch, "train_ce": ce,
                              "validation_accuracy": valid["accuracy"]}), flush=True)
            history.append(entry)
            rank = (valid["accuracy"], -valid["raw_cross_entropy"])
            if best_score is None or rank > best_score:
                best_epoch, best_score = epoch, rank
                save_candidate(best, model, tokenizer, meta)
        # Evaluate the frozen choice once with the same merged settings for both models.
        del model, tokenizer, params, named, optimizer
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()
        load = LoadOptions(dtype=torch.bfloat16 if use_bf16 else torch.float32, merge=True,
                           backend="torch", fused=False, cuda_graphs=False, temperature=1.0)
        holdout = {}
        for role, source in (("baseline", checkpoint), ("candidate", Checkpoint(str(best)))):
            tok, scorer = source.load(args.device, load)
            holdout[role] = evaluate(scorer, tok, splits["test"], options, keys, args, nullcontext)
            del tok, scorer
            gc.collect()
        report = {
            "seed": args.seed, "labels": keys, "selected_epoch": best_epoch,
            "selection": "Highest validation accuracy, then lowest validation raw cross-entropy; first epoch on tie",
            "batch_size": args.batch_size, "learning_rate": args.lr, "max_row_tokens": args.max_row_tokens,
            "device": args.device, "forward_dtype": "bf16" if use_bf16 else "fp32",
            "initial_weights_sha256": checkpoint.weights_sha256(), "history": history, "test": holdout,
            "test_accuracy_delta": holdout["candidate"]["accuracy"] - holdout["baseline"]["accuracy"],
        }
        dump(best / "request.json", {"model": "kev-latest", "state": render_state("Replace this with a request."),
                                     "questions": {"route": make_question(labels)}})
        dump(stage / "report.json", report)
        # Serving can discover best/ only after all writes and held-out evaluation succeed.
        (stage / "report.json").rename(args.output / "report.json")
        best.rename(args.output / "best")
        print(json.dumps({"selected_epoch": best_epoch, "test_accuracy_delta": report["test_accuracy_delta"]}))


def main():
    run(arguments())


if __name__ == "__main__":
    main()
