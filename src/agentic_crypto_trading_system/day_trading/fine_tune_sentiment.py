#!/usr/bin/env python3
"""Fine-tune a sentiment model for crypto news."""
# Docs: https://huggingface.co/docs/transformers/training

import argparse
import json
import logging
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def check_dependencies():
    missing = []
    for pkg in ["datasets", "peft", "evaluate", "transformers", "accelerate", "sklearn"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        logger.error("Missing: %s. Run: poetry install --with finetune", ", ".join(missing))
        sys.exit(1)

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset-name", default="ElKulako/stocktwits-crypto")
    p.add_argument("--base-model", default="ProsusAI/finbert")
    p.add_argument("--output-dir", default="output/fine-tuned-sentiment")
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--use-lora", action="store_true")
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--merge-lora", action="store_true")
    p.add_argument("--evaluate", action="store_true")
    p.add_argument("--push-to-hub", type=str, default=None)
    return p.parse_args()


def load_and_prepare_dataset(dataset_name, tokenizer, max_length=512, test_size=0.2):
    from datasets import load_dataset

    try:
        ds = load_dataset(dataset_name, "sentences_allagree", trust_remote_code=True)
    except Exception as e:
        logger.error("Failed to load dataset '%s': %s", dataset_name, e)
        sys.exit(1)

    # Split if no validation set
    if "validation" not in ds:
        split = ds["train"].train_test_split(test_size=test_size)
        train_ds, val_ds = split["train"], split["test"]
    else:
        train_ds, val_ds = ds["train"], ds["validation"]

    # Map labels → integers
    LABEL_MAP = {"negative": 0, "bearish": 0, "neutral": 1, "positive": 2, "bullish": 2}

    def map_labels(example):
        example["label"] = LABEL_MAP[example["sentiment"].lower()]
        return example

    # train_ds = train_ds.map(map_labels)
    # val_ds = val_ds.map(map_labels)

    # Tokenize
    def tokenize(batch):
        return tokenizer(batch["sentence"], truncation=True, max_length=max_length, padding="max_length")

    train_ds = train_ds.map(tokenize, batched=True)
    val_ds = val_ds.map(tokenize, batched=True)

    # Set format for PyTorch
    train_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    val_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    logger.info("Train: %d, Val: %d", len(train_ds), len(val_ds))
    return train_ds, val_ds, LABEL_MAP


def load_base_model(model_name, num_labels):
    from transformers import AutoModelForSequenceClassification
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    logger.info("Loaded %s (num_labels=%d)", model_name, num_labels)
    return model

def apply_lora(model, r=8, alpha=32):
    from peft import LoraConfig, TaskType, get_peft_model
    config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=r, lora_alpha=alpha, lora_dropout=0.1,
        target_modules=["query", "value"],
    )
    model = get_peft_model(model, config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info("LoRA: %d / %d trainable (%.2f%%)", trainable, total, 100 * trainable / total)
    return model

def train_model(model, train_ds, val_ds, args):
    from transformers import Trainer, TrainingArguments
    import numpy as np

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"accuracy": (preds == labels).mean()}

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    start = time.time()
    try:
        trainer.train()
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.error("OOM! Try: --batch-size 8 or --use-lora")
            sys.exit(1)
        raise
    logger.info("Training done in %.1fs", time.time() - start)
    trainer.save_model(args.output_dir)
    return trainer

def evaluate_against_baseline(fine_tuned_path, baseline_name, val_ds, output_dir):
    from transformers import pipeline
    from sklearn.metrics import classification_report, confusion_matrix
    import numpy as np

    labels = [int(x) for x in val_ds["label"]]
    texts = val_ds["text"] if "text" in val_ds.column_names else [str(x) for x in val_ds["input_ids"]]

    def predict(model_path):
        pipe = pipeline("sentiment-analysis", model=model_path, device="cpu")
        # You'll need to map pipeline labels back to 0/1/2 here
        pass

def save_and_push(trainer, tokenizer, args, eval_metrics=None):
    tokenizer.save_pretrained(args.output_dir)
    logger.info("Saved to %s", args.output_dir)

    if args.use_lora and args.merge_lora:
        trainer.model = trainer.model.merge_and_unload()
        trainer.save_model(args.output_dir + "-merged")
        logger.info("Merged LoRA -> %s-merged", args.output_dir)

    if args.push_to_hub:
        import os
        if not os.environ.get("HF_TOKEN"):
            logger.error("Set HF_TOKEN env var or run: huggingface-cli login")
            sys.exit(1)
        trainer.push_to_hub(args.push_to_hub)
        logger.info("Pushed to https://huggingface.co/%s", args.push_to_hub)


def main():
    check_dependencies()
    args = parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    train_ds, val_ds, label_map = load_and_prepare_dataset(args.dataset_name, tokenizer, args.max_length)

    model = load_base_model(args.base_model, num_labels=len(set(label_map.values())))
    if args.use_lora:
        model = apply_lora(model, r=args.lora_r, alpha=args.lora_alpha)

    trainer = train_model(model, train_ds, val_ds, args)

    eval_metrics = None
    if args.evaluate:
        eval_metrics = evaluate_against_baseline(args.output_dir, "ElKulako/cryptobert", val_ds, args.output_dir)

    save_and_push(trainer, tokenizer, args, eval_metrics)

if __name__ == "__main__":
    main()
