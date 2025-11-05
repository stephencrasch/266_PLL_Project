# train_modernbert_pseudolabel_round.py
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    set_seed,
)
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import numpy as np
import json
import os
import re

# =======================
# Config (EDIT THESE)
# =======================
set_seed(42)

INPUT_JSON = "all_category_results_with_tfidf.json"  # <-- your round output file
OUTPUT_DIR = "./modernbert_round"                    # will append _{round_name}
ROUND_NAME = "r0"                                    # e.g., "r0", "r1", ...

MODEL_NAME = "answerdotai/ModernBERT-base"
MAX_LEN = 512
PER_CLASS_CAP = 2000          # cap examples per class (after dedupe); set None to disable
VAL_FRAC = 0.15               # validation split fraction
BATCH_TRAIN = 8               # reduce if MPS VRAM is tight
BATCH_EVAL  = 8
EPOCHS = 3
LR = 2e-5
WARMUP = 0.06
WEIGHT_DECAY = 0.01

# If you want to normalize expanded categories back to a base label,
# list your base class names here (all lowercase). Example:
# BASE_LABELS = ["plumber", "hvac", "cocktail bar", "thai", "italian"]
BASE_LABELS = []  # [] means "use category text as-is"

# =======================
# Helpers
# =======================
def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).lower()

def map_category_to_base(cat: str, base_labels: list[str]) -> str:
    """
    Map expanded category 'cat' (e.g., 'plumber water repair leak') back to a base label
    using longest substring match against BASE_LABELS. If BASE_LABELS is empty, return cat.
    """
    if not base_labels:
        return normalize_text(cat)
    cat_norm = normalize_text(cat)
    # choose the base label that appears as a substring with the longest length
    candidates = [(bl, len(bl)) for bl in base_labels if bl in cat_norm]
    if candidates:
        return max(candidates, key=lambda x: x[1])[0]
    # fallback: if no match, just use the original normalized category
    return cat_norm

def stratified_split(df: pd.DataFrame, label_col: str, val_frac: float, seed: int = 42):
    """
    Simple stratified split per label. (Sklearn's train_test_split needs label counts >= 2; this works even for tiny classes.)
    """
    train_parts, val_parts = [], []
    rng = np.random.RandomState(seed)
    for lbl, g in df.groupby(label_col):
        idx = np.arange(len(g))
        rng.shuffle(idx)
        n_val = max(1, int(len(g) * val_frac)) if len(g) > 1 else 1
        val_idx = idx[:n_val]
        trn_idx = idx[n_val:] if len(g) - n_val > 0 else idx[:0]
        val_parts.append(g.iloc[val_idx])
        train_parts.append(g.iloc[trn_idx])
    train_df = pd.concat(train_parts, ignore_index=True)
    val_df   = pd.concat(val_parts, ignore_index=True)
    # Edge case: if any class had only 1 example, it went to val; ensure train isn’t empty
    if len(train_df) == 0:
        # move half of val back to train (last resort)
        half = len(val_df) // 2
        train_df = val_df.iloc[:half].copy()
        val_df   = val_df.iloc[half:].copy()
    return train_df, val_df

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }

# =======================
# Load pseudo-labeled JSON for THIS ROUND
# Expected shape:
# {
#   "similarity_threshold": ...,
#   "k_results": ...,
#   "categories": [
#       {
#         "category": "Cocktail Bar bartender friendly atmosphere",
#         "reviews": [{"text": "...", ...}, ...]
#       },
#       ...
#   ]
# }
# =======================
with open(INPUT_JSON) as f:
    data = json.load(f)

rows = []
for cat in data.get("categories", []):
    cat_name = cat.get("category", "")
    base_label = map_category_to_base(cat_name, [normalize_text(x) for x in BASE_LABELS])

    for r in cat.get("reviews", []):
        txt = (r.get("text") or "").strip()
        if txt:
            rows.append({"text": txt, "label": base_label})

df = pd.DataFrame(rows)
# Deduplicate exact texts
df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

# Optional: cap per class for balance
if PER_CLASS_CAP is not None:
    df = (
        df.groupby("label", group_keys=False)
          .apply(lambda g: g.sample(n=min(len(g), PER_CLASS_CAP), random_state=42))
          .reset_index(drop=True)
    )

print("Class counts:")
print(df["label"].value_counts())

# Build stable id maps from observed labels THIS ROUND
labels_sorted = sorted(df["label"].unique())
label2id = {lbl: i for i, lbl in enumerate(labels_sorted)}
id2label = {i: lbl for lbl, i in label2id.items()}
df["labels"] = df["label"].map(label2id)

# Stratified split
train_df, val_df = stratified_split(df[["text", "labels", "label"]], label_col="label", val_frac=VAL_FRAC, seed=42)

# =======================
# Tokenize
# =======================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
collator = DataCollatorWithPadding(tokenizer=tokenizer)

def tok(batch):
    return tokenizer(batch["text"], truncation=True, max_length=MAX_LEN)

train_ds = Dataset.from_pandas(train_df[["text", "labels"]], preserve_index=False).map(
    tok, batched=True, remove_columns=["text"]
)
val_ds = Dataset.from_pandas(val_df[["text", "labels"]], preserve_index=False).map(
    tok, batched=True, remove_columns=["text"]
)

ds = DatasetDict({"train": train_ds, "validation": val_ds})

# =======================
# Model
# =======================
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(labels_sorted),
    id2label=id2label,
    label2id=label2id,
    trust_remote_code=True,
)

# =======================
# Train
# =======================
out_dir = f"{OUTPUT_DIR}_{ROUND_NAME}"
os.makedirs(out_dir, exist_ok=True)

training_args = TrainingArguments(
    output_dir=out_dir,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_TRAIN,
    per_device_eval_batch_size=BATCH_EVAL,
    learning_rate=LR,
    weight_decay=WEIGHT_DECAY,
    warmup_ratio=WARMUP,
    logging_steps=50,
    save_total_limit=2,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    tokenizer=tokenizer,
    data_collator=collator,
    compute_metrics=compute_metrics,
)

trainer.train()
print(trainer.evaluate())

# Save checkpoint + label mapping for this round
trainer.save_model(out_dir)
tokenizer.save_pretrained(out_dir)
with open(os.path.join(out_dir, "labels.json"), "w") as f:
    json.dump({"labels": labels_sorted, "label2id": label2id}, f, indent=2)
