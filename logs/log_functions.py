import re
import json
from ast import literal_eval
from pathlib import Path
import csv
import os
import pandas as pd
import wandb
from types import SimpleNamespace

import sys
sys.path.append('../')
from init_wandb import get_wandb_obj

# --- PARSE METRICS -----------------------------------------------------------
FLOAT = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'

_LOSS_RE = re.compile(
    rf'^\s*(Train|Test)\s+loss:\s*({FLOAT})\s+\(MSE:\s*({FLOAT}),\s+perct:\s*({FLOAT}),\s+SSIM:\s*({FLOAT}),\)'
    rf'(?:.*?\|\s*([0-9]+)\s*/\s*([0-9]+)\s*\[)?'
)

def parse_log_metrics(path):
    """
    Returns a list of rows: dict(split, step, total_steps, loss, mse, perct, ssim).
    If tqdm step not present, assigns monotonically increasing steps per split.
    """
    rows = []
    step_counters = {"Train": 0, "Test": 0}  # fallback counters

    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            m = _LOSS_RE.search(line.strip())
            if not m:
                continue
            split, loss, mse, perct, ssim, _, _ = m.groups()

            rows.append({
                "split": split,       # "Train" or "Test"
                "loss": float(loss),
                "mse": float(mse),
                "perct": float(perct),
                "ssim": float(ssim),
            })
    return rows


# --- PARSE CONFIG ("Parameters:" block) --------------------------------------
_PARAM_START_RE = re.compile(r'^\s*Parameters:\s*$')
# allow ANY key chars up to the first colon (handles Unicode like λ)
_PARAM_ITEM_RE  = re.compile(r'^\s*-\s*(.+?)\s*:\s*(.*?)\s*$')

def _coerce_value(s):
    """Coerce to int/float/bool/None/list if possible, else return the raw string."""
    from ast import literal_eval
    s = s.strip()
    # common literals pass through literal_eval
    try:
        return literal_eval(s)
    except Exception:
        # leave as string (paths like ./concat..., tokens like linear, etc.)
        return s

def parse_config_block(path):
    """
    Scans the log for the 'Parameters:' block and returns a dict of args.
    The block ends at the first blank line after it starts.
    """
    args = {}
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    in_block = False
    for line in lines:
        if not in_block:
            if _PARAM_START_RE.match(line):
                in_block = True
            continue

        # inside parameters block
        if not line.strip():              # blank line ends block
            break

        m = _PARAM_ITEM_RE.match(line)
        if not m:
            # If it doesn't match "- key: value", keep scanning (don’t break early)
            continue

        key, val = m.groups()
        key = key.strip()
        args[key] = _coerce_value(val)

    return args


def save_metrics_csv(rows, csv_path):
    """
    rows: list of dicts from parse_log_metrics
    """
    if not rows:
        raise ValueError("No metric rows parsed from log.")
    fields = ["split", "loss", "mse", "perct", "ssim"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        

def save_config_json(args_dict, json_path):
    with open(json_path, "w") as f:
        json.dump(args_dict, f, indent=2)



def ns_from_json(json_path):
    d = json.load(open(json_path))
    return SimpleNamespace(
        results_folder=d.get("results_folder", "./run"),
        img_size=d.get("img_size", 64),
        down=d.get("down", 8),
        timesteps=d.get("timesteps", 1000),
        sampling_timesteps=d.get("sampling_timesteps", 150),
        beta_schedule=d.get("beta_schedule", "linear"),
        batch_size=d.get("batch_size", 16),
        lr=d.get("lr", 1e-4),
        n_epochs=d.get("n_epochs", 0),
        ema_decay=d.get("ema_decay", 0.995),
        blank_prob=d.get("blank_prob", 0),
        t2w_offset=d.get("t2w_offset", 0),
        save_every=d.get("save_every", 0),
        sample_every=d.get("sample_every", 0),
        use_T2W=d.get("use_T2W", False),
        use_mask=d.get("use_mask", False),
        finetune=d.get("finetune", False),
        controlnet=d.get("controlnet", False),
        upsample=d.get("upsample", False),
    )

def pair_alternating_train_test(df):
    # clean split column
    df = df.copy()
    df["split"] = df["split"].str.strip()

    # sanity check: we expect alternating Train/Test
    if not set(df["split"]).issubset({"Train","Test"}):
        raise ValueError("CSV contains splits other than Train/Test.")

    paired = []
    i = 0
    while i < len(df):
        # grab Train (if present)
        t = df.iloc[i] if i < len(df) and df.iloc[i]["split"] == "Train" else None
        # grab Test  (if present)
        v = df.iloc[i+1] if i+1 < len(df) and df.iloc[i+1]["split"] == "Test" else None

        # If order flips (rare), try to recover:
        if t is None and i < len(df) and df.iloc[i]["split"] == "Test":
            v = df.iloc[i]
            t = df.iloc[i+1] if i+1 < len(df) and df.iloc[i+1]["split"] == "Train" else None

        paired.append({
            "Train - total": float(t["loss"]) if t is not None else None,
            "Train - MSE":   float(t["mse"])  if t is not None else None,
            "Train - perceptual": float(t["perct"]) if t is not None else None,
            "Train - SSIM":  float(t["ssim"]) if t is not None else None,
            "Test - total":  float(v["loss"]) if v is not None else None,
            "Test - MSE":    float(v["mse"])  if v is not None else None,
            "Test - perceptual":  float(v["perct"]) if v is not None else None,
            "Test - SSIM":   float(v["ssim"]) if v is not None else None,
        })

        i += 2  # advance by two rows per step
    return paired

def log_csv_json_to_wandb(csv_path, json_path, stride=50, offset=0):
    args = ns_from_json(json_path)
    run = get_wandb_obj(args)

    try:  
        # make 'step' the x-axis
        run.define_metric("step")
        run.define_metric("*", step_metric="step")
        
        df = pd.read_csv(csv_path)
        paired = pair_alternating_train_test(df)

        for i, row in enumerate(paired):
            step_val = offset + i * stride   # 0, 50, 100, ...
            run.log(row, step=step_val)

        # summaries (unchanged)
        if paired:
            last = paired[-1]
            for k, v in last.items():
                if v is not None:
                    run.summary[f"final_{k}"] = v
            vt = [r["total_test"] for r in paired if r["total_test"] is not None]
            if vt:
                run.summary["best_total_test"] = min(vt)

    finally:
        run.finish()
