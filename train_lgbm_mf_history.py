#!/usr/bin/env python3
"""
Train a LightGBM CTR model with aggregate, history, and MF/two-tower features.

Dependencies:
    python3 -m pip install numpy lightgbm

Example:
    python3 train_lgbm_mf_history.py \
        --train train.csv \
        --evaluation evaluation.csv \
        --out submission_lgbm_mf_history.csv

The script is intentionally self-contained and streams the large CSV files. By
default it trains LightGBM on a deterministic sample of the training rows so it
can run on a laptop; increase --max-train-rows or set it to 0 to use all rows.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from typing import Dict, List, Sequence, Tuple


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def elapsed(start: float) -> str:
    seconds = time.time() - start
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{seconds / 60:.1f}m"


def require_ml_deps():
    try:
        import lightgbm as lgb  # type: ignore
        import numpy as np  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "Missing ML dependency. Install the required packages first:\n"
            "  python3 -m pip install numpy lightgbm\n\n"
            f"Original import error: {exc}"
        )
    return np, lgb


def stable_unit_hash(value: str, seed: int) -> float:
    raw = f"{seed}:{value}".encode("utf-8")
    digest = hashlib.blake2b(raw, digest_size=8).digest()
    return int.from_bytes(digest, "little") / float(1 << 64)


def sigmoid_scalar(x: float) -> float:
    x = max(min(x, 35.0), -35.0)
    return 1.0 / (1.0 + math.exp(-x))


def logit(p: float) -> float:
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def pair_key(a: str, b: str) -> str:
    return a + "\x1f" + b


def parse_json_array(value: str) -> List:
    if not value or value == "[]":
        return []
    return json.loads(value)


def update_stats(stats: Dict[str, List[int]], key: str, clicked: int) -> None:
    vals = stats.get(key)
    if vals is None:
        stats[key] = [1, clicked]
    else:
        vals[0] += 1
        vals[1] += clicked


def build_label_stats(train_path: str) -> dict:
    start = time.time()
    log("pass 1/4: scanning train.csv for aggregate CTR stats ...")

    item_stats: Dict[str, List[int]] = {}
    user_stats: Dict[str, List[int]] = {}
    item_time_stats: Dict[str, List[int]] = {}
    user_time_stats: Dict[str, List[int]] = {}
    item_day_stats: Dict[str, List[int]] = {}

    total_rows = 0
    total_clicks = 0
    min_ts = None
    max_ts = None

    with open(train_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            y = int(row["clicked"])
            uid = row["user_id"]
            iid = row["item_id"]
            tb = row["time_bucket"]
            dt = row["day_type"]
            ts = int(row["timestamp"])

            update_stats(item_stats, iid, y)
            update_stats(user_stats, uid, y)
            update_stats(item_time_stats, pair_key(iid, tb), y)
            update_stats(user_time_stats, pair_key(uid, tb), y)
            update_stats(item_day_stats, pair_key(iid, dt), y)

            total_rows += 1
            total_clicks += y
            min_ts = ts if min_ts is None else min(min_ts, ts)
            max_ts = ts if max_ts is None else max(max_ts, ts)

            if total_rows % 1_000_000 == 0:
                log(f"  scanned {total_rows:,} rows")

    global_ctr = total_clicks / max(total_rows, 1)
    user_to_idx = {uid: idx for idx, uid in enumerate(user_stats.keys(), start=1)}
    item_to_idx = {iid: idx for idx, iid in enumerate(item_stats.keys(), start=1)}

    log(
        "  done: "
        f"{total_rows:,} rows, {len(user_to_idx):,} users, {len(item_to_idx):,} items, "
        f"global CTR={global_ctr:.6f} ({elapsed(start)})"
    )

    return {
        "item_stats": item_stats,
        "user_stats": user_stats,
        "item_time_stats": item_time_stats,
        "user_time_stats": user_time_stats,
        "item_day_stats": item_day_stats,
        "total_rows": total_rows,
        "total_clicks": total_clicks,
        "global_ctr": global_ctr,
        "global_bias": logit(global_ctr),
        "min_ts": min_ts or 0,
        "max_ts": max_ts or 0,
        "user_to_idx": user_to_idx,
        "item_to_idx": item_to_idx,
    }


def collect_mf_samples(
    np,
    train_path: str,
    stats: dict,
    max_rows: int,
    seed: int,
) -> Tuple:
    start = time.time()
    total_rows = stats["total_rows"]
    cap = total_rows if max_rows <= 0 else min(max_rows, total_rows)
    sample_prob = 1.0 if max_rows <= 0 else min(1.0, cap / max(total_rows, 1))

    log(f"pass 2/4: collecting up to {cap:,} rows for MF/two-tower training ...")
    user_idx = np.empty(cap, dtype=np.int32)
    item_idx = np.empty(cap, dtype=np.int32)
    y = np.empty(cap, dtype=np.float32)

    n = 0
    with open(train_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for seen, row in enumerate(reader, start=1):
            if sample_prob < 1.0 and stable_unit_hash(row["rowid"], seed) > sample_prob:
                continue
            if n >= cap:
                continue
            user_idx[n] = stats["user_to_idx"].get(row["user_id"], 0)
            item_idx[n] = stats["item_to_idx"].get(row["item_id"], 0)
            y[n] = float(row["clicked"])
            n += 1
            if seen % 1_000_000 == 0:
                log(f"  sampled {n:,} / {seen:,} scanned rows")

    log(f"  collected {n:,} MF rows ({elapsed(start)})")
    return user_idx[:n], item_idx[:n], y[:n]


def train_logistic_mf(
    np,
    user_idx,
    item_idx,
    y,
    n_users: int,
    n_items: int,
    factors: int,
    epochs: int,
    lr: float,
    reg: float,
    batch_size: int,
    seed: int,
    global_ctr: float,
    pos_weight: float,
) -> dict:
    start = time.time()
    rng = np.random.default_rng(seed)
    user_emb = rng.normal(0.0, 0.03, size=(n_users + 1, factors)).astype(np.float32)
    item_emb = rng.normal(0.0, 0.03, size=(n_items + 1, factors)).astype(np.float32)
    user_emb[0, :] = 0.0
    item_emb[0, :] = 0.0

    user_bias = np.zeros(n_users + 1, dtype=np.float32)
    item_bias = np.zeros(n_items + 1, dtype=np.float32)
    global_bias = np.float32(logit(global_ctr))
    n = len(y)

    log(
        f"training MF/two-tower embeddings: rows={n:,}, factors={factors}, "
        f"epochs={epochs}"
    )

    for epoch in range(1, epochs + 1):
        order = rng.permutation(n)
        loss_sum = 0.0
        batches = 0
        for start_ix in range(0, n, batch_size):
            ix = order[start_ix : start_ix + batch_size]
            u = user_idx[ix]
            i = item_idx[ix]
            target = y[ix]

            old_u = user_emb[u].copy()
            old_i = item_emb[i].copy()
            z = global_bias + user_bias[u] + item_bias[i] + np.sum(old_u * old_i, axis=1)
            z = np.clip(z, -35.0, 35.0)
            pred = 1.0 / (1.0 + np.exp(-z))
            weight = np.where(target > 0.5, pos_weight, 1.0).astype(np.float32)
            grad = (pred - target) * weight

            np.add.at(user_emb, u, -lr * (grad[:, None] * old_i + reg * old_u))
            np.add.at(item_emb, i, -lr * (grad[:, None] * old_u + reg * old_i))
            np.add.at(user_bias, u, -lr * (grad + reg * user_bias[u]))
            np.add.at(item_bias, i, -lr * (grad + reg * item_bias[i]))

            eps = 1e-7
            batch_loss = -(
                target * np.log(np.clip(pred, eps, 1.0))
                + (1.0 - target) * np.log(np.clip(1.0 - pred, eps, 1.0))
            )
            loss_sum += float(np.mean(batch_loss))
            batches += 1

        user_emb[0, :] = 0.0
        item_emb[0, :] = 0.0
        user_bias[0] = 0.0
        item_bias[0] = 0.0
        log(f"  MF epoch {epoch}/{epochs}: logloss={loss_sum / max(batches, 1):.5f}")

    log(f"  MF training done ({elapsed(start)})")
    return {
        "user_emb": user_emb,
        "item_emb": item_emb,
        "user_bias": user_bias,
        "item_bias": item_bias,
        "global_bias": float(global_bias),
    }


class FeatureBuilder:
    def __init__(self, np, stats: dict, mf: dict, smoothing: float, include_embeddings: bool):
        self.np = np
        self.stats = stats
        self.mf = mf
        self.smoothing = smoothing
        self.include_embeddings = include_embeddings
        self.factors = int(mf["item_emb"].shape[1])
        self.global_ctr = float(stats["global_ctr"])

        names = [
            "user_code",
            "item_code",
            "time_bucket",
            "day_type",
            "item_impr_log",
            "item_click_log",
            "item_ctr",
            "user_impr_log",
            "user_click_log",
            "user_ctr",
            "item_time_impr_log",
            "item_time_ctr",
            "user_time_impr_log",
            "user_time_ctr",
            "item_day_impr_log",
            "item_day_ctr",
            "hist_len",
            "hist_unique",
            "hist_ctr_mean",
            "hist_ctr_max",
            "hist_ctr_last",
            "hist_item_impr_mean_log",
            "candidate_in_history",
            "candidate_hist_count",
            "candidate_last_pos_from_end",
            "candidate_age_hours_log",
            "last_history_age_hours_log",
            "mean_history_age_hours_log",
            "mf_logit",
            "mf_prob",
            "mf_dot",
            "mf_user_bias",
            "mf_item_bias",
            "history_dot",
            "history_cosine",
            "history_recent_dot",
            "history_recent_cosine",
        ]
        if include_embeddings:
            names.extend([f"user_emb_{i}" for i in range(self.factors)])
            names.extend([f"item_emb_{i}" for i in range(self.factors)])
            names.extend([f"hist_emb_{i}" for i in range(self.factors)])
            names.extend([f"hist_recent_emb_{i}" for i in range(self.factors)])
        self.names = names
        self.categorical_names = ["user_code", "item_code", "time_bucket", "day_type"]

    def n_features(self) -> int:
        return len(self.names)

    def categorical_indices(self) -> List[int]:
        name_to_idx = {name: i for i, name in enumerate(self.names)}
        return [name_to_idx[name] for name in self.categorical_names]

    def _stat_values(
        self,
        stats: Dict[str, List[int]],
        key: str,
        exclude_impressions: int = 0,
        exclude_clicks: int = 0,
    ) -> Tuple[int, int, float]:
        vals = stats.get(key)
        if vals is None:
            return 0, 0, self.global_ctr
        impressions = max(vals[0] - exclude_impressions, 0)
        clicks = max(vals[1] - exclude_clicks, 0)
        ctr = (clicks + self.smoothing * self.global_ctr) / (impressions + self.smoothing)
        return impressions, clicks, ctr

    def build(self, row: dict, history_items: Sequence[str], history_timestamps: Sequence[int]) -> List[float]:
        np = self.np
        uid = row["user_id"]
        iid = row["item_id"]
        time_bucket = int(row["time_bucket"])
        day_type = int(row["day_type"])
        ts = int(row["timestamp"])
        clicked = int(row["clicked"]) if "clicked" in row else None
        exclude_impr = 1 if clicked is not None else 0
        exclude_click = clicked if clicked is not None else 0

        user_code = self.stats["user_to_idx"].get(uid, 0)
        item_code = self.stats["item_to_idx"].get(iid, 0)

        item_impr, item_click, item_ctr = self._stat_values(
            self.stats["item_stats"], iid, exclude_impr, exclude_click
        )
        user_impr, user_click, user_ctr = self._stat_values(
            self.stats["user_stats"], uid, exclude_impr, exclude_click
        )
        item_time_impr, _, item_time_ctr = self._stat_values(
            self.stats["item_time_stats"], pair_key(iid, str(time_bucket)), exclude_impr, exclude_click
        )
        user_time_impr, _, user_time_ctr = self._stat_values(
            self.stats["user_time_stats"], pair_key(uid, str(time_bucket)), exclude_impr, exclude_click
        )
        item_day_impr, _, item_day_ctr = self._stat_values(
            self.stats["item_day_stats"], pair_key(iid, str(day_type)), exclude_impr, exclude_click
        )

        hist_len = len(history_items)
        hist_unique = len(set(history_items)) if hist_len else 0
        candidate_count = 0
        candidate_last_ix = -1
        hist_ctr_sum = 0.0
        hist_ctr_max = 0.0
        hist_item_impr_log_sum = 0.0

        hist_vec = np.zeros(self.factors, dtype=np.float32)
        hist_recent_vec = np.zeros(self.factors, dtype=np.float32)
        hist_vec_count = 0
        recent_weight_sum = 0.0

        for ix, hist_item in enumerate(history_items):
            h_exclude_impr = exclude_impr if hist_item == iid else 0
            h_exclude_click = exclude_click if hist_item == iid else 0
            h_impr, _, h_ctr = self._stat_values(
                self.stats["item_stats"], hist_item, h_exclude_impr, h_exclude_click
            )
            hist_ctr_sum += h_ctr
            hist_ctr_max = max(hist_ctr_max, h_ctr)
            hist_item_impr_log_sum += math.log1p(h_impr)

            if hist_item == iid:
                candidate_count += 1
                candidate_last_ix = ix

            h_code = self.stats["item_to_idx"].get(hist_item, 0)
            if h_code:
                emb = self.mf["item_emb"][h_code]
                hist_vec += emb
                hist_vec_count += 1
                # Later clicks in the history get larger weight.
                recency_weight = 0.75 ** (hist_len - 1 - ix)
                hist_recent_vec += recency_weight * emb
                recent_weight_sum += recency_weight

        if hist_vec_count:
            hist_vec /= float(hist_vec_count)
        if recent_weight_sum:
            hist_recent_vec /= float(recent_weight_sum)

        hist_ctr_mean = hist_ctr_sum / hist_len if hist_len else self.global_ctr
        hist_ctr_last = (
            self._stat_values(
                self.stats["item_stats"],
                history_items[-1],
                exclude_impr if history_items[-1] == iid else 0,
                exclude_click if history_items[-1] == iid else 0,
            )[2]
            if hist_len
            else self.global_ctr
        )
        hist_item_impr_mean_log = hist_item_impr_log_sum / hist_len if hist_len else 0.0

        if candidate_last_ix >= 0 and history_timestamps:
            candidate_age_hours = max((ts - int(history_timestamps[candidate_last_ix])) / 3600.0, 0.0)
            candidate_last_pos_from_end = hist_len - 1 - candidate_last_ix
        else:
            candidate_age_hours = 24.0 * 365.0
            candidate_last_pos_from_end = 99

        if history_timestamps:
            ages = [max((ts - int(h_ts)) / 3600.0, 0.0) for h_ts in history_timestamps]
            last_history_age_hours = ages[-1]
            mean_history_age_hours = sum(ages) / len(ages)
        else:
            last_history_age_hours = 24.0 * 365.0
            mean_history_age_hours = 24.0 * 365.0

        u_emb = self.mf["user_emb"][user_code]
        i_emb = self.mf["item_emb"][item_code]
        user_bias = float(self.mf["user_bias"][user_code])
        item_bias = float(self.mf["item_bias"][item_code])
        mf_dot = float(np.dot(u_emb, i_emb))
        mf_logit = float(self.mf["global_bias"] + user_bias + item_bias + mf_dot)
        mf_prob = sigmoid_scalar(mf_logit)

        i_norm = float(np.linalg.norm(i_emb))
        hist_norm = float(np.linalg.norm(hist_vec))
        hist_recent_norm = float(np.linalg.norm(hist_recent_vec))
        history_dot = float(np.dot(i_emb, hist_vec))
        history_recent_dot = float(np.dot(i_emb, hist_recent_vec))
        history_cosine = history_dot / (i_norm * hist_norm + 1e-6)
        history_recent_cosine = history_recent_dot / (i_norm * hist_recent_norm + 1e-6)

        feats = [
            float(user_code),
            float(item_code),
            float(time_bucket),
            float(day_type),
            math.log1p(item_impr),
            math.log1p(item_click),
            item_ctr,
            math.log1p(user_impr),
            math.log1p(user_click),
            user_ctr,
            math.log1p(item_time_impr),
            item_time_ctr,
            math.log1p(user_time_impr),
            user_time_ctr,
            math.log1p(item_day_impr),
            item_day_ctr,
            float(hist_len),
            float(hist_unique),
            hist_ctr_mean,
            hist_ctr_max,
            hist_ctr_last,
            hist_item_impr_mean_log,
            1.0 if candidate_count else 0.0,
            float(candidate_count),
            float(candidate_last_pos_from_end),
            math.log1p(candidate_age_hours),
            math.log1p(last_history_age_hours),
            math.log1p(mean_history_age_hours),
            mf_logit,
            mf_prob,
            mf_dot,
            user_bias,
            item_bias,
            history_dot,
            history_cosine,
            history_recent_dot,
            history_recent_cosine,
        ]

        if self.include_embeddings:
            feats.extend(float(v) for v in u_emb)
            feats.extend(float(v) for v in i_emb)
            feats.extend(float(v) for v in hist_vec)
            feats.extend(float(v) for v in hist_recent_vec)
        return feats


def build_lgbm_matrix(
    np,
    train_path: str,
    builder: FeatureBuilder,
    total_rows: int,
    max_rows: int,
    seed: int,
    valid_fraction: float,
) -> Tuple:
    start = time.time()
    cap = total_rows if max_rows <= 0 else min(max_rows, total_rows)
    sample_prob = 1.0 if max_rows <= 0 else min(1.0, cap / max(total_rows, 1))
    n_features = builder.n_features()

    log(f"pass 3/4: building LightGBM matrix with up to {cap:,} sampled rows ...")
    X = np.empty((cap, n_features), dtype=np.float32)
    y = np.empty(cap, dtype=np.float32)
    is_valid = np.zeros(cap, dtype=bool)

    n = 0
    last_event = None
    last_hist_items: List[str] = []
    last_hist_ts: List[int] = []

    with open(train_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for seen, row in enumerate(reader, start=1):
            if sample_prob < 1.0 and stable_unit_hash(row["rowid"], seed) > sample_prob:
                continue
            if n >= cap:
                continue

            event_id = row["event_id"]
            if event_id != last_event:
                last_event = event_id
                last_hist_items = parse_json_array(row["history_item_ids"])
                last_hist_ts = parse_json_array(row["history_timestamps"])

            X[n, :] = builder.build(row, last_hist_items, last_hist_ts)
            y[n] = float(row["clicked"])
            if valid_fraction > 0.0:
                is_valid[n] = stable_unit_hash(event_id, seed + 10_000) < valid_fraction
            n += 1

            if seen % 1_000_000 == 0:
                log(f"  built {n:,} / {seen:,} scanned rows")

    log(f"  built {n:,} LightGBM rows ({elapsed(start)})")
    return X[:n, :], y[:n], is_valid[:n]


def train_lightgbm(np, lgb, X, y, is_valid, builder: FeatureBuilder, args):
    start = time.time()
    valid_count = int(is_valid.sum())

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "min_data_in_leaf": args.min_data_in_leaf,
        "feature_fraction": args.feature_fraction,
        "bagging_fraction": args.bagging_fraction,
        "bagging_freq": 1,
        "lambda_l2": args.lambda_l2,
        "max_bin": args.max_bin,
        "verbosity": -1,
        "seed": args.seed,
        "num_threads": args.num_threads,
    }

    categorical = builder.categorical_indices()

    if valid_count and valid_count < len(y):
        train_mask = ~is_valid
        log(
            f"training LightGBM with {int(train_mask.sum()):,} train rows and "
            f"{valid_count:,} event-hash validation rows ..."
        )
        train_set = lgb.Dataset(
            X[train_mask],
            label=y[train_mask],
            feature_name=builder.names,
            categorical_feature=categorical,
            free_raw_data=False,
        )
        valid_set = lgb.Dataset(
            X[is_valid],
            label=y[is_valid],
            feature_name=builder.names,
            categorical_feature=categorical,
            reference=train_set,
            free_raw_data=False,
        )
        booster = lgb.train(
            params,
            train_set,
            num_boost_round=args.num_boost_round,
            valid_sets=[train_set, valid_set],
            valid_names=["train", "valid"],
            callbacks=[
                lgb.early_stopping(args.early_stopping_rounds),
                lgb.log_evaluation(period=50),
            ],
        )
        best_round = booster.best_iteration if booster.best_iteration else args.num_boost_round
        if not args.no_refit_full_after_valid:
            log(f"refitting LightGBM on all {len(y):,} sampled rows for {best_round:,} rounds ...")
            full_set = lgb.Dataset(
                X,
                label=y,
                feature_name=builder.names,
                categorical_feature=categorical,
                free_raw_data=False,
            )
            booster = lgb.train(params, full_set, num_boost_round=best_round)
    else:
        log(f"training LightGBM with {len(y):,} rows ...")
        train_set = lgb.Dataset(
            X,
            label=y,
            feature_name=builder.names,
            categorical_feature=categorical,
            free_raw_data=False,
        )
        booster = lgb.train(params, train_set, num_boost_round=args.num_boost_round)

    log(f"  LightGBM training done ({elapsed(start)})")
    return booster


def write_predictions(np, evaluation_path: str, out_path: str, builder: FeatureBuilder, booster, chunk_size: int) -> None:
    start = time.time()
    n_features = builder.n_features()
    X_chunk = np.empty((chunk_size, n_features), dtype=np.float32)
    rowids: List[str] = []
    total = 0

    log("pass 4/4: scoring evaluation.csv and writing submission ...")
    with open(evaluation_path, newline="", encoding="utf-8") as f, open(
        out_path, "w", newline="", encoding="utf-8"
    ) as out_f:
        reader = csv.DictReader(f)
        writer = csv.writer(out_f)
        writer.writerow(["rowid", "predicted_probability"])

        last_event = None
        last_hist_items: List[str] = []
        last_hist_ts: List[int] = []

        def flush() -> None:
            nonlocal rowids, total
            if not rowids:
                return
            best_iteration = booster.best_iteration if booster.best_iteration else None
            preds = booster.predict(X_chunk[: len(rowids), :], num_iteration=best_iteration)
            for rid, pred in zip(rowids, preds):
                pred = min(max(float(pred), 0.0), 1.0)
                writer.writerow([rid, f"{pred:.8g}"])
            total += len(rowids)
            rowids = []

        for row in reader:
            event_id = row["event_id"]
            if event_id != last_event:
                last_event = event_id
                last_hist_items = parse_json_array(row["history_item_ids"])
                last_hist_ts = parse_json_array(row["history_timestamps"])

            if len(rowids) >= chunk_size:
                flush()

            X_chunk[len(rowids), :] = builder.build(row, last_hist_items, last_hist_ts)
            rowids.append(row["rowid"])

            if (total + len(rowids)) % 100_000 == 0:
                log(f"  scored {total + len(rowids):,} rows")

        flush()

    log(f"done: wrote {total:,} predictions to {out_path} ({elapsed(start)})")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", default="train.csv")
    p.add_argument("--evaluation", default="evaluation.csv")
    p.add_argument("--out", default="submission_lgbm_mf_history.csv")
    p.add_argument("--model-out", default="lgbm_mf_history_model.txt")

    p.add_argument("--max-train-rows", type=int, default=1_500_000)
    p.add_argument("--mf-sample-rows", type=int, default=2_000_000)
    p.add_argument("--valid-fraction", type=float, default=0.08)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--num-threads", type=int, default=0)

    p.add_argument("--smoothing", type=float, default=50.0)
    p.add_argument("--mf-factors", type=int, default=16)
    p.add_argument("--mf-epochs", type=int, default=2)
    p.add_argument("--mf-lr", type=float, default=0.015)
    p.add_argument("--mf-reg", type=float, default=1e-5)
    p.add_argument("--mf-batch-size", type=int, default=8192)
    p.add_argument("--mf-pos-weight", type=float, default=4.0)
    p.add_argument("--no-embedding-columns", action="store_true")

    p.add_argument("--num-boost-round", type=int, default=700)
    p.add_argument("--early-stopping-rounds", type=int, default=80)
    p.add_argument("--no-refit-full-after-valid", action="store_true")
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--num-leaves", type=int, default=127)
    p.add_argument("--min-data-in-leaf", type=int, default=200)
    p.add_argument("--feature-fraction", type=float, default=0.9)
    p.add_argument("--bagging-fraction", type=float, default=0.85)
    p.add_argument("--lambda-l2", type=float, default=2.0)
    p.add_argument("--max-bin", type=int, default=255)
    p.add_argument("--predict-chunk-size", type=int, default=100_000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    np, lgb = require_ml_deps()

    if not os.path.exists(args.train):
        raise SystemExit(f"train file not found: {args.train}")
    if not os.path.exists(args.evaluation):
        raise SystemExit(f"evaluation file not found: {args.evaluation}")

    stats = build_label_stats(args.train)
    mf_user_idx, mf_item_idx, mf_y = collect_mf_samples(
        np,
        args.train,
        stats,
        max_rows=args.mf_sample_rows,
        seed=args.seed + 1,
    )
    mf = train_logistic_mf(
        np,
        mf_user_idx,
        mf_item_idx,
        mf_y,
        n_users=len(stats["user_to_idx"]),
        n_items=len(stats["item_to_idx"]),
        factors=args.mf_factors,
        epochs=args.mf_epochs,
        lr=args.mf_lr,
        reg=args.mf_reg,
        batch_size=args.mf_batch_size,
        seed=args.seed + 2,
        global_ctr=stats["global_ctr"],
        pos_weight=args.mf_pos_weight,
    )

    builder = FeatureBuilder(
        np,
        stats,
        mf,
        smoothing=args.smoothing,
        include_embeddings=not args.no_embedding_columns,
    )
    X, y, is_valid = build_lgbm_matrix(
        np,
        args.train,
        builder,
        total_rows=stats["total_rows"],
        max_rows=args.max_train_rows,
        seed=args.seed + 3,
        valid_fraction=args.valid_fraction,
    )
    booster = train_lightgbm(np, lgb, X, y, is_valid, builder, args)
    booster.save_model(args.model_out)
    log(f"saved model to {args.model_out}")

    write_predictions(
        np,
        args.evaluation,
        args.out,
        builder,
        booster,
        chunk_size=args.predict_chunk_size,
    )


if __name__ == "__main__":
    main()
