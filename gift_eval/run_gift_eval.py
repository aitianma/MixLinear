"""
Run MixLinear on the full GIFT-Eval benchmark (97 dataset/freq/term configs)
and produce a `results/MixLinear/all_results.csv` file in the format the
GIFT-Eval leaderboard expects (see gift-eval/README.md, "Submitting your
results").

MixLinear is trained from scratch, separately, for each of the 97 configs
(it is a "deep-learning" model in gift-eval's taxonomy, not a pretrained /
zero-shot foundation model) -- same convention as the other from-scratch
baselines already on the leaderboard (DLinear, PatchTST, TiDE, N-BEATS, ...).

Usage (sharded across GPUs, run one process per shard in parallel):
    python run_gift_eval.py --shard 0 --num-shards 3 --device cuda:0
    python run_gift_eval.py --shard 1 --num-shards 3 --device cuda:1
    python run_gift_eval.py --shard 2 --num-shards 3 --device cuda:2

Then merge the per-shard CSVs with merge_results.py.
"""
import argparse
import csv
import json
import logging
import math
import os
import sys
import time
import traceback

import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gluonts_wrapper import MixLinearEstimator  # noqa: E402

from gluonts.ev.metrics import (
    MSE, MAE, MASE, MAPE, SMAPE, MSIS, RMSE, NRMSE, ND, MeanWeightedSumQuantileLoss,
)
from gluonts.model import evaluate_model
from gluonts.time_feature import get_seasonality
from gift_eval.data import Dataset

logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

PRETTY_NAMES = {
    "saugeenday": "saugeen",
    "temperature_rain_with_missing": "temperature_rain",
    "kdd_cup_2018_with_missing": "kdd_cup_2018",
    "car_parts_with_missing": "car_parts",
}

MODEL_NAME = "MixLinear"

METRICS = [
    MSE(forecast_type="mean"),
    MSE(forecast_type=0.5),
    MAE(),
    MASE(),
    MAPE(),
    SMAPE(),
    MSIS(),
    RMSE(),
    NRMSE(),
    ND(),
    MeanWeightedSumQuantileLoss(quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
]

CSV_HEADER = [
    "dataset", "model",
    "eval_metrics/MSE[mean]", "eval_metrics/MSE[0.5]", "eval_metrics/MAE[0.5]",
    "eval_metrics/MASE[0.5]", "eval_metrics/MAPE[0.5]", "eval_metrics/sMAPE[0.5]",
    "eval_metrics/MSIS", "eval_metrics/RMSE[mean]", "eval_metrics/NRMSE[mean]",
    "eval_metrics/ND[0.5]", "eval_metrics/mean_weighted_sum_quantile_loss",
    "domain", "num_variates",
]


def choose_hparams(freq: str, pred_len: int):
    """Pick period_len / context_length / lpf that (a) respect MixLinear's
    hard shape constraints (context_length % period_len == 0, and
    lpf <= ceil(context_length / period_len)) and (b) scale sensibly with
    the dataset's native seasonality across every frequency in gift-eval.

    period_len must be even: MixLinear's conv1d uses kernel_size=period_len+1
    with padding=period_len//2, which only preserves the sequence length
    (as the reference implementation assumes) when period_len is even --
    the reference scripts always hardcode period_len=24 and never hit this.
    Frequencies with seasonality 1 (D/W/A/...) are common in gift-eval, so
    we round up to the nearest even value >= 2 instead of reusing season=1.
    """
    season = get_seasonality(freq)
    period_len = max(2, min(season, 60))
    if period_len % 2 == 1:
        period_len += 1
    seg_num_y = math.ceil(pred_len / period_len)
    ctx_periods = max(seg_num_y * 3, 8)
    ctx_periods = min(ctx_periods, 60)
    context_length = period_len * ctx_periods
    lpf = max(1, min(5, ctx_periods))
    return period_len, context_length, lpf


def build_config_list(dataset_lists_path: str):
    lists = json.load(open(dataset_lists_path))
    short_datasets = lists["short_datasets"]
    med_long_datasets = set(lists["med_long_datasets"])
    all_datasets = sorted(set(short_datasets) | med_long_datasets)

    configs = []
    for ds_name in all_datasets:
        for term in ["short", "medium", "long"]:
            if term != "short" and ds_name not in med_long_datasets:
                continue
            configs.append((ds_name, term))
    return configs


def resolve_ds_key_freq(ds_name: str, dataset_properties_map: dict):
    if "/" in ds_name:
        ds_key, ds_freq = ds_name.split("/")
        ds_key = PRETTY_NAMES.get(ds_key.lower(), ds_key.lower())
    else:
        ds_key = PRETTY_NAMES.get(ds_name.lower(), ds_name.lower())
        ds_freq = dataset_properties_map[ds_key]["frequency"]
    return ds_key, ds_freq


def run_one(ds_name: str, term: str, dataset_properties_map: dict, device: str, epochs: int, patience: int, lightning_dir: str):
    ds_key, ds_freq = resolve_ds_key_freq(ds_name, dataset_properties_map)
    ds_config = f"{ds_key}/{ds_freq}/{term}"

    to_univariate = Dataset(name=ds_name, term=term, to_univariate=False).target_dim != 1
    dataset = Dataset(name=ds_name, term=term, to_univariate=to_univariate)
    season_length = get_seasonality(dataset.freq)

    period_len, context_length, lpf = choose_hparams(dataset.freq, dataset.prediction_length)

    estimator = MixLinearEstimator(
        prediction_length=dataset.prediction_length,
        context_length=context_length,
        period_len=period_len,
        lpf=lpf,
        alpha=0.5,
        lr=1e-2,
        batch_size=64,
        num_batches_per_epoch=100,
        trainer_kwargs={
            "max_epochs": epochs,
            "accelerator": "gpu" if device.startswith("cuda") else "cpu",
            "devices": [int(device.split(":")[1])] if device.startswith("cuda") else 1,
            "enable_progress_bar": False,
            "enable_model_summary": False,
            "logger": False,
            "default_root_dir": lightning_dir,
            "callbacks": [EarlyStopping(monitor="val_loss", patience=patience, mode="min")],
        },
    )

    predictor = estimator.train(
        training_data=dataset.training_dataset,
        validation_data=dataset.validation_dataset,
    )

    res = evaluate_model(
        predictor,
        test_data=dataset.test_data,
        metrics=METRICS,
        batch_size=512,
        axis=None,
        mask_invalid_label=True,
        allow_nan_forecast=False,
        seasonality=season_length,
    )

    row = [
        ds_config, MODEL_NAME,
        res["MSE[mean]"][0], res["MSE[0.5]"][0], res["MAE[0.5]"][0],
        res["MASE[0.5]"][0], res["MAPE[0.5]"][0], res["sMAPE[0.5]"][0],
        res["MSIS"][0], res["RMSE[mean]"][0], res["NRMSE[mean]"][0],
        res["ND[0.5]"][0], res["mean_weighted_sum_quantile_loss"][0],
        dataset_properties_map[ds_key]["domain"], dataset_properties_map[ds_key]["num_variates"],
    ]
    return row, (period_len, context_length, lpf)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gift-eval-repo", required=True, help="path to cloned gift-eval repo (for dataset_properties.json)")
    parser.add_argument("--dataset-lists", required=True, help="path to dataset_lists.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--only", default=None, help="comma-separated ds_name:term for a quick smoke test")
    args = parser.parse_args()

    dataset_properties_map = json.load(open(os.path.join(args.gift_eval_repo, "notebooks", "dataset_properties.json")))

    if args.only:
        configs = []
        for item in args.only.split(","):
            ds_name, term = item.split(":")
            configs.append((ds_name, term))
    else:
        configs = build_config_list(args.dataset_lists)
        configs = [c for i, c in enumerate(configs) if i % args.num_shards == args.shard]

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, f"shard_{args.shard}.csv")
    log_path = os.path.join(args.output_dir, f"shard_{args.shard}.log")

    write_header = not os.path.exists(csv_path)
    done = set()
    if not write_header:
        with open(csv_path) as f:
            for r in csv.DictReader(f):
                done.add(r["dataset"])

    csv_file = open(csv_path, "a", newline="")
    writer = csv.writer(csv_file)
    if write_header:
        writer.writerow(CSV_HEADER)
        csv_file.flush()

    log_file = open(log_path, "a")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    log(f"shard {args.shard}/{args.num_shards} device={args.device} configs={len(configs)}")

    for ds_name, term in configs:
        ds_key, ds_freq = resolve_ds_key_freq(ds_name, dataset_properties_map)
        ds_config = f"{ds_key}/{ds_freq}/{term}"
        if ds_config in done:
            log(f"SKIP (already done) {ds_config}")
            continue
        t0 = time.time()
        try:
            row, hparams = run_one(ds_name, term, dataset_properties_map, args.device, args.epochs, args.patience, os.path.join(args.output_dir, "lightning"))
            writer.writerow(row)
            csv_file.flush()
            dt = time.time() - t0
            log(f"OK   {ds_config} period_len/ctx/lpf={hparams} MASE={row[5]:.4f} sMAPE={row[6]:.4f} ({dt:.1f}s)")
        except Exception as e:
            dt = time.time() - t0
            log(f"FAIL {ds_config} ({dt:.1f}s): {e}")
            log(traceback.format_exc())
        finally:
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()

    csv_file.close()
    log_file.close()


if __name__ == "__main__":
    main()
