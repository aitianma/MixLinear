# MixLinear on GIFT-Eval

Replication code for MixLinear's submission to the [GIFT-Eval leaderboard](https://huggingface.co/spaces/Salesforce/GIFT-Eval)
([benchmark repo](https://github.com/SalesforceAIResearch/gift-eval)).

MixLinear is trained from scratch, separately, on each of the 97 dataset/frequency/term
configurations in GIFT-Eval (it is a `deep-learning` submission, not a pretrained/zero-shot
foundation model) — the same convention used by the other from-scratch baselines already on the
leaderboard (DLinear, PatchTST, TiDE, N-BEATS, ...). No test data is used for training; each config
trains on `Dataset.training_dataset` and early-stops on `Dataset.validation_dataset`, both of which
gift-eval constructs from the pre-test portion of each series.

## Files

- `model.py` — `MixLinearCore`, the architecture from the main MixLinear repo (`models/MixLinear.py`),
  unchanged except for removing debug prints.
- `gluonts_wrapper.py` — a GluonTS `PyTorchLightningEstimator` for MixLinear, built the same way
  GluonTS's own `SimpleFeedForwardEstimator` is (network → `LightningModule` → `Estimator` wiring
  transformation/instance-splitting/predictor construction). MixLinear outputs a single point
  forecast, so it's exposed to GluonTS's `evaluate_model` as a 1-sample `SampleForecast`.
  MixLinear has no cross-channel parameters, so multivariate datasets are converted to independent
  univariate series (`to_univariate=True`) — mathematically equivalent to running it jointly over
  all channels, and how gift-eval's own example baselines (e.g. `feedforward.ipynb`) handle
  multivariate data too.
- `run_gift_eval.py` — iterates all 97 gift-eval configs (or a `--shard`/`--num-shards` subset, for
  running in parallel across multiple GPUs), trains + evaluates MixLinear on each, and writes
  `results/MixLinear/all_results.csv` in the format gift-eval's leaderboard expects.

## Hyperparameter choice across frequencies

MixLinear's `period_len` must evenly divide the context length, and its low-pass filter width
`lpf` must not exceed the number of time-domain segments. `choose_hparams()` in `run_gift_eval.py`
derives `period_len` from each dataset's native seasonality (`gluonts.time_feature.get_seasonality`),
rounded up to an even number — required because MixLinear's `conv1d` (kernel size
`period_len + 1`, padding `period_len // 2`) only preserves sequence length for even `period_len`;
the reference training scripts always hardcode `period_len=24` and never hit this edge case, which
GIFT-Eval's daily/weekly/yearly-frequency datasets do. `alpha=0.5` (equal time/frequency mix) is
used everywhere; per-dataset tuning (as in `scripts/MixLinear/*_best.sh`) was not run for this
submission.

## Reproducing

```bash
pip install -e ".[baseline]"   # from a clone of SalesforceAIResearch/gift-eval
hf download Salesforce/GiftEval --repo-type=dataset --local-dir <PATH>
export GIFT_EVAL=<PATH>

python run_gift_eval.py \
  --gift-eval-repo <path to gift-eval clone> \
  --dataset-lists dataset_lists.json \
  --output-dir results/MixLinear \
  --device cuda:0 --epochs 30 --patience 5
```

Pass `--shard i --num-shards N` (on N processes, each with a different `--device`) to split the 97
configs across multiple GPUs.
