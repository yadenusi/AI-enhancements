"""
SageMaker training entry point for the predictive maintenance classifier.

Designed to run inside the SageMaker XGBoost built in container (framework
mode) as part of a SageMaker Pipeline that is triggered weekly by
EventBridge. Reads curated feature data written by the Glue ETL job and
writes a labeled failure classifier to the model directory SageMaker
expects.

Expected input CSV schema (no header), one row per instance-hour:
  label, cpu_util_mean, cpu_util_max, cpu_util_std, disk_io_mean,
  disk_io_max, network_out_mean, network_out_std,
  status_check_failed_count, memory_util_mean

label is 1 if the instance experienced a failure or forced replacement
within the following 4 hours, 0 otherwise. Labels are backfilled from
AWS Health events and EC2 status check history during feature engineering.
"""

import argparse
import os

import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--eta", type=float, default=0.2)
    parser.add_argument("--num-round", type=int, default=150)
    parser.add_argument("--scale-pos-weight", type=float, default=8.0,
                         help="Failures are rare; weight the positive class")
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    return parser.parse_args()


def load_training_data(train_dir):
    files = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith(".csv")]
    frames = [pd.read_csv(f, header=None) for f in files]
    data = pd.concat(frames, ignore_index=True)
    y = data.iloc[:, 0]
    x = data.iloc[:, 1:]
    return x, y


def main():
    args = parse_args()
    x, y = load_training_data(args.train)
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    dtrain = xgb.DMatrix(x_train, label=y_train)
    dval = xgb.DMatrix(x_val, label=y_val)

    params = {
        "max_depth": args.max_depth,
        "eta": args.eta,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "scale_pos_weight": args.scale_pos_weight,
    }

    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=args.num_round,
        evals=[(dtrain, "train"), (dval, "validation")],
        early_stopping_rounds=15,
        verbose_eval=10,
    )

    val_pred = booster.predict(dval)
    auc = roc_auc_score(y_val, val_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_val, (val_pred >= 0.5).astype(int), average="binary", zero_division=0
    )
    print(f"validation_auc={auc:.4f} precision={precision:.4f} recall={recall:.4f} f1={f1:.4f}")

    model_path = os.path.join(args.model_dir, "xgboost-model")
    booster.save_model(model_path)


if __name__ == "__main__":
    main()
