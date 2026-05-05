"""
PRISM CLI: command-line interface for reliability evaluation.

Usage:
    prism evaluate --train-features X_train.npy --train-labels y_train.npy \
                   --test-features X_test.npy --test-labels y_test.npy \
                   --dataset pcam --model-name MyModel

    prism compare --results my_model_results.csv \
                  --results-dir /path/to/prism/results \
                  --dataset pcam --fraction 0.1
"""

import argparse
import numpy as np
import pandas as pd
import sys

from .evaluator import PRISMEvaluator


def cmd_evaluate(args):
    print(f"Loading embeddings...")
    X_train = np.load(args.train_features)
    y_train = np.load(args.train_labels)
    X_test  = np.load(args.test_features)
    y_test  = np.load(args.test_labels)

    print(f"Train: {X_train.shape} | Test: {X_test.shape}")

    evaluator = PRISMEvaluator()
    results = evaluator.evaluate(
        train_features=X_train,
        train_labels=y_train,
        test_features=X_test,
        test_labels=y_test,
        dataset=args.dataset,
        model_name=args.model_name,
    )

    summary = results.groupby("fraction")[["auroc", "ece", "brier", "ece_scaled"]].mean().round(4)
    print(f"\n=== {args.model_name} on {args.dataset} ===")
    print(summary.to_string())

    out = args.output or f"{args.model_name.lower()}_{args.dataset}_results.csv"
    results.to_csv(out, index=False)
    print(f"\nSaved: {out}")

    return results


def cmd_compare(args):
    results = pd.read_csv(args.results)
    evaluator = PRISMEvaluator(results_dir=args.results_dir)

    fraction = float(args.fraction) if args.fraction else None
    comparison = evaluator.compare(results, dataset=args.dataset, fraction=fraction)

    print(f"\n=== PRISM CRI Comparison on {args.dataset} ===")
    print(comparison.to_string(index=False))

    if args.output:
        comparison.to_csv(args.output, index=False)
        print(f"\nSaved: {args.output}")


def main():
    parser = argparse.ArgumentParser(
        prog="prism",
        description="PRISM: Pathology Reliability In Scarce-label Medicine benchmark",
    )
    sub = parser.add_subparsers(dest="command")

    # evaluate
    p_eval = sub.add_parser("evaluate", help="Evaluate model embeddings")
    p_eval.add_argument("--train-features", required=True)
    p_eval.add_argument("--train-labels",   required=True)
    p_eval.add_argument("--test-features",  required=True)
    p_eval.add_argument("--test-labels",    required=True)
    p_eval.add_argument("--dataset",        required=True, help="e.g. pcam, mhist, crc")
    p_eval.add_argument("--model-name",     default="CustomModel")
    p_eval.add_argument("--output",         default=None)

    # compare
    p_cmp = sub.add_parser("compare", help="Compare against PRISM reference models")
    p_cmp.add_argument("--results",     required=True, help="CSV from evaluate command")
    p_cmp.add_argument("--results-dir", required=True, help="Path to PRISM results directory")
    p_cmp.add_argument("--dataset",     required=True)
    p_cmp.add_argument("--fraction",    default=None, help="e.g. 0.1 for 10% labels")
    p_cmp.add_argument("--output",      default=None)

    args = parser.parse_args()

    if args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "compare":
        cmd_compare(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
