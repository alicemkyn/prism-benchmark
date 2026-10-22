"""
PRISM CLI: command-line interface for reliability evaluation.

Usage:
    prism evaluate --train-features X_train.npy --train-labels y_train.npy \
                   --test-features X_test.npy --test-labels y_test.npy \
                   --val-features X_val.npy --val-labels y_val.npy \
                   --dataset pcam --model-name MyModel

    prism compare --results mymodel_pcam_results.csv \
                  --dataset pcam --fraction 0.1 [--results-dir results_v2]

    prism reference --fraction 1.0 [--results-dir results_v2]

``--results-dir`` defaults to the results_v2 reference CSVs bundled with the
package. ``python -m prism_bench`` is equivalent to ``prism``.
"""

import argparse
import sys

import numpy as np
import pandas as pd

from . import __version__
from .evaluator import PRISMEvaluator, LABEL_FRACTIONS, SEEDS

SUMMARY_COLUMNS = ["auroc", "f1_macro", "brier", "ece_fixed",
                   "temperature", "ece_scaled_fixed", "degeneracy_share"]


def _floats(s):
    return [float(x) for x in s.split(",") if x.strip()]


def _ints(s):
    return [int(x) for x in s.split(",") if x.strip()]


def cmd_evaluate(args):
    print("Loading embeddings...")
    X_train = np.load(args.train_features)
    y_train = np.load(args.train_labels)
    X_test = np.load(args.test_features)
    y_test = np.load(args.test_labels)

    if (args.val_features is None) != (args.val_labels is None):
        sys.exit("error: --val-features and --val-labels must be given together")
    X_val = np.load(args.val_features) if args.val_features else None
    y_val = np.load(args.val_labels) if args.val_labels else None

    msg = f"Train: {X_train.shape} | Test: {X_test.shape}"
    if X_val is not None:
        msg += f" | Val: {X_val.shape}"
    print(msg)

    evaluator = PRISMEvaluator()
    results = evaluator.evaluate(
        train_features=X_train,
        train_labels=y_train,
        test_features=X_test,
        test_labels=y_test,
        dataset=args.dataset,
        model_name=args.model_name,
        val_features=X_val,
        val_labels=y_val,
        label_fractions=args.label_fractions,
        seeds=args.seeds,
        C=args.C,
    )

    cols = [c for c in SUMMARY_COLUMNS if c in results.columns]
    summary = results.groupby("fraction")[cols].mean().round(4)
    print(f"\n=== {args.model_name} on {args.dataset} ===")
    print(summary.to_string())

    out = args.output or f"{args.model_name.lower()}_{args.dataset}_results.csv"
    results.to_csv(out, index=False)
    print(f"\nSaved: {out}")
    return results


def cmd_compare(args):
    results = pd.read_csv(args.results)
    evaluator = PRISMEvaluator(results_dir=args.results_dir)

    fraction = float(args.fraction) if args.fraction is not None else None
    comparison = evaluator.compare(
        results, dataset=args.dataset, fraction=fraction,
        aggregation=args.aggregation,
        custom_ood_stability=args.ood_stability,
    )

    print(f"\n=== PRISM Composite Reliability Index (CRI) on {args.dataset} ===")
    print(f"reference: {evaluator.reference_path} ({evaluator.reference_layout})")
    print(comparison.to_string(index=False))

    if args.output:
        comparison.to_csv(args.output, index=False)
        print(f"\nSaved: {args.output}")
    return comparison


def cmd_reference(args):
    evaluator = PRISMEvaluator(results_dir=args.results_dir)
    tab = evaluator.reference_cri(fraction=float(args.fraction))
    print(f"\n=== Reference CRI, pooled over six datasets, fraction={args.fraction} ===")
    print(f"reference: {evaluator.reference_path} ({evaluator.reference_layout})")
    print(tab.round(4).to_string(index=False))
    if args.output:
        tab.to_csv(args.output, index=False)
        print(f"\nSaved: {args.output}")
    return tab


def build_parser():
    parser = argparse.ArgumentParser(
        prog="prism",
        description="PRISM: Pathology Reliability In Scarce-label Medicine. "
                    "A tile-level benchmark of pathology foundation models.",
    )
    parser.add_argument("--version", action="version",
                        version=f"prism-bench {__version__}")
    sub = parser.add_subparsers(dest="command")

    # evaluate
    p_eval = sub.add_parser("evaluate", help="Evaluate model embeddings")
    p_eval.add_argument("--train-features", required=True, help=".npy (N, D)")
    p_eval.add_argument("--train-labels", required=True, help=".npy (N,)")
    p_eval.add_argument("--test-features", required=True, help=".npy (M, D)")
    p_eval.add_argument("--test-labels", required=True, help=".npy (M,)")
    p_eval.add_argument("--val-features", default=None,
                        help=".npy (V, D); held-out split for temperature "
                             "scaling (required for ECE_scaled and CRI)")
    p_eval.add_argument("--val-labels", default=None, help=".npy (V,)")
    p_eval.add_argument("--dataset", required=True,
                        help="e.g. pcam, mhist, crc, bracs, lunghist700, "
                             "spider_breast (any name for evaluate)")
    p_eval.add_argument("--model-name", default="CustomModel")
    p_eval.add_argument("--label-fractions", type=_floats, default=None,
                        help=f"comma-separated, default "
                             f"{','.join(str(f) for f in LABEL_FRACTIONS)}")
    p_eval.add_argument("--seeds", type=_ints, default=None,
                        help=f"comma-separated, default "
                             f"{','.join(str(s) for s in SEEDS)}")
    p_eval.add_argument("--C", type=float, default=1.0,
                        help="logistic-regression inverse regularisation "
                             "(PRISM protocol: 1.0)")
    p_eval.add_argument("--output", default=None)

    # compare
    p_cmp = sub.add_parser("compare", help="Compare against PRISM reference models")
    p_cmp.add_argument("--results", required=True, help="CSV from `prism evaluate`")
    p_cmp.add_argument("--results-dir", default=None,
                       help="results_v2 directory (default: copy bundled "
                            "with the package); a legacy results/ directory "
                            "is accepted with a warning")
    p_cmp.add_argument("--dataset", required=True,
                       help="one of the six PRISM datasets, or 'all'")
    p_cmp.add_argument("--fraction", default=None, help="e.g. 0.1 for 10%% labels")
    p_cmp.add_argument("--aggregation", default="multiplicative",
                       choices=["multiplicative", "arithmetic", "geometric",
                                "worst_axis"])
    p_cmp.add_argument("--ood-stability", type=float, default=None,
                       help="measured OOD_Stability of your model "
                            "(default: reference mean, flagged as imputed)")
    p_cmp.add_argument("--output", default=None)

    # reference
    p_ref = sub.add_parser("reference",
                           help="Pooled reference CRI of the eight FMs")
    p_ref.add_argument("--fraction", default="1.0")
    p_ref.add_argument("--results-dir", default=None)
    p_ref.add_argument("--output", default=None)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "compare":
        cmd_compare(args)
    elif args.command == "reference":
        cmd_reference(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
