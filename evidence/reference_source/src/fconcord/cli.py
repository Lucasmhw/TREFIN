from __future__ import annotations

import argparse
from pathlib import Path

from fconcord.ablation import run_ablation_protocol, run_retrained_ablation
from fconcord.ablation_audit import audit_ablation_protocol
from fconcord.audit import audit_completed_run
from fconcord.benchmark import run_benchmark
from fconcord.benchmark_audit import audit_benchmark_protocol
from fconcord.benchmark_protocol import run_benchmark_protocol
from fconcord.comparison import build_comparison_report
from fconcord.comparison_visualization import generate_comparison_figures
from fconcord.config import load_config
from fconcord.experiment import regenerate_figures, run_experiment
from fconcord.foundation import run_foundation_benchmark, run_foundation_protocol
from fconcord.protocol import run_protocol
from fconcord.squared_error_report import build_squared_error_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fconcord")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="train, reload, test, audit and plot")
    run.add_argument("--config", required=True, type=Path)
    run.add_argument("--run-dir", type=Path, default=None)
    figures = subparsers.add_parser("figures", help="regenerate all figures from a completed run")
    figures.add_argument("--run-dir", required=True, type=Path)
    audit = subparsers.add_parser("audit", help="validate a completed run and its artifacts")
    audit.add_argument("--run-dir", required=True, type=Path)
    protocol = subparsers.add_parser(
        "protocol", help="execute preregistered rolling folds and seeds"
    )
    protocol.add_argument("--config", required=True, type=Path)
    protocol.add_argument("--output-dir", type=Path, default=None)
    squared_error = subparsers.add_parser(
        "squared-error-report",
        help="derive MSE and OOS R2 from frozen protocol test predictions",
    )
    squared_error.add_argument("--protocol-dir", required=True, type=Path)
    squared_error.add_argument("--output", type=Path, default=None)
    benchmark = subparsers.add_parser(
        "benchmark", help="run equal-budget baselines against a completed F-CONCORD run"
    )
    benchmark.add_argument("--config", required=True, type=Path)
    benchmark.add_argument("--reference-run", required=True, type=Path)
    benchmark.add_argument("--output-dir", required=True, type=Path)
    benchmark_protocol = subparsers.add_parser(
        "benchmark-protocol", help="run baselines on every reference fold and seed"
    )
    benchmark_protocol.add_argument("--config", required=True, type=Path)
    benchmark_protocol.add_argument("--reference-protocol", required=True, type=Path)
    benchmark_protocol.add_argument("--output-dir", required=True, type=Path)
    ablation = subparsers.add_parser(
        "ablation", help="retrain one structural ablation against a reference run"
    )
    ablation.add_argument("--reference-run", required=True, type=Path)
    ablation.add_argument("--variant", required=True)
    ablation.add_argument("--output-dir", required=True, type=Path)
    ablation_protocol = subparsers.add_parser(
        "ablation-protocol", help="retrain all structural ablations across folds/seeds"
    )
    ablation_protocol.add_argument("--reference-protocol", required=True, type=Path)
    ablation_protocol.add_argument("--output-dir", required=True, type=Path)
    foundation = subparsers.add_parser(
        "foundation", help="run official zero-shot foundation-model baselines"
    )
    foundation.add_argument("--config", required=True, type=Path)
    foundation.add_argument("--reference-run", required=True, type=Path)
    foundation.add_argument("--output-dir", required=True, type=Path)
    foundation_protocol = subparsers.add_parser(
        "foundation-protocol", help="run official foundations once per registered fold"
    )
    foundation_protocol.add_argument("--config", required=True, type=Path)
    foundation_protocol.add_argument("--reference-protocol", required=True, type=Path)
    foundation_protocol.add_argument("--output-dir", required=True, type=Path)
    benchmark_audit = subparsers.add_parser(
        "benchmark-audit", help="independently audit a completed benchmark protocol"
    )
    benchmark_audit.add_argument("--protocol-dir", required=True, type=Path)
    benchmark_audit.add_argument("--reference-protocol", required=True, type=Path)
    ablation_audit = subparsers.add_parser(
        "ablation-audit", help="independently audit a retrained-ablation protocol"
    )
    ablation_audit.add_argument("--protocol-dir", required=True, type=Path)
    ablation_audit.add_argument("--reference-protocol", required=True, type=Path)
    comparison = subparsers.add_parser(
        "comparison", help="build seed-averaged origin-level final statistical report"
    )
    comparison.add_argument("--reference-protocol", required=True, type=Path)
    comparison.add_argument("--native-protocol", required=True, type=Path)
    comparison.add_argument("--timemixer-protocol", type=Path, default=None)
    comparison.add_argument("--foundation-protocol", type=Path, default=None)
    comparison.add_argument("--ablation-protocol", type=Path, default=None)
    comparison.add_argument("--output-dir", required=True, type=Path)
    comparison_figures = subparsers.add_parser(
        "comparison-figures", help="render the frozen final statistical comparison"
    )
    comparison_figures.add_argument("--comparison-dir", required=True, type=Path)
    comparison_figures.add_argument("--output-dir", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "run":
        run_path = run_experiment(load_config(args.config), args.run_dir)
        print(run_path.resolve())
    elif args.command == "figures":
        manifest = regenerate_figures(args.run_dir)
        print(f"generated {len(manifest)} figures")
    elif args.command == "audit":
        report = audit_completed_run(args.run_dir)
        print(f"run integrity passed: {report['passed']}")
    elif args.command == "protocol":
        output = run_protocol(args.config, args.output_dir)
        print(output.resolve())
    elif args.command == "squared-error-report":
        output = build_squared_error_report(args.protocol_dir, args.output)
        print(output.resolve())
    elif args.command == "benchmark":
        output = run_benchmark(args.config, args.reference_run, args.output_dir)
        print(output.resolve())
    elif args.command == "benchmark-protocol":
        output = run_benchmark_protocol(
            args.config, args.reference_protocol, args.output_dir
        )
        print(output.resolve())
    elif args.command == "ablation":
        output = run_retrained_ablation(
            args.reference_run, args.variant, args.output_dir
        )
        print(output.resolve())
    elif args.command == "ablation-protocol":
        output = run_ablation_protocol(args.reference_protocol, args.output_dir)
        print(output.resolve())
    elif args.command == "foundation":
        output = run_foundation_benchmark(
            args.config, args.reference_run, args.output_dir
        )
        print(output.resolve())
    elif args.command == "foundation-protocol":
        output = run_foundation_protocol(
            args.config, args.reference_protocol, args.output_dir
        )
        print(output.resolve())
    elif args.command == "benchmark-audit":
        report = audit_benchmark_protocol(
            args.protocol_dir, args.reference_protocol
        )
        print(f"benchmark protocol audit passed: {report['passed']}")
    elif args.command == "ablation-audit":
        report = audit_ablation_protocol(
            args.protocol_dir, args.reference_protocol
        )
        print(f"ablation protocol audit passed: {report['passed']}")
    elif args.command == "comparison":
        output = build_comparison_report(
            args.reference_protocol,
            args.native_protocol,
            args.output_dir,
            timemixer_protocol=args.timemixer_protocol,
            foundation_protocol=args.foundation_protocol,
            ablation_protocol=args.ablation_protocol,
        )
        print(output.resolve())
    elif args.command == "comparison-figures":
        output = generate_comparison_figures(
            args.comparison_dir, args.output_dir
        )
        print(output.resolve())
    else:
        raise RuntimeError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
