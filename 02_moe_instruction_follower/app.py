import argparse

from core import Config
from experiments import benchmark_crossover, demo, plot_surfaces, run_sweep
from train import train


def main():
    p = argparse.ArgumentParser(
        description="MoE instruction-follower: train, sweep, benchmark, analyze"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sweep", help="run the accuracy grid -> datasheet.jsonl")
    s.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    s.add_argument("--steps", type=int, default=8000)
    s.add_argument("--d-models", type=int, nargs="+", default=[16, 32, 64, 128])
    s.add_argument("--n-ops", type=int, nargs="+", default=[4, 8, 16])
    s.add_argument("--n-dig", type=int, nargs="+", default=[10, 20])

    b = sub.add_parser("bench", help="sparse vs naive MoE compute crossover")
    b.add_argument("--d-models", type=int, nargs="+", default=[64, 128, 256, 512, 1024])

    sub.add_parser("plot", help="render accuracy surfaces from datasheet.jsonl")
    sub.add_parser("demo", help="interactive: prompt models, compare outputs")

    t = sub.add_parser("train", help="train one config")
    t.add_argument("--d-model", type=int, default=64)
    t.add_argument("--n-layers", type=int, default=2)
    t.add_argument("--n-heads", type=int, default=1)
    t.add_argument("--experts", type=int, default=None)
    t.add_argument("--top-k", type=int, default=2)
    t.add_argument("--steps", type=int, default=8000)
    t.add_argument("--seed", type=int, default=0)

    args = p.parse_args()

    if args.cmd == "sweep":
        run_sweep(
            seeds=args.seeds,
            steps=args.steps,
            d_models=args.d_models,
            n_ops=args.n_ops,
            n_dig=args.n_dig,
        )
    elif args.cmd == "bench":
        benchmark_crossover(d_models=args.d_models)
    elif args.cmd == "plot":
        plot_surfaces()
    elif args.cmd == "demo":
        demo()
    elif args.cmd == "train":
        cfg = Config(
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_head=args.n_heads,
            n_experts=args.experts,
            top_k=args.top_k,
            steps=args.steps,
            seed=args.seed,
        )
        train(cfg)


if __name__ == "__main__":
    main()
