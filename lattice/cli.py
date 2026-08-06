"""CLI: run, compare, serve."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from lattice.sim import SimConfig, compare_strategies, run_sim


def _print_result(res) -> None:
    m = res.metrics
    a = m.attribution
    print()
    print(f"  strategy     {res.strategy}")
    print(f"  trades       {res.trades_n}   mm fills {m.n_fills}   informed fills {m.informed_fills}")
    print(f"  inventory    {m.final_inventory:+d}   cash {m.final_cash:,.2f}")
    print(f"  P&L          {m.pnl:,.2f}   Sharpe {m.sharpe:+.2f}   max DD {m.max_drawdown:,.2f}")
    print(f"  spread P&L   {a.spread_pnl:+,.2f}")
    print(f"  inventory    {a.inventory_pnl:+,.2f}")
    print(f"  markout 1s   {a.markout_1s:+,.2f}   5s {a.markout_5s:+,.2f}")
    print(f"  fees         {a.fees:+,.2f}")
    if a.option_premium:
        print(f"  opt premium  {a.option_premium:+,.2f}   mtm {a.option_mtm:+,.2f}")
        print(f"  total        {a.total:+,.2f}")
    print(f"  avg queue    {m.avg_queue_at_fill:.1f} shares ahead at fill")
    for n in res.notes:
        print(f"  note         {n}")
    print()


def cmd_run(args: argparse.Namespace) -> int:
    cfg = SimConfig(
        seed=args.seed,
        horizon_s=args.horizon,
        strategy=args.strategy,
        latency_ms=args.latency,
        options_enabled=args.options,
        hedge_mode=args.hedge,
        value_process="two_state" if args.strategy in ("glosten_milgrom", "gm", "glosten") else "bm",
    )
    res = run_sim(cfg)
    _print_result(res)
    if args.out:
        Path(args.out).write_text(json.dumps(res.to_jsonable(), indent=2))
        print(f"  wrote {args.out}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    print(f"\n  Monte Carlo  n_seeds={args.seeds}  horizon={args.horizon}s\n")
    rows = compare_strategies(
        SimConfig(horizon_s=args.horizon, options_enabled=False),
        n_seeds=args.seeds,
        seed0=args.seed,
    )
    hdr = f"{'strategy':22} {'mean P&L':>10} {'std':>8} {'Sharpe':>8} {'spread':>10} {'markout1s':>10} {'max DD':>10}"
    print("  " + hdr)
    print("  " + "-" * len(hdr))
    for r in rows:
        print(
            f"  {r['strategy']:22} {r['mean_pnl']:10.2f} {r['std_pnl']:8.2f} "
            f"{r['mean_sharpe']:8.2f} {r['mean_spread_pnl']:10.2f} "
            f"{r['mean_markout_1s']:10.2f} {r['mean_max_dd']:10.2f}"
        )
    print()
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2))
    return 0


def cmd_options(args: argparse.Namespace) -> int:
    """Same option, three hedging styles — the Wolverine question."""
    print("\n  Short-option hedge cost in a live book\n")
    modes = ("none", "taker", "mm")
    rows = []
    for mode in modes:
        pnls = []
        for i in range(args.seeds):
            cfg = SimConfig(
                seed=args.seed + i * 13,
                horizon_s=args.horizon,
                strategy="avellaneda_stoikov",
                options_enabled=True,
                hedge_mode=mode if mode != "none" else "mm",
                quote_qty=40,
            )
            if mode == "none":
                cfg.options_enabled = True
                cfg.hedge_mode = "none"
                cfg.strategy = "naive"
            res = run_sim(cfg)
            pnls.append(res.metrics.pnl)
        import numpy as np

        rows.append((mode, float(np.mean(pnls)), float(np.std(pnls))))
        print(f"  hedge={mode:12}  mean P&L {np.mean(pnls):+10.2f}   std {np.std(pnls):8.2f}")
    print()
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from lattice.server import app

    print(f"\n  Lattice  →  http://{args.host}:{args.port}\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="lattice",
        description="Lattice — market microstructure laboratory",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run a single simulation")
    r.add_argument("--strategy", default="avellaneda_stoikov")
    r.add_argument("--seed", type=int, default=7)
    r.add_argument("--horizon", type=float, default=45.0)
    r.add_argument("--latency", type=float, default=8.0)
    r.add_argument("--options", action="store_true")
    r.add_argument("--hedge", default="mm", choices=("mm", "taker", "frictionless", "none"))
    r.add_argument("--out", default="")
    r.set_defaults(func=cmd_run)

    c = sub.add_parser("compare", help="Monte Carlo strategy comparison")
    c.add_argument("--seeds", type=int, default=7)
    c.add_argument("--seed", type=int, default=11)
    c.add_argument("--horizon", type=float, default=30.0)
    c.add_argument("--out", default="")
    c.set_defaults(func=cmd_compare)

    o = sub.add_parser("options", help="options hedge-mode comparison")
    o.add_argument("--seeds", type=int, default=5)
    o.add_argument("--seed", type=int, default=3)
    o.add_argument("--horizon", type=float, default=30.0)
    o.set_defaults(func=cmd_options)

    s = sub.add_parser("serve", help="open the research terminal")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(func=cmd_serve)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
