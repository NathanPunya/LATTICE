"""P&L attribution, markouts, and risk statistics.

A fill is not a profit. The only honest way to read a market-maker's day is
to split P&L into pieces that correspond to economically distinct bets:

- spread: the half-spread you captured at the moment of the fill
- inventory / position: mark-to-market of the residual position
- adverse selection / markout: how the mid moved *after* you were filled
- fees
- option premium vs. payoff vs. hedge (when the overlay is on)

If spread is green and markout is red, you were the dinner, not the diner.
That is the sentence interviewers want to hear you say about your own book.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Optional

import numpy as np

from lattice.types import Fill, Liquidity, Side


NS = 1_000_000_000


@dataclass
class MarkoutPoint:
    horizon_s: float
    mean: float
    mean_informed: float
    mean_uninformed: float
    n: int


@dataclass
class Attribution:
    spread_pnl: float
    inventory_pnl: float
    markout_1s: float
    markout_5s: float
    fees: float
    option_premium: float
    option_payoff: float
    option_mtm: float

    @property
    def trading_pnl(self) -> float:
        return self.spread_pnl + self.inventory_pnl - self.fees

    @property
    def option_pnl(self) -> float:
        return self.option_premium + self.option_payoff + self.option_mtm

    @property
    def total(self) -> float:
        return self.trading_pnl + self.option_pnl


@dataclass
class Metrics:
    n_fills: int
    buy_qty: int
    sell_qty: int
    maker_fills: int
    taker_fills: int
    informed_fills: int
    final_inventory: int
    final_cash: float
    equity_path: list[float]
    inventory_path: list[int]
    time_s: list[float]
    pnl: float
    mean_pnl: float
    std_pnl: float
    sharpe: float
    max_drawdown: float
    fill_ratio: float
    avg_queue_at_fill: float
    markouts: list[MarkoutPoint]
    attribution: Attribution
    extra: dict = field(default_factory=dict)


def _mid_after(times: np.ndarray, mids: np.ndarray, t0: int, horizon_s: float) -> Optional[float]:
    target = t0 + int(horizon_s * NS)
    idx = int(np.searchsorted(times, target, side="left"))
    if idx >= len(mids):
        return None
    return float(mids[idx])


def compute_markouts(
    fills: list[Fill],
    snap_ts: list[int],
    snap_mid: list[float],
    tick_size: float,
    horizons: tuple[float, ...] = (0.5, 1.0, 2.0, 5.0, 10.0),
) -> list[MarkoutPoint]:
    if not fills or not snap_ts:
        return [MarkoutPoint(h, 0.0, 0.0, 0.0, 0) for h in horizons]
    times = np.asarray(snap_ts, dtype=np.int64)
    mids = np.asarray(snap_mid, dtype=np.float64)
    out: list[MarkoutPoint] = []
    for h in horizons:
        vals: list[float] = []
        inf: list[float] = []
        uninf: list[float] = []
        for f in fills:
            m1 = _mid_after(times, mids, f.ts_ns, h)
            if m1 is None:
                continue
            # Signed markout in dollars, from the MM's perspective.
            # If we bought (BID fill), mid going down is bad.
            sign = 1.0 if f.side is Side.BID else -1.0
            mk = sign * (m1 - f.mid_at_fill_ticks) * tick_size * f.qty
            vals.append(mk)
            (inf if f.informed else uninf).append(mk)
        out.append(
            MarkoutPoint(
                horizon_s=h,
                mean=float(np.mean(vals)) if vals else 0.0,
                mean_informed=float(np.mean(inf)) if inf else 0.0,
                mean_uninformed=float(np.mean(uninf)) if uninf else 0.0,
                n=len(vals),
            )
        )
    return out


def max_drawdown(equity: list[float]) -> float:
    if not equity:
        return 0.0
    peak = equity[0]
    dd = 0.0
    for x in equity:
        peak = max(peak, x)
        dd = min(dd, x - peak)
    return dd


def sharpe_from_path(equity: list[float], dt_s: float, horizon_s: float) -> tuple[float, float, float]:
    """Annualize from step-to-step equity diffs. Returns (mean, std, sharpe)."""
    if len(equity) < 3:
        return 0.0, 0.0, 0.0
    diffs = np.diff(np.asarray(equity, dtype=np.float64))
    mu = float(np.mean(diffs))
    sd = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
    steps_per_year = (365.0 * 24 * 3600) / max(dt_s, 1e-9)
    # For a session of length horizon_s we report *session* Sharpe as well
    # as a toy annualization. Session Sharpe is the honest number.
    session_steps = max(horizon_s / max(dt_s, 1e-9), 1.0)
    sharpe = (mu / sd) * sqrt(session_steps) if sd > 1e-12 else 0.0
    return mu, sd, sharpe


def attribute(
    fills: list[Fill],
    tick_size: float,
    fee_per_share: float,
    final_inventory: int,
    final_mid_ticks: float,
    markouts: list[MarkoutPoint],
    option_premium: float = 0.0,
    option_payoff: float = 0.0,
    option_mtm: float = 0.0,
) -> Attribution:
    spread = 0.0
    fees = 0.0
    # Inventory P&L: every fill marked to the *terminal* mid, which is the
    # unique decomposition cash + inv * S_T. Spread is the half-spread at
    # fill vs the contemporaneous mid; inventory is the residual move.
    cash = 0.0
    inv = 0
    for f in fills:
        px = f.price_ticks * tick_size
        mid = f.mid_at_fill_ticks * tick_size
        if f.side is Side.BID:
            cash -= px * f.qty
            inv += f.qty
            spread += (mid - px) * f.qty
        else:
            cash += px * f.qty
            inv -= f.qty
            spread += (px - mid) * f.qty
        if f.liquidity is Liquidity.TAKER:
            fees += fee_per_share * f.qty
        else:
            fees -= fee_per_share * 0.4 * f.qty  # maker rebate, 40% of taker fee
    terminal = final_mid_ticks * tick_size
    # cash + inv * S_T  is total trading P&L. Spread was the instantaneous
    # edge; inventory P&L is the rest (including adverse selection).
    trading = cash + inv * terminal
    inventory_pnl = trading - spread + fees  # fees taken out of trading already? 
    # Let's be explicit:
    # trading_pnl_gross = cash + inv * S_T
    # trading_pnl_gross = spread + (cash + inv*S_T - spread)
    # inventory_pnl := trading_pnl_gross - spread
    inventory_pnl = trading - spread
    mk1 = next((m.mean * max(m.n, 1) for m in markouts if abs(m.horizon_s - 1.0) < 1e-9), 0.0)
    mk5 = next((m.mean * max(m.n, 1) for m in markouts if abs(m.horizon_s - 5.0) < 1e-9), 0.0)
    return Attribution(
        spread_pnl=spread,
        inventory_pnl=inventory_pnl,
        markout_1s=mk1,
        markout_5s=mk5,
        fees=fees,
        option_premium=option_premium,
        option_payoff=option_payoff,
        option_mtm=option_mtm,
    )


def summarize(
    fills: list[Fill],
    equity: list[float],
    inventory: list[int],
    time_s: list[float],
    snap_ts: list[int],
    snap_mid: list[float],
    tick_size: float,
    fee_per_share: float,
    n_quotes: int,
    horizon_s: float,
    option_premium: float = 0.0,
    option_payoff: float = 0.0,
    option_mtm: float = 0.0,
    extra: Optional[dict] = None,
) -> Metrics:
    buy_qty = sum(f.qty for f in fills if f.side is Side.BID)
    sell_qty = sum(f.qty for f in fills if f.side is Side.ASK)
    maker = sum(1 for f in fills if f.liquidity is Liquidity.MAKER)
    taker = sum(1 for f in fills if f.liquidity is Liquidity.TAKER)
    informed = sum(1 for f in fills if f.informed)
    final_inv = inventory[-1] if inventory else 0
    final_mid = snap_mid[-1] if snap_mid else 0.0
    markouts = compute_markouts(fills, snap_ts, snap_mid, tick_size)
    attr = attribute(
        fills,
        tick_size,
        fee_per_share,
        final_inv,
        final_mid,
        markouts,
        option_premium,
        option_payoff,
        option_mtm,
    )
    dt = (time_s[-1] - time_s[0]) / max(len(time_s) - 1, 1) if len(time_s) > 1 else 1.0
    mu, sd, sh = sharpe_from_path(equity, dt, horizon_s)
    avg_q = float(np.mean([f.queue_at_fill for f in fills if f.liquidity is Liquidity.MAKER])) if any(
        f.liquidity is Liquidity.MAKER for f in fills
    ) else 0.0
    cash = 0.0
    # reconstruct cash for reporting
    inv_run = 0
    for f in fills:
        px = f.price_ticks * tick_size
        if f.side is Side.BID:
            cash -= px * f.qty
            inv_run += f.qty
        else:
            cash += px * f.qty
            inv_run -= f.qty
    pnl = equity[-1] if equity else 0.0
    return Metrics(
        n_fills=len(fills),
        buy_qty=buy_qty,
        sell_qty=sell_qty,
        maker_fills=maker,
        taker_fills=taker,
        informed_fills=informed,
        final_inventory=final_inv,
        final_cash=cash,
        equity_path=equity,
        inventory_path=inventory,
        time_s=time_s,
        pnl=pnl,
        mean_pnl=mu,
        std_pnl=sd,
        sharpe=sh,
        max_drawdown=max_drawdown(equity),
        fill_ratio=(len(fills) / n_quotes) if n_quotes else 0.0,
        avg_queue_at_fill=avg_q,
        markouts=markouts,
        attribution=attr,
        extra=extra or {},
    )
