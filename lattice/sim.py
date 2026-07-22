"""Event-driven microstructure simulator.

The world has four actors and one clock:

- a latent fair value, either a Brownian mid or a two-state Glosten–Milgrom V
- uninformed (noise) market orders, Poisson in time and random in side
- informed market orders that *know* V and only trade when it is on their side
- a crowd of background limit orders that rest around V, giving the book depth
- the agent, who cancels and replaces on a timer, with latency

Nothing fills except through the matching engine. If the agent prints money,
it is because of an economic mechanism we can name, not because a backtest
leaked future prices into a signal.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from pydantic import BaseModel, Field

from lattice.analytics import Metrics, summarize
from lattice.book import OrderBook
from lattice.options import OptionContract
from lattice.strategies import (
    AvellanedaStoikov,
    GlostenMilgrom,
    InventorySkew,
    NaiveMM,
    Quote,
    Strategy,
)
from lattice.types import EventKind, Fill, Liquidity, Side, TickSize, TIF, Trade


NS = 1_000_000_000
MM_ID = "mm"
BG_ID = "bg"
NOISE_ID = "noise"
INFO_ID = "informed"


class SimConfig(BaseModel):
    seed: int = 7
    horizon_s: float = 60.0
    tick_size: float = 0.01
    s0: float = 100.0

    # Latent value. `sigma_ticks_per_sqrt_s` is *trading-time* vol: a 40s
    # session is a high-resolution slice of a day, not 40 calendar seconds
    # of 20% annualized equity vol (which would barely move the book).
    value_process: str = "bm"  # "bm" | "two_state"
    sigma_ticks_per_sqrt_s: float = 5.0
    sigma_annual: float = 0.20  # kept for the options overlay's calendar clock
    jump_prob: float = 0.0
    jump_ticks: int = 8
    v_low: float = 99.90
    v_high: float = 100.10
    p_high: float = 0.5

    # Flow
    noise_lambda: float = 8.0  # market orders / second
    informed_lambda: float = 2.5
    informed_edge_boost: float = 0.35  # extra intensity per tick of mispricing
    mo_qty_mean: float = 40.0
    mo_qty_min: int = 10
    mo_qty_max: int = 120

    # Background book
    bg_levels: int = 8
    bg_qty: int = 80
    bg_refresh_s: float = 0.35
    bg_cancel_lambda: float = 4.0

    # Agent
    strategy: str = "avellaneda_stoikov"
    requote_s: float = 0.15
    latency_ms: float = 8.0
    quote_qty: int = 50
    as_gamma: float = 0.05
    as_k: float = 1.5
    as_sigma_ticks: float = 8.0
    naive_half_ticks: int = 1
    gm_mu: float = 0.28
    inv_skew_per_100: float = 1.6
    max_inventory: int = 500

    # Fees (per share, dollars)
    taker_fee: float = 0.0002

    # Options overlay
    options_enabled: bool = False
    option_kind: str = "call"
    option_strike: float = 100.0
    option_tau_days: float = 5.0
    option_iv: float = 0.22
    option_rv: float = 0.18  # realized vol driving the BM, informational
    option_qty_short: int = 20
    option_rate: float = 0.0
    hedge_mode: str = "mm"  # "mm" | "taker" | "frictionless" | "none"
    hedge_every_s: float = 1.0

    # Recording
    snapshot_s: float = 0.12
    max_frames: int = 500
    book_depth: int = 8


@dataclass(order=True)
class _Event:
    ts_ns: int
    seq: int
    kind: EventKind = field(compare=False)
    payload: dict = field(default_factory=dict, compare=False)


@dataclass
class Frame:
    t: float
    mid: float
    fair: float
    bid: Optional[float]
    ask: Optional[float]
    mm_bid: Optional[float]
    mm_ask: Optional[float]
    inventory: int
    equity: float
    cash: float
    spread_ticks: Optional[int]
    p_high: Optional[float]
    target_inv: float
    option_delta: Optional[float]
    bids: list[dict]
    asks: list[dict]


@dataclass
class SimResult:
    config: dict
    metrics: Metrics
    frames: list[Frame]
    fills: list[dict]
    trades_n: int
    strategy: str
    notes: list[str] = field(default_factory=list)

    def to_jsonable(self) -> dict[str, Any]:
        m = self.metrics
        attr = m.attribution
        # Downsample paths for the wire.
        path_n = min(len(m.time_s), 400)
        idx = np.linspace(0, max(len(m.time_s) - 1, 0), path_n, dtype=int) if m.time_s else []
        return {
            "config": self.config,
            "strategy": self.strategy,
            "notes": self.notes,
            "trades_n": self.trades_n,
            "metrics": {
                "n_fills": m.n_fills,
                "buy_qty": m.buy_qty,
                "sell_qty": m.sell_qty,
                "maker_fills": m.maker_fills,
                "taker_fills": m.taker_fills,
                "informed_fills": m.informed_fills,
                "final_inventory": m.final_inventory,
                "final_cash": m.final_cash,
                "pnl": m.pnl,
                "sharpe": m.sharpe,
                "max_drawdown": m.max_drawdown,
                "fill_ratio": m.fill_ratio,
                "avg_queue_at_fill": m.avg_queue_at_fill,
                "markouts": [
                    {
                        "horizon_s": p.horizon_s,
                        "mean": p.mean,
                        "mean_informed": p.mean_informed,
                        "mean_uninformed": p.mean_uninformed,
                        "n": p.n,
                    }
                    for p in m.markouts
                ],
                "attribution": {
                    "spread_pnl": attr.spread_pnl,
                    "inventory_pnl": attr.inventory_pnl,
                    "markout_1s": attr.markout_1s,
                    "markout_5s": attr.markout_5s,
                    "fees": attr.fees,
                    "option_premium": attr.option_premium,
                    "option_payoff": attr.option_payoff,
                    "option_mtm": attr.option_mtm,
                    "trading_pnl": attr.trading_pnl,
                    "option_pnl": attr.option_pnl,
                    "total": attr.total,
                },
                "extra": m.extra,
            },
            "path": {
                "t": [m.time_s[i] for i in idx],
                "equity": [m.equity_path[i] for i in idx],
                "inventory": [m.inventory_path[i] for i in idx],
            },
            "frames": [
                {
                    "t": f.t,
                    "mid": f.mid,
                    "fair": f.fair,
                    "bid": f.bid,
                    "ask": f.ask,
                    "mm_bid": f.mm_bid,
                    "mm_ask": f.mm_ask,
                    "inventory": f.inventory,
                    "equity": f.equity,
                    "cash": f.cash,
                    "spread_ticks": f.spread_ticks,
                    "p_high": f.p_high,
                    "target_inv": f.target_inv,
                    "option_delta": f.option_delta,
                    "bids": f.bids,
                    "asks": f.asks,
                }
                for f in self.frames
            ],
            "fills": self.fills[:800],
        }


class Simulator:
    def __init__(self, cfg: SimConfig, strategy: Optional[Strategy] = None) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.tick = TickSize(cfg.tick_size)
        self.book = OrderBook(mm_trader_id=MM_ID)
        self.strategy = strategy or self._build_strategy()
        self.events: list[_Event] = []
        self.seq = 0
        self.ts = 0
        self.horizon_ns = int(cfg.horizon_s * NS)

        self.s0_ticks = self.tick.to_ticks(cfg.s0)
        if cfg.value_process == "two_state":
            high = bool(self.rng.random() < cfg.p_high)
            self.fair_ticks = float(self.tick.to_ticks(cfg.v_high if high else cfg.v_low))
            self.true_high = high
        else:
            self.fair_ticks = float(self.s0_ticks)
            self.true_high = None

        self.sigma_ticks = float(cfg.sigma_ticks_per_sqrt_s)

        self.inventory = 0
        self.cash = 0.0
        self.fills: list[Fill] = []
        self.all_trades: list[Trade] = []
        self.n_quotes = 0
        self.equity_path: list[float] = []
        self.inv_path: list[int] = []
        self.time_s: list[float] = []
        self.snap_ts: list[int] = []
        self.snap_mid: list[float] = []
        self.frames: list[Frame] = []
        self.last_mid_ticks = float(self.s0_ticks)
        self.pending_quote: Optional[Quote] = None
        self.notes: list[str] = []

        self.option: Optional[OptionContract] = None
        self.option_premium = 0.0
        if cfg.options_enabled:
            tau0 = cfg.option_tau_days / 365.0
            self.option = OptionContract(
                kind=cfg.option_kind,
                strike=cfg.option_strike,
                tau0=tau0,
                rate=cfg.option_rate,
                implied_vol=cfg.option_iv,
                multiplier=1,
                qty_short=cfg.option_qty_short,
            )
            # Sold at IV on the initial spot. Premium is cash in the door.
            self.option_premium = -self.option.value(cfg.s0, 0.0, cfg.option_iv)
            # value() for a short is already negative of the unit price * qty,
            # so -value = premium received. Wait: value() returns unit * mult * qty_short
            # for the *contract value as marked*, and qty_short is positive meaning short.
            # Looking at options.py: value = unit * multiplier * qty_short > 0
            # and payoff for short is -unit. That's inconsistent.
            # We'll treat option.value as the unmarked unit value of the long,
            # and premium received = qty_short * unit_price.
            unit = self.option.value(cfg.s0, 0.0, cfg.option_iv) / max(cfg.option_qty_short, 1)
            self.option_premium = unit * cfg.option_qty_short
            self.cash += self.option_premium
            self.notes.append(
                f"Sold {cfg.option_qty_short} {cfg.option_kind}s @ IV={cfg.option_iv:.0%} "
                f"for premium ${self.option_premium:.2f}. Hedge mode={cfg.hedge_mode}."
            )

        self.target_inv = 0.0

    def _build_strategy(self) -> Strategy:
        cfg = self.cfg
        name = cfg.strategy.lower()
        if name in ("naive", "naive_mm"):
            return NaiveMM(half_ticks=cfg.naive_half_ticks, qty=cfg.quote_qty)
        if name in ("as", "avellaneda", "avellaneda_stoikov"):
            return AvellanedaStoikov(
                gamma=cfg.as_gamma,
                sigma_ticks=max(cfg.as_sigma_ticks, 1.0),
                k=cfg.as_k,
                qty=cfg.quote_qty,
            )
        if name in ("gm", "glosten", "glosten_milgrom"):
            return GlostenMilgrom(
                v_low_ticks=self.tick.to_ticks(cfg.v_low),
                v_high_ticks=self.tick.to_ticks(cfg.v_high),
                mu=cfg.gm_mu,
                prior=cfg.p_high,
                qty=cfg.quote_qty,
                max_inv=cfg.max_inventory,
            )
        if name in ("skew", "inventory", "inventory_skew"):
            return InventorySkew(
                half_ticks=cfg.naive_half_ticks,
                skew_per_100=cfg.inv_skew_per_100,
                qty=cfg.quote_qty,
            )
        raise ValueError(f"unknown strategy {cfg.strategy}")

    # ----------------------------------------------------------------- events
    def _exp(self, rate: float) -> int:
        if rate <= 0:
            return self.horizon_ns * 2
        dt = float(self.rng.exponential(1.0 / rate))
        return max(int(dt * NS), 1)

    def _push(self, ts: int, kind: EventKind, payload: Optional[dict] = None) -> None:
        if ts > self.horizon_ns:
            return
        self.seq += 1
        heapq.heappush(self.events, _Event(ts, self.seq, kind, payload or {}))

    def _qty(self) -> int:
        q = int(self.rng.exponential(self.cfg.mo_qty_mean))
        return int(min(max(q, self.cfg.mo_qty_min), self.cfg.mo_qty_max))

    # ----------------------------------------------------------- value process
    def _advance_value(self, now: int) -> None:
        dt = (now - self.ts) / NS
        if dt <= 0:
            self.ts = now
            return
        cfg = self.cfg
        if cfg.value_process == "bm":
            d = self.sigma_ticks * np.sqrt(dt) * float(self.rng.normal())
            if cfg.jump_prob > 0 and self.rng.random() < cfg.jump_prob * dt:
                d += cfg.jump_ticks * (1.0 if self.rng.random() < 0.5 else -1.0)
            self.fair_ticks += d
        self.ts = now

    def _spot(self) -> float:
        return self.tick.to_price(self.fair_ticks)

    def _mid_or_fair(self) -> float:
        m = self.book.mid_ticks()
        return float(m) if m is not None else float(self.fair_ticks)

    def _t_frac(self) -> float:
        return max((self.horizon_ns - self.ts) / max(self.horizon_ns, 1), 1e-6)

    def _t_years(self) -> float:
        return (self.ts / NS) / (365.0 * 24 * 3600)

    def _update_target(self) -> None:
        if self.option is None or self.cfg.hedge_mode == "none":
            self.target_inv = 0.0
            return
        spot = self._spot()
        # Hedge target in shares = -position_delta of the option.
        self.target_inv = -self.option.delta(spot, self._t_years(), self.cfg.option_iv)

    def _effective_inventory(self) -> int:
        return int(round(self.inventory - self.target_inv))

    def _equity(self) -> float:
        mid = self._mid_or_fair()
        eq = self.cash + self.inventory * self.tick.to_price(mid)
        if self.option is not None:
            # MTM of short option: we received premium (in cash). Liability is
            # current unmarked value * qty_short.
            t = self._t_years()
            unit = self.option.value(self._spot(), t, self.cfg.option_iv) / max(
                self.cfg.option_qty_short, 1
            )
            eq -= unit * self.cfg.option_qty_short
        return eq

    # --------------------------------------------------------------- matching
    def _apply_trades(self, trades: list[Trade], taker_is_mm: bool = False) -> None:
        mid = self._mid_or_fair()
        for tr in trades:
            self.all_trades.append(tr)
            mm_involved = tr.is_mm_maker or tr.is_mm_taker
            if not mm_involved:
                continue
            if tr.is_mm_maker:
                # Maker: if aggressor was BID, they bought from our ASK.
                mm_side = Side.ASK if tr.aggressor_side is Side.BID else Side.BID
                liq = Liquidity.MAKER
            else:
                mm_side = tr.aggressor_side
                liq = Liquidity.TAKER
            px = self.tick.to_price(tr.price_ticks)
            if mm_side is Side.BID:
                self.inventory += tr.qty
                self.cash -= px * tr.qty
            else:
                self.inventory -= tr.qty
                self.cash += px * tr.qty
            if liq is Liquidity.TAKER:
                self.cash -= self.cfg.taker_fee * tr.qty
            else:
                self.cash += self.cfg.taker_fee * 0.4 * tr.qty
            fill = Fill(
                ts_ns=tr.ts_ns,
                side=mm_side,
                price_ticks=tr.price_ticks,
                qty=tr.qty,
                liquidity=liq,
                informed=tr.informed,
                queue_at_insert=tr.maker_queue_at_insert,
                queue_at_fill=tr.maker_queue_at_fill,
                mid_at_fill_ticks=mid,
                trade_id=tr.trade_id,
            )
            self.fills.append(fill)
            self.strategy.on_fill(mm_side, tr.qty, tr.informed, mid)

    # ----------------------------------------------------------------- actors
    def _seed_book(self) -> None:
        self._refresh_background()

    def _refresh_background(self) -> None:
        self.book.cancel_all(BG_ID)
        center = int(round(self.fair_ticks))
        for i in range(2, self.cfg.bg_levels + 2):
            decay = max(int(self.cfg.bg_qty * (0.75 ** (i - 1))), 15)
            bid_px = center - i
            ask_px = center + i
            jitter_b = int(self.rng.integers(0, 8))
            jitter_a = int(self.rng.integers(0, 8))
            self.book.submit(BG_ID, Side.BID, bid_px, decay + jitter_b, self.ts, TIF.GTC)
            self.book.submit(BG_ID, Side.ASK, ask_px, decay + jitter_a, self.ts, TIF.GTC)

    def _noise_mo(self) -> None:
        side = Side.BID if self.rng.random() < 0.5 else Side.ASK
        _, trades = self.book.market(NOISE_ID, side, self._qty(), self.ts, informed=False)
        self._apply_trades(trades)

    def _informed_mo(self) -> None:
        mid = self._mid_or_fair()
        edge = self.fair_ticks - mid
        if abs(edge) < 0.25:
            # No edge, informed sits out (they are not noise).
            return
        side = Side.BID if edge > 0 else Side.ASK
        _, trades = self.book.market(INFO_ID, side, self._qty(), self.ts, informed=True)
        self._apply_trades(trades)

    def _decide_quotes(self) -> None:
        self._update_target()
        mid = self._mid_or_fair()
        q = self.strategy.quote(mid, self._effective_inventory(), self._t_frac())
        delay = int(self.cfg.latency_ms * 1e6)
        self._push(self.ts + max(delay, 1), EventKind.MM_REQUOTE, {"quote": q, "apply": True})

    def _apply_quotes(self, q: Quote) -> None:
        self.book.cancel_all(MM_ID)
        q = self._clamp_maker(q)
        # Hard inventory cap: stop quoting the dangerous side.
        if self.inventory >= self.cfg.max_inventory:
            q.bid_qty = 0
        if self.inventory <= -self.cfg.max_inventory:
            q.ask_qty = 0
        if q.bid_qty > 0:
            _, tr_b = self.book.submit(MM_ID, Side.BID, q.bid_ticks, q.bid_qty, self.ts, TIF.GTC)
            self._apply_trades(tr_b, taker_is_mm=True)
            self.n_quotes += 1
        if q.ask_qty > 0:
            _, tr_a = self.book.submit(MM_ID, Side.ASK, q.ask_ticks, q.ask_qty, self.ts, TIF.GTC)
            self._apply_trades(tr_a, taker_is_mm=True)
            self.n_quotes += 1
        self.pending_quote = q
        nxt = self.ts + int(self.cfg.requote_s * NS)
        self._push(nxt, EventKind.MM_REQUOTE, {"apply": False})

    def _clamp_maker(self, q: Quote) -> Quote:
        """Force quotes to rest. Crossing is a take, not a make."""
        bb, ba = self.book.best_bid(), self.book.best_ask()
        bid, ask = q.bid_ticks, q.ask_ticks
        if ba is not None:
            bid = min(bid, ba - 1)
        if bb is not None:
            ask = max(ask, bb + 1)
        if ask <= bid:
            if bb is not None and ba is not None and ba - bb >= 2:
                bid = (bb + ba) // 2
                ask = bid + 1
                bid = min(bid, ba - 1)
                ask = max(ask, bb + 1)
            else:
                ask = bid + 1
        q.bid_ticks = bid
        q.ask_ticks = ask
        return q

    def _taker_hedge(self) -> None:
        self._update_target()
        residual = int(round(self.target_inv - self.inventory))
        if residual == 0:
            return
        side = Side.BID if residual > 0 else Side.ASK
        _, trades = self.book.market(MM_ID, side, abs(residual), self.ts, informed=False)
        self._apply_trades(trades, taker_is_mm=True)

    def _frictionless_hedge(self) -> None:
        self._update_target()
        residual = self.target_inv - self.inventory
        if abs(residual) < 1e-9:
            return
        mid = self.tick.to_price(self._mid_or_fair())
        # Cross the spread: pay half-spread as a cost.
        half = 0.5 * self.cfg.tick_size * (self.book.spread_ticks() or 2)
        if residual > 0:
            self.cash -= (mid + half) * residual
            self.inventory += residual
        else:
            self.cash += (mid - half) * (-residual)
            self.inventory += residual  # residual negative
        self.inventory = int(round(self.inventory))

    def _bg_cancel_one(self) -> None:
        resting = self.book.resting_for(BG_ID)
        if not resting:
            return
        victim = resting[int(self.rng.integers(0, len(resting)))]
        self.book.cancel(victim.order_id)

    # -------------------------------------------------------------- recording
    def _record(self) -> None:
        snap = self.book.snapshot(self.ts, depth=self.cfg.book_depth)
        mid = snap.mid_ticks if snap.mid_ticks is not None else self.fair_ticks
        self.last_mid_ticks = float(mid)
        t = self.ts / NS
        eq = self._equity()
        self.time_s.append(t)
        self.equity_path.append(eq)
        self.inv_path.append(int(round(self.inventory)))
        self.snap_ts.append(self.ts)
        self.snap_mid.append(float(mid))

        mm_bid = mm_ask = None
        for o in self.book.resting_for(MM_ID):
            px = self.tick.to_price(o.price_ticks)
            if o.side is Side.BID:
                mm_bid = px if mm_bid is None else max(mm_bid, px)
            else:
                mm_ask = px if mm_ask is None else min(mm_ask, px)

        p_high = None
        if isinstance(self.strategy, GlostenMilgrom):
            p_high = self.strategy.p

        opt_d = None
        if self.option is not None:
            opt_d = self.option.delta(self._spot(), self._t_years(), self.cfg.option_iv)

        def pack(levels: list) -> list[dict]:
            return [
                {
                    "px": self.tick.to_price(lv.price_ticks),
                    "qty": lv.qty,
                    "mm": lv.mm_qty,
                }
                for lv in levels
            ]

        if len(self.frames) < self.cfg.max_frames:
            self.frames.append(
                Frame(
                    t=t,
                    mid=self.tick.to_price(mid),
                    fair=self.tick.to_price(self.fair_ticks),
                    bid=self.tick.to_price(snap.best_bid) if snap.best_bid is not None else None,
                    ask=self.tick.to_price(snap.best_ask) if snap.best_ask is not None else None,
                    mm_bid=mm_bid,
                    mm_ask=mm_ask,
                    inventory=int(round(self.inventory)),
                    equity=eq,
                    cash=self.cash,
                    spread_ticks=snap.spread_ticks,
                    p_high=p_high,
                    target_inv=self.target_inv,
                    option_delta=opt_d,
                    bids=pack(snap.bids),
                    asks=pack(snap.asks),
                )
            )

    # ------------------------------------------------------------------- run
    def run(self) -> SimResult:
        cfg = self.cfg
        self._seed_book()
        now = 0
        self._push(0, EventKind.SNAPSHOT)
        self._push(self._exp(cfg.noise_lambda), EventKind.NOISE_MO)
        self._push(self._exp(cfg.informed_lambda), EventKind.INFORMED_MO)
        self._push(int(cfg.bg_refresh_s * NS), EventKind.BG_LIMIT)
        self._push(self._exp(cfg.bg_cancel_lambda), EventKind.BG_CANCEL)
        self._push(0, EventKind.MM_REQUOTE, {"apply": False})
        if cfg.options_enabled and cfg.hedge_mode in ("taker", "frictionless"):
            self._push(int(cfg.hedge_every_s * NS), EventKind.OPTION_HEDGE)

        while self.events:
            ev = heapq.heappop(self.events)
            self._advance_value(ev.ts_ns)
            kind = ev.kind

            if kind is EventKind.NOISE_MO:
                self._noise_mo()
                self._push(self.ts + self._exp(cfg.noise_lambda), EventKind.NOISE_MO)
            elif kind is EventKind.INFORMED_MO:
                # Intensity boost when the book is stale vs fair value.
                mid = self._mid_or_fair()
                boost = 1.0 + cfg.informed_edge_boost * abs(self.fair_ticks - mid)
                # Thinning: sometimes skip when boost was baked into the wait.
                if self.rng.random() < min(boost / (1.0 + cfg.informed_edge_boost * 4), 1.0):
                    self._informed_mo()
                self._push(self.ts + self._exp(cfg.informed_lambda), EventKind.INFORMED_MO)
            elif kind is EventKind.BG_LIMIT:
                self._refresh_background()
                self._push(self.ts + int(cfg.bg_refresh_s * NS), EventKind.BG_LIMIT)
            elif kind is EventKind.BG_CANCEL:
                self._bg_cancel_one()
                self._push(self.ts + self._exp(cfg.bg_cancel_lambda), EventKind.BG_CANCEL)
            elif kind is EventKind.MM_REQUOTE:
                if ev.payload.get("apply"):
                    q: Quote = ev.payload["quote"]
                    self._apply_quotes(q)
                else:
                    self._decide_quotes()
            elif kind is EventKind.OPTION_HEDGE:
                if cfg.hedge_mode == "taker":
                    self._taker_hedge()
                elif cfg.hedge_mode == "frictionless":
                    self._frictionless_hedge()
                self._push(self.ts + int(cfg.hedge_every_s * NS), EventKind.OPTION_HEDGE)
            elif kind is EventKind.SNAPSHOT:
                self._record()
                self._push(self.ts + int(cfg.snapshot_s * NS), EventKind.SNAPSHOT)

        # Terminal mark.
        self._record()
        option_payoff = 0.0
        option_mtm = 0.0
        if self.option is not None:
            spot_T = self._spot()
            unit_payoff = (
                max(spot_T - self.option.strike, 0.0)
                if self.option.kind == "call"
                else max(self.option.strike - spot_T, 0.0)
            )
            option_payoff = -unit_payoff * self.cfg.option_qty_short
            # MTM already in equity via unmarked value; for attribution we
            # report premium, terminal payoff, and leftover time-value separately.
            t = self._t_years()
            unit_now = self.option.value(spot_T, t, self.cfg.option_iv) / max(self.cfg.option_qty_short, 1)
            # Remaining unmarked vs intrinsic: we flatten MTM into payoff at horizon
            # if tau still > 0, keep mtm as -(unit) + intrinsic already in payoff? 
            # Cleaner: option_pnl = premium + (-current_mark)
            option_mtm = -unit_now * self.cfg.option_qty_short
            # Don't also add payoff if we MTM to the live mark (sim ends before expiry
            # unless tau_days is tiny). Prefer live mark.
            option_payoff = 0.0

        extra = {
            "true_high": self.true_high,
            "final_fair": self.tick.to_price(self.fair_ticks),
            "n_quotes": self.n_quotes,
            "target_inv": self.target_inv,
        }
        if isinstance(self.strategy, GlostenMilgrom):
            extra["p_high"] = self.strategy.p
            extra["belief_error"] = abs(
                self.strategy.p - (1.0 if self.true_high else 0.0)
            ) if self.true_high is not None else None

        metrics = summarize(
            fills=self.fills,
            equity=self.equity_path,
            inventory=self.inv_path,
            time_s=self.time_s,
            snap_ts=self.snap_ts,
            snap_mid=self.snap_mid,
            tick_size=cfg.tick_size,
            fee_per_share=cfg.taker_fee,
            n_quotes=max(self.n_quotes, 1),
            horizon_s=cfg.horizon_s,
            option_premium=self.option_premium,
            option_payoff=option_payoff,
            option_mtm=option_mtm,
            extra=extra,
        )
        fill_dicts = [
            {
                "t": f.ts_ns / NS,
                "side": "bid" if f.side is Side.BID else "ask",
                "px": self.tick.to_price(f.price_ticks),
                "qty": f.qty,
                "liq": f.liquidity.value,
                "informed": f.informed,
                "queue": f.queue_at_fill,
                "mid": self.tick.to_price(f.mid_at_fill_ticks),
            }
            for f in self.fills
        ]
        return SimResult(
            config=cfg.model_dump(),
            metrics=metrics,
            frames=self.frames,
            fills=fill_dicts,
            trades_n=len(self.all_trades),
            strategy=self.strategy.name,
            notes=self.notes,
        )


def run_sim(cfg: Optional[SimConfig] = None, **kwargs) -> SimResult:
    if cfg is None:
        cfg = SimConfig(**kwargs)
    elif kwargs:
        cfg = cfg.model_copy(update=kwargs)
    return Simulator(cfg).run()


STRATEGIES = ("naive", "inventory_skew", "avellaneda_stoikov", "glosten_milgrom")


def compare_strategies(
    base: Optional[SimConfig] = None,
    strategies: tuple[str, ...] = STRATEGIES,
    n_seeds: int = 5,
    seed0: int = 11,
) -> list[dict]:
    """Monte Carlo comparison. Same market, different quoting policies."""
    base = base or SimConfig()
    rows = []
    for name in strategies:
        pnls = []
        sharpes = []
        spreads = []
        markouts = []
        dds = []
        for i in range(n_seeds):
            cfg = base.model_copy(
                update={"strategy": name, "seed": seed0 + i * 17}
            )
            if name == "glosten_milgrom":
                cfg.value_process = "two_state"
            else:
                cfg.value_process = "bm"
            res = Simulator(cfg).run()
            pnls.append(res.metrics.pnl)
            sharpes.append(res.metrics.sharpe)
            spreads.append(res.metrics.attribution.spread_pnl)
            markouts.append(res.metrics.attribution.markout_1s)
            dds.append(res.metrics.max_drawdown)
        rows.append(
            {
                "strategy": name,
                "mean_pnl": float(np.mean(pnls)),
                "std_pnl": float(np.std(pnls)),
                "mean_sharpe": float(np.mean(sharpes)),
                "mean_spread_pnl": float(np.mean(spreads)),
                "mean_markout_1s": float(np.mean(markouts)),
                "mean_max_dd": float(np.mean(dds)),
                "n": n_seeds,
            }
        )
    return rows
