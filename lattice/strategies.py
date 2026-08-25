"""Quoting policies.

Three policies, increasing in how much of the microstructure they 'see':

1. Naive — symmetric quotes around the touch / mid. A control.
2. Avellaneda-Stoikov (2008) — inventory-averse optimal MM on a Brownian mid.
3. Glosten-Milgrom Bayesian — quotes that are fair *conditional on being hit*,
   so the MM is compensated for adverse selection rather than pretending it
   does not exist.

A production desk is some blend of (2) and (3) plus a dozen overlays. The
point of implementing all three in the same book is that P&L differences are
then identified, not anecdotal.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from math import log

from lattice.types import Side


@dataclass
class Quote:
    bid_ticks: int
    ask_ticks: int
    bid_qty: int
    ask_qty: int
    reservation_ticks: float
    half_spread_ticks: float
    extra: dict = field(default_factory=dict)


class Strategy(ABC):
    name: str

    @abstractmethod
    def quote(
        self,
        mid_ticks: float,
        inventory: int,
        t_frac: float,
        tick: int = 1,
    ) -> Quote:
        """t_frac is time remaining in (0, 1], 1 at the open."""
        ...

    def on_fill(self, side: Side, qty: int, informed: bool, mid_ticks: float) -> None:
        """Optional belief update (used by the Bayesian MM)."""
        return None

    def on_mid(self, mid_ticks: float) -> None:
        return None


class NaiveMM(Strategy):
    """Always quote `half_ticks` away from mid, ignore inventory."""

    name = "naive"

    def __init__(self, half_ticks: int = 2, qty: int = 50) -> None:
        self.half_ticks = half_ticks
        self.qty = qty

    def quote(self, mid_ticks: float, inventory: int, t_frac: float, tick: int = 1) -> Quote:
        bid = int(mid_ticks) - self.half_ticks
        ask = int(mid_ticks) + (0 if mid_ticks == int(mid_ticks) else 1) + self.half_ticks
        if ask <= bid:
            ask = bid + 1
        return Quote(bid, ask, self.qty, self.qty, mid_ticks, float(self.half_ticks), {"inventory": inventory})


class AvellanedaStoikov(Strategy):
    """Closed-form AS quotes, discretized onto the tick grid.

    Reservation price:
        r = s - q * γ * σ² * (T-t)

    Optimal spread:
        δ^a + δ^b = γ σ² (T-t) + (2/γ) ln(1 + γ/k)

    Quotes sit symmetrically around r, not around s. Inventory is measured
    in lots of quote size so a realistic position moves the reservation by
    a few ticks, not a few dollars. sigma is session vol in ticks. k is the
    decay of fill intensity with distance (ticks^-1).

    """

    name = "avellaneda_stoikov"

    def __init__(
        self,
        gamma: float = 0.05,
        sigma_ticks: float = 8.0,
        k: float = 1.5,
        A: float = 1.0,
        qty: int = 50,
        min_half: int = 1,
        max_half: int = 6,
    ) -> None:
        if gamma <= 0:
            raise ValueError("gamma must be positive")
        if k <= 0:
            raise ValueError("k must be positive")
        self.gamma = gamma
        self.sigma_ticks = sigma_ticks
        self.k = k
        self.A = A
        self.qty = qty
        self.min_half = min_half
        self.max_half = max_half

    def quote(self, mid_ticks: float, inventory: int, t_frac: float, tick: int = 1) -> Quote:
        t_frac = max(t_frac, 1e-6)
        sigma2 = self.sigma_ticks * self.sigma_ticks
        # Inventory in lots of quote size, otherwise one position unit
        # shifts the reservation by tens of ticks and the agent starts taking.
        q_lots = inventory / max(self.qty, 1)
        reservation = mid_ticks - q_lots * self.gamma * sigma2 * t_frac
        max_skew = 5.0
        reservation = min(max(reservation, mid_ticks - max_skew), mid_ticks + max_skew)
        spread = self.gamma * sigma2 * t_frac + (2.0 / self.gamma) * log(1.0 + self.gamma / self.k)
        half = 0.5 * spread
        half = min(max(half, float(self.min_half)), float(self.max_half))

        bid = int(round(reservation - half))
        ask = int(round(reservation + half))
        if ask <= bid:
            ask = bid + 1

        # Inventory-skewed size: offer more on the side that reduces risk.
        bid_qty = self.qty
        ask_qty = self.qty
        if inventory > 0:
            ask_qty = int(self.qty * 1.4)
            bid_qty = int(self.qty * 0.6)
        elif inventory < 0:
            bid_qty = int(self.qty * 1.4)
            ask_qty = int(self.qty * 0.6)
        bid_qty = max(bid_qty, 10)
        ask_qty = max(ask_qty, 10)

        return Quote(
            bid_ticks=bid,
            ask_ticks=ask,
            bid_qty=bid_qty,
            ask_qty=ask_qty,
            reservation_ticks=reservation,
            half_spread_ticks=half,
            extra={"spread": spread, "t_frac": t_frac, "inventory": inventory},
        )


class GlostenMilgrom(Strategy):
    """Two-state Glosten-Milgrom market maker.

    Latent value V ∈ {V_low, V_high}. After each trade the MM updates
    p = P(V = V_high | history). Quotes are the *conditional* expectations
    given that the next order is a buy or a sell, which is exactly the
    condition under which the quote would be filled. That is the adverse-
    selection adjustment naive quoting is missing.

    We still discretize onto ticks and optionally overlay a small inventory
    skew so the strategy does not blow up position while it learns.
    """

    name = "glosten_milgrom"

    def __init__(
        self,
        v_low_ticks: float,
        v_high_ticks: float,
        mu: float = 0.25,
        prior: float = 0.5,
        qty: int = 50,
        inv_skew_ticks: float = 0.15,
        max_inv: int = 400,
    ) -> None:
        if not 0.0 < mu < 1.0:
            raise ValueError("mu (informed fraction) must be in (0,1)")
        self.v_low = v_low_ticks
        self.v_high = v_high_ticks
        self.mu = mu
        self.p = prior  # P(high)
        self.qty = qty
        self.inv_skew_ticks = inv_skew_ticks
        self.max_inv = max_inv

    def expected_v(self) -> float:
        return self.p * self.v_high + (1.0 - self.p) * self.v_low

    def _ask(self) -> float:
        """E[V | buy]. Informed buys only if V is high."""
        mu, p = self.mu, self.p
        # P(buy | high) = mu * 1 + (1-mu) * 1/2
        # P(buy | low)  = mu * 0 + (1-mu) * 1/2
        p_buy_high = mu + (1.0 - mu) * 0.5
        p_buy_low = (1.0 - mu) * 0.5
        p_buy = p_buy_high * p + p_buy_low * (1.0 - p)
        p_high_buy = p_buy_high * p / max(p_buy, 1e-12)
        return p_high_buy * self.v_high + (1.0 - p_high_buy) * self.v_low

    def _bid(self) -> float:
        """E[V | sell]."""
        mu, p = self.mu, self.p
        p_sell_high = (1.0 - mu) * 0.5
        p_sell_low = mu + (1.0 - mu) * 0.5
        p_sell = p_sell_high * p + p_sell_low * (1.0 - p)
        p_high_sell = p_sell_high * p / max(p_sell, 1e-12)
        return p_high_sell * self.v_high + (1.0 - p_high_sell) * self.v_low

    def on_fill(self, side: Side, qty: int, informed: bool, mid_ticks: float) -> None:
        # Bayesian update as if we observed a buy (hit our ask) or sell (hit our bid).
        mu, p = self.mu, self.p
        if side is Side.ASK:
            # We sold → someone bought.
            p_buy_high = mu + (1.0 - mu) * 0.5
            p_buy_low = (1.0 - mu) * 0.5
            lik_high, lik_low = p_buy_high, p_buy_low
        else:
            p_sell_high = (1.0 - mu) * 0.5
            p_sell_low = mu + (1.0 - mu) * 0.5
            lik_high, lik_low = p_sell_high, p_sell_low
        num = lik_high * p
        den = num + lik_low * (1.0 - p)
        self.p = min(max(num / max(den, 1e-12), 1e-4), 1.0 - 1e-4)

    def quote(self, mid_ticks: float, inventory: int, t_frac: float, tick: int = 1) -> Quote:
        bid_f = self._bid()
        ask_f = self._ask()
        # Inventory skew: lean against the position.
        skew = self.inv_skew_ticks * inventory
        bid_f -= skew
        ask_f -= skew

        bid = int(round(bid_f))
        ask = int(round(ask_f))
        if ask <= bid:
            ask = bid + 1

        # Flatten if we are near the inventory cap.
        bid_qty = self.qty
        ask_qty = self.qty
        if inventory >= self.max_inv * 0.7:
            bid_qty = 0
            ask_qty = int(self.qty * 1.5)
        elif inventory <= -self.max_inv * 0.7:
            ask_qty = 0
            bid_qty = int(self.qty * 1.5)

        return Quote(
            bid_ticks=bid,
            ask_ticks=ask,
            bid_qty=max(bid_qty, 0),
            ask_qty=max(ask_qty, 0),
            reservation_ticks=self.expected_v(),
            half_spread_ticks=0.5 * (ask - bid),
            extra={"p_high": self.p, "e_v": self.expected_v(), "inventory": inventory},
        )


class InventorySkew(Strategy):
    """AS-style reservation without the intensity term. Useful baseline."""

    name = "inventory_skew"

    def __init__(self, half_ticks: int = 2, skew_per_100: float = 1.5, qty: int = 50) -> None:
        self.half_ticks = half_ticks
        self.skew_per_100 = skew_per_100
        self.qty = qty

    def quote(self, mid_ticks: float, inventory: int, t_frac: float, tick: int = 1) -> Quote:
        skew = (inventory / 100.0) * self.skew_per_100
        reservation = mid_ticks - skew
        bid = int(round(reservation - self.half_ticks))
        ask = int(round(reservation + self.half_ticks))
        if ask <= bid:
            ask = bid + 1
        return Quote(bid, ask, self.qty, self.qty, reservation, float(self.half_ticks), {"skew": skew})


def make_strategy(name: str, **kwargs) -> Strategy:
    name = name.lower().replace("-", "_")
    if name in ("naive", "naive_mm"):
        return NaiveMM(**kwargs)
    if name in ("as", "avellaneda", "avellaneda_stoikov"):
        return AvellanedaStoikov(**kwargs)
    if name in ("gm", "glosten", "glosten_milgrom"):
        return GlostenMilgrom(**kwargs)
    if name in ("skew", "inventory", "inventory_skew"):
        return InventorySkew(**kwargs)
    raise ValueError(f"unknown strategy: {name}")
