"""Black-Scholes pricing, Greeks, and a short-option delta-hedge overlay.

An options market-maker (Wolverine's world) does not need a crystal ball for
the underlying. They sell convexity, hedge the delta in the cash/futures
book, and live or die on whether realized vol comes in below the vol they
sold — after transaction costs.

The interesting question this module is built to answer: can you *earn* the
spread while hedging, instead of paying it?
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log, sqrt

from scipy.stats import norm


def _d1(spot: float, strike: float, tau: float, rate: float, vol: float) -> float:
    if tau <= 0 or vol <= 0 or spot <= 0 or strike <= 0:
        return 0.0
    return (log(spot / strike) + (rate + 0.5 * vol * vol) * tau) / (vol * sqrt(tau))


def _d2(spot: float, strike: float, tau: float, rate: float, vol: float) -> float:
    if tau <= 0 or vol <= 0:
        return 0.0
    return _d1(spot, strike, tau, rate, vol) - vol * sqrt(tau)


def bs_call(spot: float, strike: float, tau: float, rate: float, vol: float) -> float:
    if tau <= 0:
        return max(spot - strike, 0.0)
    d1 = _d1(spot, strike, tau, rate, vol)
    d2 = d1 - vol * sqrt(tau)
    return spot * norm.cdf(d1) - strike * exp(-rate * tau) * norm.cdf(d2)


def bs_put(spot: float, strike: float, tau: float, rate: float, vol: float) -> float:
    if tau <= 0:
        return max(strike - spot, 0.0)
    d1 = _d1(spot, strike, tau, rate, vol)
    d2 = d1 - vol * sqrt(tau)
    return strike * exp(-rate * tau) * norm.cdf(-d2) - spot * norm.cdf(-d1)


def bs_delta_call(spot: float, strike: float, tau: float, rate: float, vol: float) -> float:
    if tau <= 0:
        return 1.0 if spot > strike else 0.0
    return float(norm.cdf(_d1(spot, strike, tau, rate, vol)))


def bs_gamma(spot: float, strike: float, tau: float, rate: float, vol: float) -> float:
    if tau <= 0 or vol <= 0 or spot <= 0:
        return 0.0
    d1 = _d1(spot, strike, tau, rate, vol)
    return float(norm.pdf(d1) / (spot * vol * sqrt(tau)))


def bs_vega(spot: float, strike: float, tau: float, rate: float, vol: float) -> float:
    """Vega per 1.00 of vol (not per vol-point)."""
    if tau <= 0 or spot <= 0:
        return 0.0
    d1 = _d1(spot, strike, tau, rate, vol)
    return float(spot * norm.pdf(d1) * sqrt(tau))


def implied_vol_call(
    price: float,
    spot: float,
    strike: float,
    tau: float,
    rate: float,
    lo: float = 1e-4,
    hi: float = 5.0,
    tol: float = 1e-8,
    max_iter: int = 80,
) -> float:
    """Bisection implied vol. Robust; no vega-blow-up near expiry."""
    intrinsic = max(spot - strike * exp(-rate * tau), 0.0) if tau > 0 else max(spot - strike, 0.0)
    if price <= intrinsic + 1e-12:
        return lo
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        px = bs_call(spot, strike, tau, rate, mid)
        if px > price:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


@dataclass
class OptionContract:
    kind: str  # "call" | "put"
    strike: float
    tau0: float  # years at t=0
    rate: float
    implied_vol: float  # vol the MM sold
    multiplier: int = 1
    qty_short: int = 1  # positive = short

    def tau(self, t_years: float) -> float:
        return max(self.tau0 - t_years, 0.0)

    def value(self, spot: float, t_years: float, vol: float | None = None) -> float:
        vol = self.implied_vol if vol is None else vol
        tau = self.tau(t_years)
        if self.kind == "call":
            unit = bs_call(spot, self.strike, tau, self.rate, vol)
        else:
            unit = bs_put(spot, self.strike, tau, self.rate, vol)
        return unit * self.multiplier * self.qty_short

    def delta(self, spot: float, t_years: float, vol: float | None = None) -> float:
        vol = self.implied_vol if vol is None else vol
        tau = self.tau(t_years)
        call_d = bs_delta_call(spot, self.strike, tau, self.rate, vol)
        unit = call_d if self.kind == "call" else call_d - 1.0
        # Short option: position delta is negative of unit delta.
        return -unit * self.multiplier * self.qty_short

    def gamma(self, spot: float, t_years: float, vol: float | None = None) -> float:
        vol = self.implied_vol if vol is None else vol
        # Short gamma.
        return -bs_gamma(spot, self.strike, self.tau(t_years), self.rate, vol) * self.multiplier * self.qty_short

    def payoff(self, spot_T: float) -> float:
        if self.kind == "call":
            unit = max(spot_T - self.strike, 0.0)
        else:
            unit = max(self.strike - spot_T, 0.0)
        # Short: we pay the payoff.
        return -unit * self.multiplier * self.qty_short
