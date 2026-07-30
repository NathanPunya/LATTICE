from math import exp

from lattice.options import (
    OptionContract,
    bs_call,
    bs_delta_call,
    bs_gamma,
    bs_put,
    implied_vol_call,
)
from lattice.strategies import AvellanedaStoikov, GlostenMilgrom, NaiveMM
from lattice.types import Side


def test_put_call_parity():
    s, k, t, r, v = 100.0, 100.0, 0.5, 0.01, 0.2
    call = bs_call(s, k, t, r, v)
    put = bs_put(s, k, t, r, v)
    assert abs(call - put - (s - k * exp(-r * t))) < 1e-8


def test_atm_delta_near_half():
    d = bs_delta_call(100, 100, 0.25, 0.0, 0.2)
    assert 0.48 < d < 0.56


def test_gamma_positive_and_peaks_near_atm():
    g_atm = bs_gamma(100, 100, 0.25, 0.0, 0.2)
    g_otm = bs_gamma(80, 100, 0.25, 0.0, 0.2)
    assert g_atm > g_otm > 0


def test_implied_vol_roundtrip():
    s, k, t, r, v = 100.0, 105.0, 0.4, 0.0, 0.33
    px = bs_call(s, k, t, r, v)
    iv = implied_vol_call(px, s, k, t, r)
    assert abs(iv - v) < 1e-4


def test_short_call_delta_is_negative():
    opt = OptionContract("call", 100.0, 5 / 365, 0.0, 0.2, qty_short=10)
    d = opt.delta(100.0, 0.0)
    assert d < 0
    # Hedge target is -delta ≈ +5 shares at ATM for 10 short calls.
    assert 3 < -d < 7


def test_as_long_inventory_skews_down():
    as_ = AvellanedaStoikov(gamma=0.05, sigma_ticks=8, k=1.5)
    q0 = as_.quote(10_000, 0, 1.0)
    q_long = as_.quote(10_000, 80, 1.0)
    assert q_long.reservation_ticks < q0.reservation_ticks
    assert q_long.ask_ticks <= q0.ask_ticks


def test_naive_ignores_inventory():
    n = NaiveMM(half_ticks=2)
    a = n.quote(10_000, 0, 1.0)
    b = n.quote(10_000, 200, 1.0)
    assert a.bid_ticks == b.bid_ticks
    assert a.ask_ticks == b.ask_ticks


def test_glosten_buy_raises_p_high():
    gm = GlostenMilgrom(v_low_ticks=9900, v_high_ticks=10100, mu=0.3, prior=0.5)
    p0 = gm.p
    gm.on_fill(Side.ASK, 10, informed=True, mid_ticks=10000)  # we sold, they bought
    assert gm.p > p0
    ask = gm._ask()
    bid = gm._bid()
    assert ask > bid
    # Adverse-selection widened quotes vs unconditional expectation.
    ev = gm.expected_v()
    assert ask >= ev
    assert bid <= ev
