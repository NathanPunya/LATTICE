"""Matching-engine invariants. If these fail, every later result is fiction."""

from __future__ import annotations

from lattice.book import OrderBook
from lattice.types import Side, TIF


def test_fifo_at_a_level():
    b = OrderBook()
    b.submit("a", Side.BID, 100, 10, ts_ns=1)
    b.submit("b", Side.BID, 100, 10, ts_ns=2)
    _, trades = b.submit("t", Side.ASK, 100, 10, ts_ns=3, tif=TIF.IOC)
    assert len(trades) == 1
    assert trades[0].maker_id == "a"
    assert trades[0].qty == 10
    assert b.level_qty(Side.BID, 100) == 10


def test_price_priority():
    b = OrderBook()
    b.submit("far", Side.BID, 99, 10, ts_ns=1)
    b.submit("near", Side.BID, 100, 10, ts_ns=2)
    _, trades = b.market("t", Side.ASK, 10, ts_ns=3)
    assert trades[0].maker_id == "near"
    assert trades[0].price_ticks == 100


def test_partial_fill_rests_remainder():
    b = OrderBook()
    b.submit("m", Side.ASK, 101, 50, ts_ns=1)
    order, trades = b.submit("t", Side.BID, 101, 20, ts_ns=2, tif=TIF.IOC)
    assert order.remaining == 0
    assert trades[0].qty == 20
    assert b.level_qty(Side.ASK, 101) == 30


def test_ioc_does_not_rest():
    b = OrderBook()
    order, trades = b.submit("t", Side.BID, 100, 10, ts_ns=1, tif=TIF.IOC)
    assert not trades
    assert b.best_bid() is None
    assert order.remaining == 0 or b.level_qty(Side.BID, 100) == 0


def test_cancel_removes_liquidity():
    b = OrderBook()
    o, _ = b.submit("m", Side.BID, 100, 25, ts_ns=1)
    assert b.cancel(o.order_id) is not None
    assert b.best_bid() is None
    assert b.cancel(o.order_id) is None


def test_queue_position_at_fill():
    b = OrderBook(mm_trader_id="mm")
    b.submit("bg", Side.ASK, 101, 40, ts_ns=1)
    mm, _ = b.submit("mm", Side.ASK, 101, 10, ts_ns=2)
    assert mm.queue_at_insert == 40
    _, trades = b.market("t", Side.BID, 45, ts_ns=3)
    mm_fill = [t for t in trades if t.maker_id == "mm"]
    assert mm_fill
    assert mm_fill[0].maker_queue_at_fill == 0
    assert mm_fill[0].maker_queue_at_insert == 40


def test_spread_never_inverted_after_resting_limits():
    b = OrderBook()
    b.submit("a", Side.BID, 100, 10, ts_ns=1)
    b.submit("b", Side.ASK, 102, 10, ts_ns=2)
    assert b.best_bid() == 100
    assert b.best_ask() == 102
    assert b.spread_ticks() == 2
    assert b.mid_ticks() == 101


def test_crossing_limit_becomes_a_trade():
    b = OrderBook()
    b.submit("m", Side.ASK, 101, 10, ts_ns=1)
    _, trades = b.submit("t", Side.BID, 101, 10, ts_ns=2)
    assert trades and trades[0].price_ticks == 101
    assert b.best_ask() is None


def test_fok_all_or_nothing():
    b = OrderBook()
    b.submit("m", Side.ASK, 101, 10, ts_ns=1)
    _, trades = b.submit_fok("t", Side.BID, 101, 50, ts_ns=2)
    assert trades == []
    assert b.level_qty(Side.ASK, 101) == 10
    _, trades = b.submit_fok("t", Side.BID, 101, 10, ts_ns=3)
    assert len(trades) == 1
    assert b.best_ask() is None


def test_cancel_all_by_trader():
    b = OrderBook()
    b.submit("mm", Side.BID, 99, 10, ts_ns=1)
    b.submit("mm", Side.ASK, 101, 10, ts_ns=2)
    b.submit("bg", Side.BID, 98, 10, ts_ns=3)
    gone = b.cancel_all("mm")
    assert len(gone) == 2
    assert b.best_bid() == 98
    assert b.best_ask() is None


def test_snapshot_flags_mm_qty():
    b = OrderBook(mm_trader_id="mm")
    b.submit("mm", Side.BID, 100, 7, ts_ns=1)
    b.submit("bg", Side.BID, 100, 3, ts_ns=2)
    snap = b.snapshot(3)
    assert snap.bids[0].mm_qty == 7
    assert snap.bids[0].qty == 10
