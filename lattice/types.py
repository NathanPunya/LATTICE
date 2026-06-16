"""Shared types for the matching engine.

Prices live in integer ticks for the entire core. Float conversion happens
only at the I/O boundary — the same discipline a production matching engine
uses to avoid binary-fraction rounding bugs that would otherwise create
phantom crosses or ghost fills.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Optional


class Side(IntEnum):
    BID = 1
    ASK = -1

    def opposite(self) -> Side:
        return Side.ASK if self is Side.BID else Side.BID


class TIF(str, Enum):
    GTC = "GTC"  # good till cancel
    IOC = "IOC"  # immediate or cancel
    FOK = "FOK"  # fill or kill (all-or-nothing)


class Liquidity(str, Enum):
    MAKER = "maker"
    TAKER = "taker"


class EventKind(str, Enum):
    NOISE_MO = "noise_mo"
    INFORMED_MO = "informed_mo"
    BG_LIMIT = "bg_limit"
    BG_CANCEL = "bg_cancel"
    MM_REQUOTE = "mm_requote"
    OPTION_HEDGE = "option_hedge"
    SNAPSHOT = "snapshot"


@dataclass(slots=True)
class Order:
    order_id: int
    trader_id: str
    side: Side
    price_ticks: int
    qty: int
    remaining: int
    ts_ns: int
    tif: TIF = TIF.GTC
    # Queue rank at the level at insert time. 0 = head of queue.
    queue_at_insert: int = 0

    @property
    def filled(self) -> int:
        return self.qty - self.remaining


@dataclass(slots=True)
class Trade:
    trade_id: int
    ts_ns: int
    price_ticks: int
    qty: int
    aggressor_side: Side
    maker_id: str
    taker_id: str
    maker_order_id: int
    taker_order_id: int
    # How many shares were ahead of the maker at this level when they posted.
    maker_queue_at_insert: int
    # Shares still ahead of the maker immediately before this fill.
    maker_queue_at_fill: int
    is_mm_maker: bool = False
    is_mm_taker: bool = False
    informed: bool = False


@dataclass(slots=True)
class BookLevel:
    price_ticks: int
    qty: int
    n_orders: int
    mm_qty: int = 0


@dataclass(slots=True)
class BookSnapshot:
    ts_ns: int
    bids: list[BookLevel]
    asks: list[BookLevel]
    last_trade_ticks: Optional[int] = None

    @property
    def best_bid(self) -> Optional[int]:
        return self.bids[0].price_ticks if self.bids else None

    @property
    def best_ask(self) -> Optional[int]:
        return self.asks[0].price_ticks if self.asks else None

    @property
    def mid_ticks(self) -> Optional[float]:
        if self.best_bid is None or self.best_ask is None:
            return None
        return 0.5 * (self.best_bid + self.best_ask)

    @property
    def spread_ticks(self) -> Optional[int]:
        if self.best_bid is None or self.best_ask is None:
            return None
        return self.best_ask - self.best_bid


@dataclass(slots=True)
class Fill:
    """A fill attributed to a particular trader (the market maker)."""

    ts_ns: int
    side: Side  # MM's side
    price_ticks: int
    qty: int
    liquidity: Liquidity
    informed: bool
    queue_at_insert: int
    queue_at_fill: int
    mid_at_fill_ticks: float
    trade_id: int


@dataclass
class TickSize:
    size: float = 0.01

    def to_ticks(self, px: float) -> int:
        return int(round(px / self.size))

    def to_price(self, ticks: int | float) -> float:
        return float(ticks) * self.size
