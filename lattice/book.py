"""Price-time priority limit order book.

The book is the source of truth. Strategies never invent fills: they post
orders, the engine matches them, and fills come back as facts. That single
invariant is what makes the later P&L attribution honest — and what most
'trading bot' side projects skip.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from sortedcontainers import SortedDict

from lattice.types import BookLevel, BookSnapshot, Order, Side, TIF, Trade


@dataclass
class _Level:
    price_ticks: int
    orders: deque[Order] = field(default_factory=deque)
    total_qty: int = 0

    def enqueue(self, order: Order) -> int:
        """Return queue position (shares ahead) at insert."""
        ahead = self.total_qty
        order.queue_at_insert = ahead
        self.orders.append(order)
        self.total_qty += order.remaining
        return ahead

    def shares_ahead(self, order_id: int) -> int:
        ahead = 0
        for o in self.orders:
            if o.order_id == order_id:
                return ahead
            ahead += o.remaining
        return ahead

    def cancel(self, order_id: int) -> Optional[Order]:
        for i, o in enumerate(self.orders):
            if o.order_id == order_id:
                self.total_qty -= o.remaining
                del self.orders[i]
                return o
        return None

    def empty(self) -> bool:
        return self.total_qty <= 0


class OrderBook:
    """FIFO (price-time) book with integer tick prices.

    Bids are keyed by price (ascending SortedDict); best bid is the last key.
    Asks are keyed by price; best ask is the first key.
    """

    def __init__(self, mm_trader_id: str = "mm") -> None:
        self._bids: SortedDict[int, _Level] = SortedDict()
        self._asks: SortedDict[int, _Level] = SortedDict()
        self._orders: dict[int, Order] = {}
        self._next_order_id = 1
        self._next_trade_id = 1
        self._last_trade_ticks: Optional[int] = None
        self.mm_trader_id = mm_trader_id

    # ------------------------------------------------------------------ ids
    def _oid(self) -> int:
        oid = self._next_order_id
        self._next_order_id += 1
        return oid

    def _tid(self) -> int:
        tid = self._next_trade_id
        self._next_trade_id += 1
        return tid

    # ----------------------------------------------------------------- books
    def _book(self, side: Side) -> SortedDict[int, _Level]:
        return self._bids if side is Side.BID else self._asks

    def best_bid(self) -> Optional[int]:
        return self._bids.peekitem(-1)[0] if self._bids else None

    def best_ask(self) -> Optional[int]:
        return self._asks.peekitem(0)[0] if self._asks else None

    def mid_ticks(self) -> Optional[float]:
        bb, ba = self.best_bid(), self.best_ask()
        if bb is None or ba is None:
            return None
        return 0.5 * (bb + ba)

    def spread_ticks(self) -> Optional[int]:
        bb, ba = self.best_bid(), self.best_ask()
        if bb is None or ba is None:
            return None
        return ba - bb

    def level_qty(self, side: Side, price_ticks: int) -> int:
        lvl = self._book(side).get(price_ticks)
        return 0 if lvl is None else lvl.total_qty

    def queue_ahead(self, order_id: int) -> int:
        order = self._orders.get(order_id)
        if order is None:
            return 0
        lvl = self._book(order.side).get(order.price_ticks)
        if lvl is None:
            return 0
        return lvl.shares_ahead(order_id)

    # --------------------------------------------------------------- posting
    def submit(
        self,
        trader_id: str,
        side: Side,
        price_ticks: int,
        qty: int,
        ts_ns: int,
        tif: TIF = TIF.GTC,
        informed: bool = False,
    ) -> tuple[Order, list[Trade]]:
        if qty <= 0:
            raise ValueError("qty must be positive")
        order = Order(
            order_id=self._oid(),
            trader_id=trader_id,
            side=side,
            price_ticks=price_ticks,
            qty=qty,
            remaining=qty,
            ts_ns=ts_ns,
            tif=tif,
        )
        trades = self._match(order, ts_ns, informed=informed)

        if order.remaining > 0:
            if tif in (TIF.IOC, TIF.FOK):
                # FOK is all-or-nothing: if we matched anything, unwind it.
                # We implement FOK by checking feasibility first.
                order.remaining = 0
            else:
                self._rest(order)

        return order, trades

    def submit_fok(
        self,
        trader_id: str,
        side: Side,
        price_ticks: int,
        qty: int,
        ts_ns: int,
        informed: bool = False,
    ) -> tuple[Order, list[Trade]]:
        """Fill-or-kill: match the full size or do nothing."""
        available = self._available(side.opposite(), price_ticks, side)
        if available < qty:
            dummy = Order(
                order_id=self._oid(),
                trader_id=trader_id,
                side=side,
                price_ticks=price_ticks,
                qty=qty,
                remaining=qty,
                ts_ns=ts_ns,
                tif=TIF.FOK,
            )
            return dummy, []
        return self.submit(trader_id, side, price_ticks, qty, ts_ns, TIF.IOC, informed)

    def market(
        self,
        trader_id: str,
        side: Side,
        qty: int,
        ts_ns: int,
        informed: bool = False,
    ) -> tuple[Order, list[Trade]]:
        """Sweep the opposite book with no price limit."""
        cap = 10**12 if side is Side.BID else 0
        # A bid market order is willing to pay any ask; an ask market order
        # is willing to sell to any bid.
        limit = cap if side is Side.BID else -cap
        # Use an extreme price so every resting opposite order is in range.
        price = 10**9 if side is Side.BID else 0
        return self.submit(trader_id, side, price, qty, ts_ns, TIF.IOC, informed)

    def cancel(self, order_id: int) -> Optional[Order]:
        order = self._orders.pop(order_id, None)
        if order is None:
            return None
        lvl = self._book(order.side).get(order.price_ticks)
        if lvl is None:
            return order
        lvl.cancel(order_id)
        if lvl.empty():
            del self._book(order.side)[order.price_ticks]
        return order

    def cancel_all(self, trader_id: str) -> list[Order]:
        ids = [oid for oid, o in self._orders.items() if o.trader_id == trader_id]
        cancelled = []
        for oid in ids:
            c = self.cancel(oid)
            if c is not None:
                cancelled.append(c)
        return cancelled

    def resting_for(self, trader_id: str) -> list[Order]:
        return [o for o in self._orders.values() if o.trader_id == trader_id]

    # --------------------------------------------------------------- matching
    def _crosses(self, order: Order, other_price: int) -> bool:
        if order.side is Side.BID:
            return order.price_ticks >= other_price
        return order.price_ticks <= other_price

    def _available(self, rest_side: Side, limit_ticks: int, aggressor: Side) -> int:
        book = self._book(rest_side)
        total = 0
        if rest_side is Side.ASK:
            for px, lvl in book.items():
                if px > limit_ticks and aggressor is Side.BID:
                    break
                total += lvl.total_qty
        else:
            for px, lvl in reversed(book.items()):
                if px < limit_ticks and aggressor is Side.ASK:
                    break
                total += lvl.total_qty
        return total

    def _match(self, order: Order, ts_ns: int, informed: bool) -> list[Trade]:
        opposite = self._book(order.side.opposite())
        trades: list[Trade] = []

        while order.remaining > 0 and opposite:
            best_px = opposite.peekitem(0)[0] if order.side is Side.BID else opposite.peekitem(-1)[0]
            if not self._crosses(order, best_px):
                break
            level = opposite[best_px]
            while order.remaining > 0 and level.orders:
                maker = level.orders[0]
                take = min(order.remaining, maker.remaining)
                queue_at_fill = level.shares_ahead(maker.order_id)
                maker.remaining -= take
                order.remaining -= take
                level.total_qty -= take

                trade = Trade(
                    trade_id=self._tid(),
                    ts_ns=ts_ns,
                    price_ticks=best_px,
                    qty=take,
                    aggressor_side=order.side,
                    maker_id=maker.trader_id,
                    taker_id=order.trader_id,
                    maker_order_id=maker.order_id,
                    taker_order_id=order.order_id,
                    maker_queue_at_insert=maker.queue_at_insert,
                    maker_queue_at_fill=queue_at_fill,
                    is_mm_maker=maker.trader_id == self.mm_trader_id,
                    is_mm_taker=order.trader_id == self.mm_trader_id,
                    informed=informed,
                )
                trades.append(trade)
                self._last_trade_ticks = best_px

                if maker.remaining == 0:
                    level.orders.popleft()
                    self._orders.pop(maker.order_id, None)

            if level.empty():
                del opposite[best_px]

        return trades

    def _rest(self, order: Order) -> None:
        book = self._book(order.side)
        lvl = book.get(order.price_ticks)
        if lvl is None:
            lvl = _Level(price_ticks=order.price_ticks)
            book[order.price_ticks] = lvl
        lvl.enqueue(order)
        self._orders[order.order_id] = order

    # -------------------------------------------------------------- snapshots
    def snapshot(self, ts_ns: int, depth: int = 10) -> BookSnapshot:
        bids: list[BookLevel] = []
        asks: list[BookLevel] = []
        for px, lvl in reversed(self._bids.items()):
            mm_qty = sum(o.remaining for o in lvl.orders if o.trader_id == self.mm_trader_id)
            bids.append(
                BookLevel(
                    price_ticks=px,
                    qty=lvl.total_qty,
                    n_orders=len(lvl.orders),
                    mm_qty=mm_qty,
                )
            )
            if len(bids) >= depth:
                break
        for px, lvl in self._asks.items():
            mm_qty = sum(o.remaining for o in lvl.orders if o.trader_id == self.mm_trader_id)
            asks.append(
                BookLevel(
                    price_ticks=px,
                    qty=lvl.total_qty,
                    n_orders=len(lvl.orders),
                    mm_qty=mm_qty,
                )
            )
            if len(asks) >= depth:
                break
        return BookSnapshot(
            ts_ns=ts_ns,
            bids=bids,
            asks=asks,
            last_trade_ticks=self._last_trade_ticks,
        )
