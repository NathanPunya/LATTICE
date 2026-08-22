# Lattice

Event-driven limit-order-book simulator for electronic market making.

Lattice is a matching engine plus a synthetic market (latent value, noise and informed flow, latency) and quoting policies that post into the book. Fills are attributed into spread, inventory, markout, fees, and an optional short-option overlay.

## Layout

```
lattice/
  book.py         FIFO matching engine, integer ticks, queue tracking
  strategies.py   Naive · inventory skew · Avellaneda–Stoikov · Glosten–Milgrom
  options.py      Black–Scholes, Greeks, implied vol, short-option contract
  sim.py          Event-driven world: latent value, noise, informed flow, latency
  analytics.py    Spread / inventory / markout / fee attribution, session Sharpe
  server.py       Research terminal API
  cli.py          run · compare · options · serve
lattice/web/      Book ladder, playback, Monte Carlo
tests/            Engine invariants, put-call parity, determinism
```

## Matching engine

Prices are integer ticks in the core; floats exist only at the I/O boundary. Resting orders are price-time (FIFO). Queue position is recorded at insert and at fill. IOC, FOK, and cancel-all are first-class. Strategies cannot invent fills — they only post, cancel, or take through the book.

## Market

Latent fair value is either Brownian motion in trading time or a two-state Glosten–Milgrom $V \in \{V_L, V_H\}$.

- Noise traders arrive as a Poisson process and hit a random side.
- Informed traders observe $V$ and trade only when they have edge.
- Background limit orders rest around $V$ so the agent has a queue to stand in.
- Agent quotes are decided on a timer, then applied after a latency delay, so they can be stale when they rest.

## Strategies

| Name | CLI flag | Behavior |
|---|---|---|
| Naive | `naive` | Symmetric quotes around the mid |
| Inventory skew | `inventory_skew` | Mid quotes pulled against inventory |
| Avellaneda–Stoikov | `avellaneda_stoikov` | Reservation price and optimal spread (2008) |
| Glosten–Milgrom | `glosten_milgrom` | Quotes fair conditional on being hit (1985) |

**Avellaneda–Stoikov.** Reservation price and optimal spread, with fill intensity $\lambda(\delta) = A e^{-k\delta}$:

$$
r(s,q,t) = s - q\gamma\sigma^2(T-t)
$$

$$
\delta^a + \delta^b = \gamma\sigma^2(T-t) + \frac{2}{\gamma}\ln\left(1+\frac{\gamma}{k}\right)
$$

Quotes sit around $r$, not $s$. The closed form assumes a single liquidity provider and a Brownian mid; the simulator does not.

**Glosten–Milgrom.** A fraction $\mu$ of flow is informed. After each trade the MM updates $p = \mathbb{P}(V = V_H \mid \text{history})$. The ask is $\mathbb{E}[V \mid \text{buy}]$, the bid is $\mathbb{E}[V \mid \text{sell}]$. The two-state value process matches this model; Brownian value is a misspecification.

**Options overlay.** A short call has position delta $-\Phi(d_1)N$. Target cash inventory is $+\Phi(d_1)N$. Hedge modes: `none`, `frictionless` (pay half-spread), `taker` (hit the book on a timer), or `mm` (skew maker quotes so hedging earns the spread).

## P&L attribution

Every fill is split into:

| Piece | What it is |
|---|---|
| Spread | Half-spread captured vs. the contemporaneous mid |
| Inventory | Residual mark-to-market to the terminal mid |
| Markout | Mid move 0.5s / 1s / 2s / 5s / 10s after the fill, split informed vs. noise |
| Fees | Taker fee vs. maker rebate |
| Option | Premium received minus live mark (when the overlay is on) |

## Install

Python 3.11+. If the system Python is older, use `uv`:

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Usage

```bash
lattice run --strategy avellaneda_stoikov --horizon 40
lattice run --strategy glosten_milgrom --seed 3
lattice run --strategy naive --options --hedge mm
lattice compare --seeds 7 --horizon 30
lattice options --seeds 5
lattice serve          # http://127.0.0.1:8000
pytest
```

| Command | What it does |
|---|---|
| `run` | Single path. `--strategy`, `--seed`, `--horizon`, `--latency`, `--options`, `--hedge`, `--out` |
| `compare` | Monte Carlo across the four quoting policies |
| `options` | Same short option under `none` / `taker` / `mm` hedge modes |
| `serve` | Research terminal: book ladder, fair value vs. mid vs. quotes, inventory and equity paths, markouts, Monte Carlo |

`run` selects `value_process=two_state` for Glosten–Milgrom and Brownian motion otherwise.

## Design notes

- Integer ticks: matching is exact; no binary-float prices in the book.
- Strategies cannot fill themselves: P&L only comes from the engine.
- Latency is applied after the decision, so quotes can rest stale.
- Volatility is in trading time. A short session is a compressed slice of a day, not calendar seconds of 20% annualized equity vol.
- Markouts are split informed / noise so the informed-flow flag is testable.

## License

MIT
