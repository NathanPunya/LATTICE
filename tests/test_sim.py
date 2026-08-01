from lattice.sim import SimConfig, Simulator, run_sim


def test_sim_is_deterministic():
    cfg = SimConfig(seed=42, horizon_s=8.0, snapshot_s=0.2, max_frames=40)
    a = Simulator(cfg).run()
    b = Simulator(cfg).run()
    assert a.metrics.pnl == b.metrics.pnl
    assert a.metrics.n_fills == b.metrics.n_fills
    assert a.trades_n == b.trades_n


def test_sim_produces_a_book_and_path():
    res = run_sim(seed=1, horizon_s=10.0, strategy="avellaneda_stoikov")
    assert res.frames
    assert res.metrics.equity_path
    assert res.frames[0].bids and res.frames[0].asks
    assert res.trades_n > 0


def test_informed_flow_marks_fills():
    res = run_sim(
        seed=9,
        horizon_s=12.0,
        strategy="naive",
        informed_lambda=6.0,
        noise_lambda=3.0,
    )
    assert res.metrics.informed_fills >= 0
    # With this much informed flow the book should trade.
    assert res.trades_n > 10


def test_jsonable_payload_has_attribution():
    payload = run_sim(seed=3, horizon_s=8.0).to_jsonable()
    attr = payload["metrics"]["attribution"]
    assert "spread_pnl" in attr
    assert "inventory_pnl" in attr
    assert payload["frames"]
    assert payload["path"]["equity"]


def test_options_overlay_credits_premium():
    res = run_sim(
        seed=5,
        horizon_s=8.0,
        options_enabled=True,
        hedge_mode="mm",
        option_qty_short=10,
    )
    assert res.metrics.attribution.option_premium > 0
    assert any("Sold" in n for n in res.notes)


def test_mm_quotes_rest_inside_the_spread():
    res = run_sim(seed=1, horizon_s=8.0, strategy="avellaneda_stoikov")
    quoted = 0
    for f in res.frames:
        if f.mm_bid is None or f.mm_ask is None:
            continue
        quoted += 1
        assert f.mm_bid < f.mm_ask
        if f.ask is not None:
            assert f.mm_bid < f.ask + 1e-9
        if f.bid is not None:
            assert f.mm_ask > f.bid - 1e-9
    assert quoted > 10


def test_gm_runs_on_two_state_value():
    cfg = SimConfig(
        seed=2,
        horizon_s=8.0,
        strategy="glosten_milgrom",
        value_process="two_state",
    )
    res = Simulator(cfg).run()
    assert res.metrics.extra["true_high"] in (True, False)
    assert 0.0 < res.metrics.extra["p_high"] < 1.0
