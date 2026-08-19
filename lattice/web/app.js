const $ = (id) => document.getElementById(id);

const state = {
  data: null,
  i: 0,
  playing: false,
  timer: null,
};

function fmt(x, d = 2) {
  if (x == null || Number.isNaN(x)) return "—";
  const n = Number(x);
  const sign = n > 0 ? "+" : "";
  return sign + n.toFixed(d);
}

function fmtPx(x, d = 2) {
  if (x == null || Number.isNaN(x)) return "—";
  return Number(x).toFixed(d);
}

function cls(x) {
  return x >= 0 ? "pos" : "neg";
}

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function resize(canvas) {
  const r = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(r.width * dpr));
  canvas.height = Math.max(1, Math.floor(r.height * dpr));
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w: r.width, h: r.height };
}

function lineChart(canvas, series, colors, x0 = 0, labels = [], signed = false) {
  const { ctx, w, h } = resize(canvas);
  ctx.clearRect(0, 0, w, h);
  const pad = { l: 8, r: 56, t: 10, b: 8 };
  const innerW = w - pad.l - pad.r;
  const innerH = h - pad.t - pad.b;
  let lo = Infinity, hi = -Infinity;
  for (const s of series) {
    for (const v of s) {
      if (v == null) continue;
      lo = Math.min(lo, v);
      hi = Math.max(hi, v);
    }
  }
  if (!Number.isFinite(lo)) return;
  if (hi - lo < 1e-9) { hi += 1; lo -= 1; }
  const padY = (hi - lo) * 0.08;
  lo -= padY; hi += padY;
  const n = Math.max(...series.map((s) => s.length), 1);
  const xAt = (i) => pad.l + (i / Math.max(n - 1, 1)) * innerW;
  const yAt = (v) => pad.t + (1 - (v - lo) / (hi - lo)) * innerH;

  ctx.strokeStyle = "#1c2533";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.l, yAt(0));
  ctx.lineTo(w - pad.r, yAt(0));
  ctx.stroke();

  series.forEach((s, k) => {
    ctx.strokeStyle = colors[k];
    ctx.lineWidth = k === 0 ? 1.6 : 1.2;
    ctx.beginPath();
    let started = false;
    s.forEach((v, i) => {
      if (v == null) return;
      const x = xAt(i), y = yAt(v);
      if (!started) { ctx.moveTo(x, y); started = true; }
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  });

  const xi = Math.min(Math.max(x0, 0), n - 1);
  const x = xAt(xi);
  ctx.strokeStyle = "#e0b35c";
  ctx.lineWidth = 1;
  ctx.setLineDash([3, 3]);
  ctx.beginPath();
  ctx.moveTo(x, pad.t);
  ctx.lineTo(x, h - pad.b);
  ctx.stroke();
  ctx.setLineDash([]);

  const callouts = [];
  series.forEach((s, k) => {
    const v = s[xi];
    if (v == null || Number.isNaN(v)) return;
    callouts.push({
      y: yAt(v),
      v,
      color: colors[k],
      name: labels[k] || "",
    });
    ctx.fillStyle = colors[k];
    ctx.beginPath();
    ctx.arc(x, yAt(v), 3.2, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "#07090c";
    ctx.lineWidth = 1;
    ctx.stroke();
  });

  callouts.sort((a, b) => a.y - b.y);
  for (let k = 1; k < callouts.length; k += 1) {
    if (callouts[k].y - callouts[k - 1].y < 13) {
      callouts[k].y = callouts[k - 1].y + 13;
    }
  }

  ctx.font = "11px IBM Plex Mono, monospace";
  ctx.textBaseline = "middle";
  const right = x > w - pad.r - 8;
  callouts.forEach((c) => {
    const text = `${c.name ? c.name + " " : ""}${signed ? fmt(c.v) : fmtPx(c.v)}`;
    const tw = ctx.measureText(text).width;
    const boxW = tw + 8;
    const boxH = 13;
    let bx = right ? x - 8 - boxW : x + 8;
    bx = Math.max(2, Math.min(bx, w - boxW - 2));
    const by = Math.max(pad.t, Math.min(c.y - boxH / 2, h - pad.b - boxH));
    ctx.fillStyle = "rgba(7, 9, 12, 0.88)";
    ctx.fillRect(bx, by, boxW, boxH);
    ctx.fillStyle = c.color;
    ctx.textAlign = "left";
    ctx.fillText(text, bx + 4, by + boxH / 2 + 0.5);
  });
}

function barChart(canvas, labels, values, colorFn) {
  const { ctx, w, h } = resize(canvas);
  ctx.clearRect(0, 0, w, h);
  const pad = { l: 8, r: 8, t: 10, b: 22 };
  const n = values.length;
  if (!n) return;
  const max = Math.max(...values.map(Math.abs), 1e-6);
  const bw = (w - pad.l - pad.r) / n;
  values.forEach((v, i) => {
    const x = pad.l + i * bw + 4;
    const zero = pad.t + (h - pad.t - pad.b) / 2;
    const mag = (Math.abs(v) / max) * ((h - pad.t - pad.b) / 2);
    ctx.fillStyle = colorFn(v, i);
    if (v >= 0) ctx.fillRect(x, zero - mag, bw - 8, mag);
    else ctx.fillRect(x, zero, bw - 8, mag);
    ctx.fillStyle = "#7a8696";
    ctx.font = "10px IBM Plex Mono, monospace";
    ctx.textAlign = "center";
    ctx.fillText(labels[i], x + (bw - 8) / 2, h - 6);
  });
}

function midAt(frames, t) {
  if (!frames.length) return null;
  let lo = 0;
  let hi = frames.length - 1;
  while (lo < hi) {
    const m = (lo + hi) >> 1;
    if (frames[m].t < t) lo = m + 1;
    else hi = m;
  }
  return frames[lo].mid;
}

function liveStats(d, i) {
  const frames = d.frames || [];
  const f = frames[i] || {};
  const t = f.t || 0;
  const fills = (d.fills || []).filter((x) => x.t <= t + 1e-9);
  const fee = (d.config && d.config.taker_fee) || 0.0002;
  const spot = f.mid || 0;
  let spread = 0;
  let cash = 0;
  let inv = 0;
  for (const x of fills) {
    const mid = x.mid != null ? x.mid : spot;
    if (x.side === "bid") {
      cash -= x.px * x.qty;
      inv += x.qty;
      spread += (mid - x.px) * x.qty;
    } else {
      cash += x.px * x.qty;
      inv -= x.qty;
      spread += (x.px - mid) * x.qty;
    }
    if (x.liq === "taker") cash -= fee * x.qty;
    else cash += fee * 0.4 * x.qty;
  }
  const trading = cash + inv * spot;
  const inventoryPnl = trading - spread;
  const opt = f.equity != null ? f.equity - trading : 0;
  const eqPath = frames.slice(0, i + 1).map((x) => x.equity);
  let peak = eqPath[0] || 0;
  let dd = 0;
  for (const x of eqPath) {
    peak = Math.max(peak, x);
    dd = Math.min(dd, x - peak);
  }
  let sharpe = 0;
  if (eqPath.length > 2) {
    const diffs = [];
    for (let k = 1; k < eqPath.length; k += 1) diffs.push(eqPath[k] - eqPath[k - 1]);
    const mu = diffs.reduce((s, v) => s + v, 0) / diffs.length;
    const varr = diffs.reduce((s, v) => s + (v - mu) * (v - mu), 0) / Math.max(diffs.length - 1, 1);
    const sd = Math.sqrt(varr);
    sharpe = sd > 1e-12 ? (mu / sd) * Math.sqrt(diffs.length) : 0;
  }
  const horizons = [0.5, 1.0, 2.0, 5.0, 10.0];
  const markouts = horizons.map((h) => {
    const vals = [];
    for (const x of fills) {
      if (x.t + h > t + 1e-9) continue;
      const m0 = x.mid != null ? x.mid : midAt(frames, x.t);
      const m1 = midAt(frames, x.t + h);
      if (m0 == null || m1 == null) continue;
      const sign = x.side === "bid" ? 1 : -1;
      vals.push(sign * (m1 - m0) * x.qty);
    }
    const mean = vals.length ? vals.reduce((s, v) => s + v, 0) / vals.length : 0;
    const total = vals.reduce((s, v) => s + v, 0);
    return { horizon_s: h, mean, total, n: vals.length };
  });
  const mk1 = markouts.find((p) => Math.abs(p.horizon_s - 1) < 1e-9);
  return {
    pnl: f.equity || 0,
    sharpe,
    max_drawdown: dd,
    n_fills: fills.length,
    informed_fills: fills.filter((x) => x.informed).length,
    attribution: {
      spread_pnl: spread,
      inventory_pnl: inventoryPnl,
      markout_1s: mk1 ? mk1.total : 0,
      fees: 0,
      option_premium: 0,
      option_mtm: opt,
    },
    markouts,
    opt,
  };
}

function renderKpis(m) {
  const a = m.attribution;
  const cells = [
    ["P&L", fmt(m.pnl), cls(m.pnl)],
    ["Sharpe", fmt(m.sharpe), cls(m.sharpe)],
    ["Spread", fmt(a.spread_pnl), cls(a.spread_pnl)],
    ["Inventory", fmt(a.inventory_pnl), cls(a.inventory_pnl)],
    ["Markout 1s", fmt(a.markout_1s), cls(a.markout_1s)],
    ["Max DD", fmt(m.max_drawdown), "neg"],
    ["Fills", String(m.n_fills), ""],
    ["Informed", String(m.informed_fills), ""],
  ];
  $("kpis").innerHTML = cells
    .map(([k, v, c]) => `<div class="cell"><div class="k">${k}</div><div class="v ${c}">${v}</div></div>`)
    .join("");
}

function renderLadder(frame) {
  if (!frame) { $("ladder").innerHTML = ""; return; }
  const maxQ = Math.max(
    1,
    ...(frame.asks || []).map((x) => x.qty),
    ...(frame.bids || []).map((x) => x.qty)
  );
  const askRows = [...(frame.asks || [])].reverse().map((lv) => {
    const ours = lv.mm > 0;
    return `<div class="lvl ask ${ours ? "ours" : ""}">
      <div class="px">${lv.px.toFixed(2)}</div>
      <div class="bar"><i style="width:${(100 * lv.qty) / maxQ}%"></i></div>
      <div>${lv.qty}${ours ? " *" : ""}</div>
    </div>`;
  }).join("");
  const bidRows = (frame.bids || []).map((lv) => {
    const ours = lv.mm > 0;
    return `<div class="lvl bid ${ours ? "ours" : ""}">
      <div class="px">${lv.px.toFixed(2)}</div>
      <div class="bar"><i style="width:${(100 * lv.qty) / maxQ}%"></i></div>
      <div>${lv.qty}${ours ? " *" : ""}</div>
    </div>`;
  }).join("");
  const mid = frame.mid != null ? frame.mid.toFixed(2) : "—";
  const spr = frame.spread_ticks != null ? `${frame.spread_ticks} tk` : "";
  $("ladder").innerHTML = `${askRows}<div class="midline">${mid}  ${spr}</div>${bidRows}`;
}

function renderFills(fills, t) {
  const rows = (fills || [])
    .filter((f) => f.t <= t + 1e-9)
    .slice(-12)
    .reverse()
    .map((f) => `<div class="fill ${f.side}">
      <span>${f.t.toFixed(2)}s</span>
      <span>${f.side}</span>
      <span>${f.px.toFixed(2)} × ${f.qty}</span>
      <span>${f.liq}</span>
      <span class="${f.informed ? "inf" : ""}">${f.informed ? "informed" : "noise"}</span>
    </div>`)
    .join("");
  $("fills").innerHTML = rows || "<div class='muted'>no fills yet</div>";
}

function drawAll() {
  const d = state.data;
  if (!d) return;
  const frames = d.frames || [];
  const i = Math.min(state.i, Math.max(frames.length - 1, 0));
  const f = frames[i] || {};
  $("clock").textContent = `${(f.t || 0).toFixed(2)}s`;
  const asof = $("attr-asof");
  if (asof) asof.textContent = `as of ${(f.t || 0).toFixed(2)}s`;
  $("inv-readout").textContent = `${f.inventory >= 0 ? "+" : ""}${f.inventory}` +
    (f.target_inv ? `  tgt ${f.target_inv.toFixed(0)}` : "");
  $("eq-readout").textContent = fmt(f.equity);

  const live = liveStats(d, i);
  renderKpis(live);

  const mids = frames.map((x) => x.mid);
  const fairs = frames.map((x) => x.fair);
  const bids = frames.map((x) => x.mm_bid);
  const asks = frames.map((x) => x.mm_ask);
  lineChart($("price"), [fairs, mids, bids, asks], ["#e0b35c", "#8aa4c8", "#3dcc8a", "#ff5f62"], i, ["fair", "mid", "bid", "ask"]);
  const setPx = (id, v) => { const el = $(id); if (el) el.textContent = fmtPx(v); };
  setPx("px-fair", f.fair);
  setPx("px-mid", f.mid);
  setPx("px-bid", f.mm_bid);
  setPx("px-ask", f.mm_ask);

  const eq = frames.map((x) => x.equity);
  const inv = frames.map((x) => x.inventory);
  lineChart($("equity"), [eq], ["#e0b35c"], i, ["eq"], true);
  lineChart($("inv"), [inv], ["#5aa8ff"], i, ["inv"], true);

  barChart(
    $("attr"),
    ["spread", "inv", "opt"],
    [live.attribution.spread_pnl, live.attribution.inventory_pnl, live.opt],
    (v) => (v >= 0 ? "#3dcc8a" : "#ff5f62")
  );
  barChart(
    $("markout"),
    live.markouts.map((p) => `${p.horizon_s}s`),
    live.markouts.map((p) => p.mean),
    (v) => (v >= 0 ? "#3dcc8a" : "#ff5f62")
  );

  renderLadder(f);
  renderFills(d.fills, f.t || 0);
}

function setPaused(paused) {
  if (paused && state.timer) {
    clearInterval(state.timer);
    state.timer = null;
  }
  state.playing = !paused;
  const btn = $("playpause");
  if (btn) btn.textContent = paused ? "Play" : "Pause";
}

function play() {
  if (!state.data || !state.data.frames.length) return;
  if (state.i >= state.data.frames.length - 1) state.i = 0;
  setPaused(false);
  if (state.timer) clearInterval(state.timer);
  state.timer = setInterval(() => {
    if (state.i >= state.data.frames.length - 1) {
      setPaused(true);
      return;
    }
    state.i += 1;
    $("scrub").value = String(state.i);
    drawAll();
  }, 28);
}

function togglePlay() {
  if (!state.data) return;
  if (state.playing) setPaused(true);
  else play();
}

async function runSim() {
  $("status").textContent = "simulating…";
  $("run").disabled = true;
  setPaused(true);
  try {
    const body = {
      seed: Number($("seed").value),
      horizon_s: Number($("horizon").value),
      strategy: $("strategy").value,
      latency_ms: Number($("latency").value),
      informed_lambda: Number($("informed").value),
      noise_lambda: Number($("noise").value),
      as_gamma: Number($("gamma").value),
      options_enabled: $("options").checked,
      hedge_mode: $("hedge").value,
    };
    const res = await fetch("/api/simulate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error(await res.text());
    state.data = await res.json();
    state.i = 0;
    $("scrub").max = String(Math.max((state.data.frames || []).length - 1, 0));
    $("scrub").value = "0";
    $("status").textContent = `${state.data.strategy} · ${state.data.trades_n} trades`;
    setPaused(true);
    drawAll();
    play();
  } catch (err) {
    $("status").textContent = "error";
    console.error(err);
    alert(err.message || err);
  } finally {
    $("run").disabled = false;
  }
}

async function runCompare() {
  $("status").textContent = "monte carlo…";
  $("compare").disabled = true;
  try {
    const res = await fetch("/api/compare", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ seeds: 5, horizon_s: Math.min(Number($("horizon").value), 30) }),
    });
    const data = await res.json();
    const rows = data.rows || [];
    $("compare-table").innerHTML = `<table>
      <tr>
        <th>strategy</th><th>mean P&L</th><th>std</th><th>Sharpe</th><th>spread</th><th>markout 1s</th>
      </tr>
      ${rows.map((r) => `<tr>
        <td>${r.strategy}</td>
        <td class="${cls(r.mean_pnl)}">${fmt(r.mean_pnl)}</td>
        <td>${r.std_pnl.toFixed(2)}</td>
        <td class="${cls(r.mean_sharpe)}">${fmt(r.mean_sharpe)}</td>
        <td class="${cls(r.mean_spread_pnl)}">${fmt(r.mean_spread_pnl)}</td>
        <td class="${cls(r.mean_markout_1s)}">${fmt(r.mean_markout_1s)}</td>
      </tr>`).join("")}
    </table>`;
    $("status").textContent = "comparison ready";
  } catch (err) {
    console.error(err);
    $("status").textContent = "error";
  } finally {
    $("compare").disabled = false;
  }
}

$("run").addEventListener("click", runSim);
$("compare").addEventListener("click", runCompare);
if ($("playpause")) $("playpause").addEventListener("click", togglePlay);
if ($("scrub")) {
  $("scrub").addEventListener("input", (e) => {
    state.i = Number(e.target.value);
    setPaused(true);
    drawAll();
  });
}
window.addEventListener("resize", drawAll);

runSim();
