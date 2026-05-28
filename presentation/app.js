/* =========================================================================
   Continuum — presentation interactions (vanilla JS, no external deps)
   ========================================================================= */
(() => {
  "use strict";

  /* ---------------------------------------------------------------------
     EDIT ME — talk data. Replace the LiveCodeBench placeholders with the
     measured numbers before presenting.
     accuracy arrays are aligned to K_AXIS = [1,2,4,8,16,32,64,128]
  --------------------------------------------------------------------- */
  const K_AXIS = [1, 2, 4, 8, 16, 32, 64, 128];
  const RESULTS = {
    onpolicy: 0.452,                                                  // reference ceiling (dashed)
    flat: [0.451, 0.449, 0.444, 0.430, 0.402, 0.331, 0.214, 0.118],  // flat clip (Trajectory)
    sdpo: [0.453, 0.452, 0.450, 0.447, 0.439, 0.421, 0.388, 0.341],  // SDPO (ours)
    rows: [
      { name: "On-policy (reference)", k8: "0.45", k32: "0.45", k128: "\u2014", band: "\u00b10.01", ours: false },
      { name: "Flat clip \u00b7 \u03b5=0.2 (Trajectory)", k8: "0.43", k32: "0.33", k128: "0.12", band: "0.10\u20130.45", ours: false },
      { name: "SDPO \u00b7 entropy-scaled (ours)", k8: "0.45", k32: "0.42", k128: "0.34", band: "0.40\u20130.47", ours: true },
    ],
  };

  /* ---------- concrete palette (SVG attrs don't resolve CSS var()) ------ */
  const C = {
    paper: "#fbfaf7", ink: "#1b1b19", inkSoft: "#3a3935", muted: "#6f6e68", faint: "#9a988f",
    line: "rgba(27,27,25,0.14)", lineSoft: "rgba(27,27,25,0.07)",
    accent: "#3340c8", accentDeep: "#232c8f", accentWash: "rgba(51,64,200,0.16)", accentWash2: "rgba(51,64,200,0.08)",
    rust: "#c0531a", rustWash: "rgba(192,83,26,0.12)", good: "#1f8a5b",
  };

  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  const SVGNS = "http://www.w3.org/2000/svg";
  const el = (tag, attrs = {}, parent) => {
    const n = document.createElementNS(SVGNS, tag);
    for (const k in attrs) n.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(n);
    return n;
  };
  // text with reliable fill (inline style beats CSS)
  const txt = (parent, x, y, str, o = {}) => {
    const t = el("text", { x, y }, parent);
    if (o.size) t.setAttribute("font-size", o.size);
    if (o.anchor) t.setAttribute("text-anchor", o.anchor);
    if (o.weight) t.setAttribute("font-weight", o.weight);
    t.style.fill = o.fill || C.muted;
    if (o.family) t.style.fontFamily = o.family;
    t.textContent = str;
    return t;
  };
  const lerp = (a, b, t) => a + (b - a) * t;
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const ONVIEW = {};
  const CAP = new URLSearchParams(location.search).has("cap"); // static-frame capture mode

  /* ====================== nav: progress, dots, keyboard ================= */
  const sections = $$("main .slide");
  const progress = $("#progress");
  const dotsNav = $("#dots");
  sections.forEach((sec) => {
    const b = document.createElement("button");
    b.innerHTML = `<span class="lbl">${sec.dataset.label || ""}</span><span class="pip"></span>`;
    b.addEventListener("click", () => sec.scrollIntoView({ behavior: "smooth" }));
    dotsNav.appendChild(b);
  });
  const pips = $$("#dots button");
  let current = 0;
  function onScroll() {
    const y = window.scrollY;
    const h = document.documentElement.scrollHeight - window.innerHeight;
    progress.style.width = `${clamp((y / h) * 100, 0, 100)}%`;
    let idx = 0;
    sections.forEach((sec, i) => { if (sec.offsetTop <= y + window.innerHeight * 0.45) idx = i; });
    if (idx !== current) {
      current = idx;
      pips.forEach((p, i) => p.setAttribute("aria-current", i === idx ? "true" : "false"));
    }
  }
  window.addEventListener("scroll", onScroll, { passive: true });
  onScroll();
  function go(dir) { sections[clamp(current + dir, 0, sections.length - 1)].scrollIntoView({ behavior: "smooth" }); }
  window.addEventListener("keydown", (e) => {
    if (["ArrowRight", "ArrowDown", "PageDown", " "].includes(e.key)) { e.preventDefault(); go(1); }
    else if (["ArrowLeft", "ArrowUp", "PageUp"].includes(e.key)) { e.preventDefault(); go(-1); }
    else if (e.key === "Home") { e.preventDefault(); sections[0].scrollIntoView({ behavior: "smooth" }); }
    else if (e.key === "End") { e.preventDefault(); sections[sections.length - 1].scrollIntoView({ behavior: "smooth" }); }
  });
  const keyhint = $("#keyhint");
  setTimeout(() => (keyhint.style.opacity = "0"), 6000);
  window.addEventListener("keydown", () => (keyhint.style.opacity = "0"));

  /* ====================== reveal + viz observers ======================= */
  const revealIO = new IntersectionObserver((entries) => {
    entries.forEach((en) => { if (en.isIntersecting) en.target.classList.add("in"); });
  }, { threshold: 0.18 });
  $$(".reveal").forEach((n) => revealIO.observe(n));

  const seen = new WeakSet();
  const vizIO = new IntersectionObserver((entries) => {
    entries.forEach((en) => {
      if (en.isIntersecting && !seen.has(en.target)) {
        seen.add(en.target);
        (ONVIEW[en.target.dataset.viz] || (() => {}))(en.target);
      }
    });
  }, { threshold: 0.25 });

  /* ====================== hero constellation =========================== */
  (function hero() {
    const cv = $("#heroCanvas");
    if (!cv) return;
    const ctx = cv.getContext("2d");
    let w, h, dpr, nodes = [], raf, running = true;
    function resize() {
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      const r = cv.parentElement.getBoundingClientRect();
      w = r.width; h = r.height;
      cv.width = w * dpr; cv.height = h * dpr;
      cv.style.width = w + "px"; cv.style.height = h + "px";
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const n = Math.round(clamp((w * h) / 22000, 28, 80));
      nodes = Array.from({ length: n }, () => ({
        x: Math.random() * w, y: Math.random() * h,
        vx: (Math.random() - 0.5) * 0.26, vy: (Math.random() - 0.5) * 0.26,
      }));
    }
    function tick() {
      if (!running) return;
      ctx.clearRect(0, 0, w, h);
      for (const p of nodes) {
        p.x += p.vx; p.y += p.vy;
        if (p.x < 0 || p.x > w) p.vx *= -1;
        if (p.y < 0 || p.y > h) p.vy *= -1;
      }
      for (let i = 0; i < nodes.length; i++)
        for (let j = i + 1; j < nodes.length; j++) {
          const a = nodes[i], b = nodes[j], dx = a.x - b.x, dy = a.y - b.y, d = Math.hypot(dx, dy);
          if (d < 130) {
            ctx.strokeStyle = `rgba(51,64,200,${0.10 * (1 - d / 130)})`;
            ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
          }
        }
      ctx.fillStyle = "rgba(51,64,200,0.38)";
      for (const p of nodes) { ctx.beginPath(); ctx.arc(p.x, p.y, 1.7, 0, Math.PI * 2); ctx.fill(); }
      if (!CAP) raf = requestAnimationFrame(tick);
    }
    resize(); tick();
    window.addEventListener("resize", resize);
    new IntersectionObserver((e) => {
      running = e[0].isIntersecting;
      if (running) tick(); else cancelAnimationFrame(raf);
    }, { threshold: 0.02 }).observe(cv);
  })();

  /* ====================== problem: drift figure ======================== */
  ONVIEW.drift = function (box) {
    const W = 1000, H = 260, svg = el("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%" }, box);
    const pad = 30;
    el("line", { x1: pad, y1: H - pad, x2: W - pad, y2: H - pad, stroke: C.line }, svg);
    el("line", { x1: pad, y1: pad, x2: pad, y2: H - pad, stroke: C.line }, svg);
    txt(svg, W - pad, H - 10, "time since deployment \u2192", { anchor: "end", size: 12, fill: C.faint });
    const world = [];
    for (let i = 0; i <= 100; i++) {
      const t = i / 100, y = (H - pad) - (28 + t * 150 + Math.sin(t * 6) * 10);
      world.push([pad + t * (W - 2 * pad), y]);
    }
    const agentY = (H - pad) - 40;
    let gap = `M ${pad} ${agentY} `; world.forEach(([x, y]) => (gap += `L ${x} ${y} `)); gap += `L ${W - pad} ${agentY} Z`;
    el("path", { d: gap, fill: C.rustWash }, svg);
    const aLine = el("line", { x1: pad, y1: agentY, x2: pad, y2: agentY, stroke: C.ink, "stroke-width": 2.5, "stroke-dasharray": "2 5" }, svg);
    let wd = `M ${world[0][0]} ${world[0][1]} `; world.forEach(([x, y]) => (wd += `L ${x} ${y} `));
    const wLine = el("path", { d: wd, fill: "none", stroke: C.accent, "stroke-width": 3 }, svg);
    const len = wLine.getTotalLength(); wLine.style.strokeDasharray = len; wLine.style.strokeDashoffset = len;
    const lw = txt(svg, W - pad - 6, world[100][1] - 10, "the world / task distribution", { anchor: "end", size: 14, weight: 600, fill: C.accent }); lw.style.opacity = 0;
    const la = txt(svg, pad + 12, agentY - 12, "frozen agent", { size: 14, weight: 600, fill: C.inkSoft }); la.style.opacity = 0;
    const lg = txt(svg, W * 0.62, agentY - 70, "drift = lost value", { size: 14, weight: 600, fill: C.rust, anchor: "middle" }); lg.style.opacity = 0;
    let t0 = null;
    (function anim(ts) {
      if (!t0) t0 = ts;
      const k = CAP ? 1 : clamp((ts - t0) / 1400, 0, 1);
      aLine.setAttribute("x2", pad + k * (W - 2 * pad));
      wLine.style.strokeDashoffset = len * (1 - k);
      [lw, la, lg].forEach((n) => (n.style.opacity = clamp((k - 0.6) / 0.4, 0, 1)));
      if (k < 1 && !CAP) requestAnimationFrame(anim);
    })(performance.now());
  };

  /* ====================== infra: async-RL loop ========================= */
  ONVIEW.loop = function (stage) {
    const W = 1000, H = 360, svg = el("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%" }, stage);
    const cy = 210;
    const defs = el("defs", {}, svg);
    [["arrowB", C.accent], ["arrowR", C.rust]].forEach(([id, c]) => {
      const m = el("marker", { id, markerWidth: 9, markerHeight: 9, refX: 7, refY: 3, orient: "auto", markerUnits: "strokeWidth" }, defs);
      el("path", { d: "M0,0 L7,3 L0,6 Z", fill: c }, m);
    });
    const box = (x, y, w, h, fill, stroke) => el("rect", { x, y, width: w, height: h, rx: 14, fill, stroke, "stroke-width": 1.5 }, svg);

    // fleet
    box(40, cy - 70, 200, 140, "#fff", C.line);
    txt(svg, 140, cy - 44, "Agent fleet", { anchor: "middle", size: 16, weight: 600, fill: C.ink, family: '"Iowan Old Style", Palatino, Georgia, serif' });
    for (let i = 0; i < 3; i++) el("rect", { x: 66 + i * 8, y: cy - 22 + i * 6, width: 140, height: 30, rx: 7, fill: i === 0 ? C.accentWash : "#fff", stroke: C.line }, svg);
    el("circle", { cx: 84, cy: cy + 17, r: 5, fill: C.good }, svg);
    txt(svg, 152, cy + 22, "coding \u00b7 support \u00b7 ops", { anchor: "middle", size: 12, fill: C.muted });

    // buffer
    box(405, cy - 55, 190, 110, "#fff", C.line);
    txt(svg, 500, cy - 28, "Rollout buffer", { anchor: "middle", size: 16, weight: 600, fill: C.ink, family: '"Iowan Old Style", Palatino, Georgia, serif' });
    for (let i = 0; i < 3; i++) el("rect", { x: 430, y: cy - 8 + i * 18, width: 140, height: 12, rx: 6, fill: C.accentWash, stroke: C.line }, svg);
    txt(svg, 500, cy + 75, "rollout + reward, pooled", { anchor: "middle", size: 12, fill: C.muted });

    // trainer
    box(760, cy - 70, 200, 140, C.accentWash2, C.accent);
    txt(svg, 860, cy - 42, "Async RL trainer", { anchor: "middle", size: 16, weight: 600, fill: C.accentDeep, family: '"Iowan Old Style", Palatino, Georgia, serif' });
    txt(svg, 860, cy - 18, "SDPO update", { anchor: "middle", size: 13, weight: 700, fill: C.accent });
    const core = el("circle", { cx: 860, cy: cy + 18, r: 16, fill: "none", stroke: C.accent, "stroke-width": 2 }, svg);
    el("circle", { cx: 860, cy: cy + 18, r: 6, fill: C.accent }, svg);
    txt(svg, 860, cy + 75, "staleness-aware, always on", { anchor: "middle", size: 12, fill: C.muted });

    // connectors
    const fwd1 = el("path", { d: `M 240 ${cy} L 405 ${cy}`, fill: "none", stroke: C.accent, "stroke-width": 2, "marker-end": "url(#arrowB)" }, svg);
    const fwd2 = el("path", { d: `M 595 ${cy} L 760 ${cy}`, fill: "none", stroke: C.accent, "stroke-width": 2, "marker-end": "url(#arrowB)" }, svg);
    const back = el("path", { d: `M 860 ${cy - 70} C 860 60, 140 60, 140 ${cy - 70}`, fill: "none", stroke: C.rust, "stroke-width": 2, "marker-end": "url(#arrowR)" }, svg);
    txt(svg, 322, cy - 12, "rollouts", { anchor: "middle", size: 12, weight: 600, fill: C.accent });
    txt(svg, 677, cy - 12, "rollouts", { anchor: "middle", size: 12, weight: 600, fill: C.accent });
    txt(svg, 500, 50, "updated weights \u2192 broadcast to fleet", { anchor: "middle", size: 12, weight: 600, fill: C.rust });

    // traveling dots
    const movers = [];
    [{ p: fwd1, c: C.accent, n: 3, rev: false, dur: 1700 }, { p: fwd2, c: C.accent, n: 3, rev: false, dur: 1700 }, { p: back, c: C.rust, n: 4, rev: true, dur: 3200 }]
      .forEach((cfg) => {
        const L = cfg.p.getTotalLength();
        for (let i = 0; i < cfg.n; i++) movers.push({ c: el("circle", { r: 4, fill: cfg.c }, svg), L, path: cfg.p, off: i / cfg.n, rev: cfg.rev, dur: cfg.dur });
      });
    let start = null;
    (function frame(ts) {
      if (!start) start = ts;
      const e = CAP ? 850 : ts - start;
      movers.forEach((m) => {
        let f = ((e / m.dur) + m.off) % 1; if (m.rev) f = 1 - f;
        const pt = m.path.getPointAtLength(f * m.L); m.c.setAttribute("cx", pt.x); m.c.setAttribute("cy", pt.y);
      });
      const s = 1 + 0.25 * Math.sin(e / 300);
      core.setAttribute("r", 16 * s); core.setAttribute("opacity", clamp(1.2 - 0.5 * s, 0.3, 1));
      if (!CAP) requestAnimationFrame(frame);
    })(performance.now());
  };

  /* ====================== SDPO: entropy clip plot ====================== */
  (function entropy() {
    const host = $("#entropyPlot");
    if (!host) return;
    const W = 560, H = 300, N = 60, svg = el("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%" }, host);
    let s = 7; const rnd = () => { s = (s * 1103515245 + 12345) & 0x7fffffff; return s / 0x7fffffff; };
    const Hraw = [];
    for (let i = 0; i < N; i++) {
      const t = i / (N - 1);
      let v = 1.1 + 0.55 * Math.sin(t * 7.5 + 0.6) + 0.4 * Math.sin(t * 17 + 1.2) + (rnd() - 0.5) * 0.18;
      Hraw.push(Math.max(0.18, v));
    }
    const Hmean = Hraw.reduce((a, b) => a + b, 0) / N, Hmin = Math.min(...Hraw), Hmax = Math.max(...Hraw);
    const padL = 40, padR = 14;
    const xOf = (i) => padL + (i / (N - 1)) * (W - padL - padR);
    const tTop = 24, tBot = 116, yH = (v) => tBot - ((v - Hmin) / (Hmax - Hmin)) * (tBot - tTop);
    const bTop = 152, bBot = 284, bMid = (bTop + bBot) / 2, RR = 0.55, yR = (r) => bMid - ((r - 1) / RR) * ((bBot - bTop) / 2);

    txt(svg, padL, 14, "teacher entropy  H\u209c", { size: 11, fill: C.muted });
    txt(svg, padL, 142, "clip band on ratio  r\u209c   (center = 1.0)", { size: 11, fill: C.muted });

    let ad = `M ${xOf(0)} ${tBot} `; Hraw.forEach((v, i) => (ad += `L ${xOf(i)} ${yH(v)} `)); ad += `L ${xOf(N - 1)} ${tBot} Z`;
    el("path", { d: ad, fill: C.rustWash }, svg);
    let ld = `M ${xOf(0)} ${yH(Hraw[0])} `; Hraw.forEach((v, i) => (ld += `L ${xOf(i)} ${yH(v)} `));
    el("path", { d: ld, fill: "none", stroke: C.rust, "stroke-width": 1.6 }, svg);
    el("line", { x1: padL, y1: yH(Hmean), x2: W - padR, y2: yH(Hmean), stroke: C.rust, "stroke-dasharray": "3 4", "stroke-width": 1, opacity: 0.7 }, svg);
    txt(svg, W - padR, yH(Hmean) - 4, "Hmean", { anchor: "end", size: 11, fill: C.rust });

    el("line", { x1: padL, y1: bMid, x2: W - padR, y2: bMid, stroke: C.line }, svg);
    txt(svg, padL - 6, bMid + 3, "1.0", { anchor: "end", size: 11, fill: C.faint });
    const flatHi = el("line", { x1: padL, x2: W - padR, stroke: C.faint, "stroke-dasharray": "4 4", "stroke-width": 1.2 }, svg);
    const flatLo = el("line", { x1: padL, x2: W - padR, stroke: C.faint, "stroke-dasharray": "4 4", "stroke-width": 1.2 }, svg);
    const flatLbl = txt(svg, W - padR, 0, "flat \u03b5\u2080 (Trajectory)", { anchor: "end", size: 11, fill: C.faint });
    const band = el("path", { fill: C.accentWash, stroke: C.accent, "stroke-width": 1.4 }, svg);

    const alphaIn = $("#alpha"), eps0In = $("#eps0"), alphaVal = $("#alphaVal"), eps0Val = $("#eps0Val");
    function redraw() {
      const alpha = parseFloat(alphaIn.value), eps0 = parseFloat(eps0In.value);
      alphaVal.textContent = alpha.toFixed(2); eps0Val.textContent = eps0.toFixed(2);
      flatHi.setAttribute("y1", yR(1 + eps0)); flatHi.setAttribute("y2", yR(1 + eps0));
      flatLo.setAttribute("y1", yR(1 - eps0)); flatLo.setAttribute("y2", yR(1 - eps0));
      flatLbl.setAttribute("y", yR(1 + eps0) - 4);
      let up = "", dn = "";
      for (let i = 0; i < N; i++) {
        const eps = eps0 * Math.pow(Hraw[i] / Hmean, alpha), x = xOf(i);
        up += `${i === 0 ? "M" : "L"} ${x} ${yR(1 + eps)} `;
        dn = `L ${x} ${yR(1 - eps)} ` + dn;
      }
      band.setAttribute("d", up + dn + "Z");
    }
    [alphaIn, eps0In].forEach((c) => c.addEventListener("input", redraw));
    redraw();

    const minI = Hraw.indexOf(Hmin);
    el("line", { x1: xOf(minI), y1: tTop, x2: xOf(minI), y2: bBot, stroke: C.accent, "stroke-width": 1, "stroke-dasharray": "2 3", opacity: 0.5 }, svg);
    txt(svg, clamp(xOf(minI), 70, W - 90), bBot + 13, "peaked teacher \u2192 tight clip", { anchor: "middle", size: 11, fill: C.accent });
  })();

  /* ====================== SDPO: variance band ========================== */
  ONVIEW.variance = function (host) {
    const W = 560, H = 300, svg = el("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%" }, host);
    const padL = 44, padB = 44, top = 18, yAcc = (a) => top + (1 - a / 0.5) * (H - padB - top);
    el("line", { x1: padL, y1: top, x2: padL, y2: H - padB, stroke: C.line }, svg);
    [0, 0.1, 0.2, 0.3, 0.4, 0.5].forEach((a) => {
      el("line", { x1: padL, y1: yAcc(a), x2: W - 10, y2: yAcc(a), stroke: C.lineSoft }, svg);
      txt(svg, padL - 8, yAcc(a) + 3, a.toFixed(1), { anchor: "end", size: 11, fill: C.faint });
    });
    txt(svg, padL - 32, top - 4, "eval acc", { size: 11, fill: C.faint });
    const cols = [
      { x: W * 0.34, label: "Flat clip", sub: "Trajectory", lo: 0.10, hi: 0.45, color: C.rust },
      { x: W * 0.72, label: "SDPO", sub: "ours", lo: 0.40, hi: 0.47, color: C.accent },
    ];
    let sd = 99; const rnd = () => { sd = (sd * 1103515245 + 12345) & 0x7fffffff; return sd / 0x7fffffff; };
    cols.forEach((c) => {
      const r = el("rect", { x: c.x - 46, y: yAcc(c.hi), width: 92, height: 0, rx: 8, fill: c.color, opacity: 0.13 }, svg);
      const targetH = yAcc(c.lo) - yAcc(c.hi); let t0 = null;
      requestAnimationFrame(function gr(ts) { if (!t0) t0 = ts; const k = clamp((ts - t0) / 700, 0, 1); r.setAttribute("height", targetH * k); if (k < 1) requestAnimationFrame(gr); });
      txt(svg, c.x, H - padB + 20, c.label, { anchor: "middle", size: 14, weight: 700, fill: C.ink });
      txt(svg, c.x, H - padB + 35, c.sub, { anchor: "middle", size: 11, fill: c.color });
      txt(svg, c.x, yAcc(c.hi) - 8, `${c.lo.toFixed(2)}\u2013${c.hi.toFixed(2)}`, { anchor: "middle", size: 11, fill: c.color });
      for (let i = 0; i < 11; i++) {
        const a = lerp(c.lo + 0.01, c.hi - 0.01, rnd()), jitter = (rnd() - 0.5) * 60;
        const dot = el("circle", { cx: c.x + jitter, cy: yAcc(a), r: 4, fill: c.color, opacity: 0 }, svg);
        setTimeout(() => { dot.style.transition = "opacity .4s"; dot.setAttribute("opacity", 0.85); }, 400 + i * 70);
      }
    });
  };

  /* ====================== Results: accuracy vs K ======================= */
  ONVIEW.acc = function (host) {
    const W = 560, H = 300, svg = el("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%" }, host);
    const padL = 44, padB = 44, padT = 18, padR = 14;
    const xs = K_AXIS.map((k) => Math.log2(k)), xmin = Math.min(...xs), xmax = Math.max(...xs);
    const xOf = (k) => padL + ((Math.log2(k) - xmin) / (xmax - xmin)) * (W - padL - padR);
    const ymin = 0.05, ymax = 0.48, yOf = (a) => padT + (1 - (a - ymin) / (ymax - ymin)) * (H - padB - padT);
    [0.1, 0.2, 0.3, 0.4].forEach((a) => {
      el("line", { x1: padL, y1: yOf(a), x2: W - padR, y2: yOf(a), stroke: C.lineSoft }, svg);
      txt(svg, padL - 8, yOf(a) + 3, a.toFixed(1), { anchor: "end", size: 11, fill: C.faint });
    });
    K_AXIS.forEach((k) => txt(svg, xOf(k), H - padB + 16, String(k), { anchor: "middle", size: 11, fill: C.faint }));
    txt(svg, W / 2, H - 8, "staleness  K  (policy lag, log scale)", { anchor: "middle", size: 11, fill: C.muted });
    txt(svg, padL - 32, padT - 4, "pass@1", { size: 11, fill: C.faint });
    el("line", { x1: padL, y1: yOf(RESULTS.onpolicy), x2: W - padR, y2: yOf(RESULTS.onpolicy), stroke: C.good, "stroke-dasharray": "4 4", "stroke-width": 1.2, opacity: 0.85 }, svg);
    txt(svg, W - padR, yOf(RESULTS.onpolicy) - 5, "on-policy", { anchor: "end", size: 11, fill: C.good });
    function line(data, color, label, labelY) {
      let d = ""; K_AXIS.forEach((k, i) => (d += `${i === 0 ? "M" : "L"} ${xOf(k)} ${yOf(data[i])} `));
      const p = el("path", { d, fill: "none", stroke: color, "stroke-width": 3, "stroke-linejoin": "round" }, svg);
      const L = p.getTotalLength(); p.style.strokeDasharray = L; p.style.strokeDashoffset = L;
      requestAnimationFrame(() => { p.style.transition = "stroke-dashoffset 1.4s ease"; p.style.strokeDashoffset = 0; });
      K_AXIS.forEach((k, i) => {
        const c = el("circle", { cx: xOf(k), cy: yOf(data[i]), r: 3.5, fill: color, opacity: 0 }, svg);
        setTimeout(() => { c.style.transition = "opacity .3s"; c.setAttribute("opacity", 1); }, 1400 + i * 40);
      });
      const t = txt(svg, xOf(K_AXIS[7]) - 4, labelY, label, { anchor: "end", size: 13, weight: 700, fill: color }); t.style.opacity = 0;
      setTimeout(() => { t.style.transition = "opacity .4s"; t.style.opacity = 1; }, 1500);
    }
    const yOfLast = (arr) => padT + (1 - (arr[7] - ymin) / (ymax - ymin)) * (H - padB - padT);
    line(RESULTS.sdpo, C.accent, "SDPO (ours)", yOfLast(RESULTS.sdpo) - 8);
    line(RESULTS.flat, C.rust, "flat clip", yOfLast(RESULTS.flat) + 16);
  };

  /* ====================== results table =============================== */
  (function table() {
    const body = $("#resultsBody"); if (!body) return;
    RESULTS.rows.forEach((r) => {
      const tr = document.createElement("tr"); if (r.ours) tr.className = "ours";
      tr.innerHTML = `<td>${r.name}</td><td>${r.k8}</td><td>${r.k32}</td><td>${r.k128}</td><td>${r.band}</td>`;
      body.appendChild(tr);
    });
  })();

  /* ====================== kernels: reward + counters =================== */
  ONVIEW.reward = function () { const m = $("#rewardMarker"); if (m) setTimeout(() => (m.style.left = "60%"), 250); };
  ONVIEW.count = function (card) {
    const node = card.querySelector("[data-count]"); if (!node) return;
    const target = parseInt(node.dataset.count, 10);
    const unit = node.querySelector(".unit"); const suffix = unit ? unit.outerHTML : "";
    let t0 = null;
    requestAnimationFrame(function step(ts) {
      if (!t0) t0 = ts;
      const k = clamp((ts - t0) / 900, 0, 1), v = Math.round(target * (1 - Math.pow(1 - k, 3)));
      node.innerHTML = v + suffix;
      if (k < 1) requestAnimationFrame(step);
    });
  };

  /* ====================== hook viz to viewport ======================== */
  function hook(sel, name) { const n = $(sel); if (n) { n.dataset.viz = name; vizIO.observe(n); } }
  hook("#driftbox", "drift");
  hook("#loopStage", "loop");
  hook("#variancePlot", "variance");
  hook("#accPlot", "acc");
  hook("#rewardBar", "reward");
  $$(".card [data-count]").forEach((n) => { const card = n.closest(".card"); card.dataset.viz = "count"; vizIO.observe(card); });

  /* ---- capture mode (dev aid): ?cap=<section-id> reveals + fires all viz ---- */
  const cap = new URLSearchParams(location.search).get("cap");
  if (cap) {
    document.documentElement.style.scrollBehavior = "auto";
    $$(".reveal").forEach((n) => n.classList.add("in"));
    [["#driftbox", "drift"], ["#loopStage", "loop"], ["#variancePlot", "variance"], ["#accPlot", "acc"], ["#rewardBar", "reward"]]
      .forEach(([sel, name]) => { const n = $(sel); if (n && !seen.has(n)) { seen.add(n); ONVIEW[name](n); } });
    $$(".card [data-count]").forEach((n) => { const c = n.closest(".card"); if (!seen.has(c)) { seen.add(c); ONVIEW.count(c); } });
    if (cap === "all") {
      sections.forEach((s) => { s.style.minHeight = "auto"; s.style.paddingTop = "56px"; s.style.paddingBottom = "56px"; });
      window.scrollTo(0, 0);
    } else {
      const target = document.getElementById(cap);
      if (target) {
        sections.forEach((s) => { if (s.id !== cap) s.style.display = "none"; });
        target.style.minHeight = "auto";
        target.style.paddingTop = "48px";
        window.scrollTo(0, 0);
      }
    }
  }
})();
