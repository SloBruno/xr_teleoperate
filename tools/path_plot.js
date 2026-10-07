// path_plot.js — offline canvas plotting for 3D point paths (no libraries, no CDN).
// Shared by tools/arm_path_web.html. 3D camera/projection adapted from
// tools/pose_compare_web.html (same orbit/zoom/views, same robot-waist axes).
"use strict";
(function (root) {
  const COL = {x: "#ff5c5c", y: "#46d27a", z: "#4ea3ff", grid: "#232836", dim: "#9aa3b2", line: "#2a2f3a"};
  const VIEWS = {front: [0, 0.05], side: [Math.PI / 2, 0.05], top: [0, Math.PI / 2 - 0.01], iso: [0.75, 0.45]};

  function setup(cv) {
    const dpr = window.devicePixelRatio || 1, w = cv.clientWidth, h = cv.clientHeight;
    if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
    const g = cv.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, w, h);
    return {g, w, h};
  }
  function niceStep(range, n) { if (!(range > 0)) return 1; const r = range / n, p = Math.pow(10, Math.floor(Math.log10(r))), m = r / p; return (m < 1.5 ? 1 : m < 3 ? 2 : m < 7 ? 5 : 10) * p; }
  // time colour: blue (start) -> green -> yellow -> red (end)
  function tcol(f, a) { const hue = 220 - 220 * Math.max(0, Math.min(1, f)); return "hsla(" + hue.toFixed(0) + ",90%,58%," + (a === undefined ? 1 : a) + ")"; }
  function valid(p) { return p && p[0] !== null && p[1] !== null && p[2] !== null; }
  function decimate(n, maxN) { const step = Math.max(1, Math.ceil(n / maxN)), idx = []; for (let i = 0; i < n; i += step) idx.push(i); if (n && idx[idx.length - 1] !== n - 1) idx.push(n - 1); return idx; }
  function bounds(paths) {
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (const P of paths) for (const p of P.pts) if (valid(p)) for (let k = 0; k < 3; k++) { lo[k] = Math.min(lo[k], p[k]); hi[k] = Math.max(hi[k], p[k]); }
    return isFinite(lo[0]) ? {lo, hi} : null;
  }
  // Draw a path as NB colour bins along time (gradient), breaking on null / time gaps.
  function gradientPath(g, map, P, width, alpha, maxN) {
    const n = P.pts.length; if (n < 2) return;
    const idx = decimate(n, maxN || 4000), t0 = P.t[0], t1 = P.t[n - 1], span = (t1 - t0) || 1, NB = 24;
    g.lineWidth = width; g.lineCap = "round"; g.lineJoin = "round";
    let b = -1, pen = false, lastT = null;
    for (const i of idx) {
      const A = valid(P.pts[i]) ? map(P.pts[i]) : null;
      const bi = Math.min(NB - 1, Math.floor((P.t[i] - t0) / span * NB));
      if (!A || (lastT !== null && P.t[i] - lastT > 0.5)) { if (pen) g.stroke(); pen = false; lastT = P.t[i]; if (!A) continue; }
      if (bi !== b && pen) { g.lineTo(A[0], A[1]); g.stroke(); pen = false; }
      if (!pen) { b = bi; g.strokeStyle = P.color || tcol((bi + 0.5) / NB, alpha); g.beginPath(); g.moveTo(A[0], A[1]); pen = true; }
      else g.lineTo(A[0], A[1]);
      lastT = P.t[i];
    }
    if (pen) g.stroke();
  }
  function endpoints(g, map, P, lbl) {
    let s = null, e = null;
    for (const p of P.pts) if (valid(p)) { s = p; break; }
    for (let i = P.pts.length - 1; i >= 0; i--) if (valid(P.pts[i])) { e = P.pts[i]; break; }
    const S = s && map(s), E = e && map(e);
    g.font = "12px system-ui";
    if (S) { g.beginPath(); g.arc(S[0], S[1], 6, 0, 2 * Math.PI); g.fillStyle = "#2f7dff"; g.fill(); g.strokeStyle = "#fff"; g.lineWidth = 1.5; g.stroke(); if (lbl) { g.fillStyle = "#cfe0ff"; g.fillText("início", S[0] + 8, S[1] - 6); } }
    if (E) { g.beginPath(); g.rect(E[0] - 5.5, E[1] - 5.5, 11, 11); g.fillStyle = "#ff4040"; g.fill(); g.strokeStyle = "#fff"; g.lineWidth = 1.5; g.stroke(); if (lbl) { g.fillStyle = "#ffd0d0"; g.fillText("fim", E[0] + 8, E[1] + 14); } }
  }

  // ---------------- 3D view ----------------
  function View3D(cv, opts) {
    this.cv = cv; this.opts = opts || {};
    this.cam = {yaw: VIEWS.iso[0], pitch: VIEWS.iso[1], dist: 1.0, tgt: [0.2, 0, 0.1]};
    this.paths = []; this.dirty = true; this.peers = [];
    const self = this, ptrs = new Map(); let pinch0 = null;
    cv.style.touchAction = "none";
    cv.addEventListener("pointerdown", (e) => { cv.setPointerCapture(e.pointerId); ptrs.set(e.pointerId, [e.clientX, e.clientY]); pinch0 = null; });
    cv.addEventListener("pointermove", (e) => {
      if (!ptrs.has(e.pointerId)) return;
      const prev = ptrs.get(e.pointerId); ptrs.set(e.pointerId, [e.clientX, e.clientY]);
      if (ptrs.size === 1) { self.cam.yaw -= (e.clientX - prev[0]) * 0.01; self.cam.pitch = Math.max(-1.55, Math.min(1.56, self.cam.pitch + (e.clientY - prev[1]) * 0.01)); }
      else if (ptrs.size === 2) { const [a, b] = [...ptrs.values()], d = Math.hypot(a[0] - b[0], a[1] - b[1]); if (pinch0) { self.cam.dist = Math.max(0.05, Math.min(10, self.cam.dist * pinch0 / d)); self.userZoom = true; } pinch0 = d; }
      self.changed();
    });
    const end = (e) => { ptrs.delete(e.pointerId); pinch0 = null; };
    cv.addEventListener("pointerup", end); cv.addEventListener("pointercancel", end);
    cv.addEventListener("wheel", (e) => { e.preventDefault(); self.userZoom = true; self.cam.dist = Math.max(0.05, Math.min(10, self.cam.dist * Math.exp(e.deltaY * 0.001))); self.changed(); }, {passive: false});
  }
  View3D.prototype.changed = function () { this.dirty = true; for (const p of this.peers) { p.cam.yaw = this.cam.yaw; p.cam.pitch = this.cam.pitch; p.dirty = true; } };
  View3D.prototype.setView = function (name) { [this.cam.yaw, this.cam.pitch] = VIEWS[name]; this.changed(); };
  View3D.prototype.fit = function () {
    const b = bounds(this.paths);
    if (!b) return;
    this.cam.tgt = [0, 1, 2].map(k => (b.lo[k] + b.hi[k]) / 2);
    const ext = Math.max(b.hi[0] - b.lo[0], b.hi[1] - b.lo[1], b.hi[2] - b.lo[2], 0.05);
    this.cam.dist = Math.max(0.15, 2.4 * ext);
    this.dirty = true;
  };
  View3D.prototype.projector = function (w, h) {
    const c = this.cam, cp = Math.cos(c.pitch);
    const eye = [c.tgt[0] + c.dist * cp * Math.cos(c.yaw), c.tgt[1] + c.dist * cp * Math.sin(c.yaw), c.tgt[2] + c.dist * Math.sin(c.pitch)];
    const f = [-cp * Math.cos(c.yaw), -cp * Math.sin(c.yaw), -Math.sin(c.pitch)];
    let r = [f[1], -f[0], 0], rn = Math.hypot(r[0], r[1]);
    if (rn < 1e-6) { r = [Math.sin(c.yaw), -Math.cos(c.yaw), 0]; rn = 1; }
    r = [r[0] / rn, r[1] / rn, 0];
    const u = [r[1] * f[2] - r[2] * f[1], r[2] * f[0] - r[0] * f[2], r[0] * f[1] - r[1] * f[0]];
    const F = 0.9 * Math.min(w, h) / (2 * Math.tan(0.45)), cx = w / 2, cy = h / 2;
    return (p) => {
      const d = [p[0] - eye[0], p[1] - eye[1], p[2] - eye[2]], zc = d[0] * f[0] + d[1] * f[1] + d[2] * f[2];
      if (zc < 0.01) return null;
      return [cx + F * (d[0] * r[0] + d[1] * r[1] + d[2] * r[2]) / zc, cy - F * (d[0] * u[0] + d[1] * u[1] + d[2] * u[2]) / zc];
    };
  };
  View3D.prototype.draw = function () {
    if (!this.dirty) return; this.dirty = false;
    const {g, w, h} = setup(this.cv), P = this.projector(w, h), self = this;
    const seg = (a, b) => { const A = P(a), B = P(b); if (A && B) { g.beginPath(); g.moveTo(A[0], A[1]); g.lineTo(B[0], B[1]); g.stroke(); } };
    const b = bounds(this.paths);
    // grid under the data: step chosen from extent
    const ext = b ? Math.max(b.hi[0] - b.lo[0], b.hi[1] - b.lo[1], 0.05) : 0.5;
    const st = niceStep(ext * 1.4, 6), z = b ? b.lo[2] - st * 0.5 : 0, cxg = Math.round(this.cam.tgt[0] / st) * st, cyg = Math.round(this.cam.tgt[1] / st) * st;
    g.strokeStyle = COL.grid; g.lineWidth = 1;
    for (let k = -5; k <= 5; k++) { seg([cxg + k * st, cyg - 5 * st, z], [cxg + k * st, cyg + 5 * st, z]); seg([cxg - 5 * st, cyg + k * st, z], [cxg + 5 * st, cyg + k * st, z]); }
    g.font = "11px system-ui"; g.fillStyle = "#5c6475";
    const gl = P([cxg + 5 * st, cyg + 5 * st, z]); if (gl) g.fillText("grade " + (st * 100).toFixed(st < 0.01 ? 1 : 0) + " cm", gl[0] + 4, gl[1]);
    // axes triad at the target (directions = robot waist frame)
    const a = Math.max(st, 0.02), o = [cxg - 5 * st, cyg - 5 * st, z];
    g.lineWidth = 2.5; g.font = "bold 12px system-ui";
    [[COL.x, [a, 0, 0], "X"], [COL.y, [0, a, 0], "Y"], [COL.z, [0, 0, a], "Z"]].forEach(([c, e, n]) => { g.strokeStyle = c; const q = [o[0] + e[0], o[1] + e[1], o[2] + e[2]]; seg(o, q); const Q = P(q); if (Q) { g.fillStyle = c; g.fillText(n, Q[0] + 3, Q[1] - 3); } });
    for (const p of this.paths) {
      // drop shadow on the grid plane helps depth perception
      if (this.opts.shadow !== false && b) { g.globalAlpha = 0.25; gradientPath(g, (q) => P([q[0], q[1], z]), Object.assign({}, p, {color: "#59606e"}), 1, 1, 1500); g.globalAlpha = 1; }
      gradientPath(g, (q) => P(q), p, p.width || 2.2, p.alpha, 4000);
    }
    for (const p of this.paths) if (p.ends !== false) endpoints(g, (q) => P(q), p, true);
    if (!b) { g.fillStyle = COL.dim; g.font = "13px system-ui"; g.fillText(this.opts.empty || "sem dados", w / 2 - 30, h / 2); }
    g.fillStyle = "#5c6475"; g.font = "11px system-ui";
    g.fillText("yaw " + (self.cam.yaw * 180 / Math.PI).toFixed(0) + "° · pitch " + (self.cam.pitch * 180 / Math.PI).toFixed(0) + "°", 6, h - 6);
  };

  // ---------------- 2D projection with equal axis scale ----------------
  // ax = [i, j] indices (0=x,1=y,2=z), flipH: draw +i to the left (so views look natural)
  function proj2d(cv, paths, ax, labels, opts) {
    opts = opts || {};
    const {g, w, h} = setup(cv), L = 44, R = 8, T = 8, B = 24, pw = w - L - R, ph = h - T - B;
    const b = bounds(paths);
    g.font = "11px system-ui";
    if (!b) { g.fillStyle = COL.dim; g.fillText("sem dados", w / 2 - 25, h / 2); return; }
    const [i, j] = ax;
    // one scale (px per m) for both axes: fit the larger relative extent, min 2 cm
    const ri = Math.max(b.hi[i] - b.lo[i], 0.02), rj = Math.max(b.hi[j] - b.lo[j], 0.02);
    const sc = Math.min(pw / (ri * 1.15), ph / (rj * 1.15));
    const ci = (b.lo[i] + b.hi[i]) / 2, cj = (b.lo[j] + b.hi[j]) / 2, sgn = opts.flipH ? -1 : 1;
    const X = (v) => L + pw / 2 + sgn * (v - ci) * sc, Y = (v) => T + ph / 2 - (v - cj) * sc;
    // grid
    const step = niceStep(Math.max(pw, ph) / sc, 6);
    g.strokeStyle = COL.line; g.lineWidth = 1; g.fillStyle = COL.dim;
    const iMin = ci - sgn * (pw / 2) / sc, iMax = ci + sgn * (pw / 2) / sc, lo_i = Math.min(iMin, iMax), hi_i = Math.max(iMin, iMax);
    for (let v = Math.ceil(lo_i / step) * step; v <= hi_i; v += step) { const x = X(v); g.beginPath(); g.moveTo(x, T); g.lineTo(x, T + ph); g.stroke(); g.fillText((v * 100).toFixed(step < 0.01 ? 1 : 0), x - 8, h - 8); }
    const jMin = cj - (ph / 2) / sc, jMax = cj + (ph / 2) / sc;
    for (let v = Math.ceil(jMin / step) * step; v <= jMax; v += step) { const y = Y(v); g.beginPath(); g.moveTo(L, y); g.lineTo(L + pw, y); g.stroke(); g.fillText((v * 100).toFixed(step < 0.01 ? 1 : 0), 4, y + 4); }
    g.fillStyle = "#c9cfdb"; g.font = "12px system-ui";
    g.fillText(labels[0] + (opts.flipH ? " ←" : " →") + " (cm)", L + pw - 92, T + ph - 6);
    g.fillText(labels[1] + " ↑", L + 4, T + 12);
    const map = (p) => [X(p[i]), Y(p[j])];
    for (const p of paths) gradientPath(g, map, p, p.width || 1.8, p.alpha, 4000);
    for (const p of paths) if (p.ends !== false) endpoints(g, map, p, false);
  }

  // ---------------- X/Y/Z vs time ----------------
  function xyzTime(cv, P, opts) {
    opts = opts || {};
    const {g, w, h} = setup(cv), L = 50, R = 8, T = 8, B = 22, pw = w - L - R, ph = h - T - B;
    g.font = "11px system-ui";
    const n = P ? P.pts.length : 0;
    if (n < 2) { g.fillStyle = COL.dim; g.fillText("sem dados", w / 2 - 25, h / 2); return; }
    const t0 = P.t[0], t1 = Math.max(P.t[n - 1], t0 + 1e-3);
    let lo = Infinity, hi = -Infinity;
    for (const p of P.pts) if (valid(p)) for (let k = 0; k < 3; k++) { lo = Math.min(lo, p[k]); hi = Math.max(hi, p[k]); }
    const pad = Math.max(0.01, (hi - lo) * 0.06); lo -= pad; hi += pad;
    const X = (t) => L + (t - t0) / (t1 - t0) * pw, Y = (v) => T + (1 - (v - lo) / (hi - lo)) * ph;
    g.strokeStyle = COL.line; g.lineWidth = 1; g.fillStyle = COL.dim;
    const ys = niceStep(hi - lo, 5);
    for (let v = Math.ceil(lo / ys) * ys; v <= hi; v += ys) { const y = Y(v); g.beginPath(); g.moveTo(L, y); g.lineTo(w - R, y); g.stroke(); g.fillText(v.toFixed(Math.max(0, -Math.floor(Math.log10(ys)))), 4, y + 4); }
    const ts = niceStep(t1 - t0, 6);
    for (let t = Math.ceil(t0 / ts) * ts; t <= t1; t += ts) { const x = X(t); g.beginPath(); g.moveTo(x, T); g.lineTo(x, T + ph); g.stroke(); g.fillText((t - t0).toFixed(ts < 1 ? 1 : 0) + " s", x - 10, h - 6); }
    const idx = decimate(n, 3000);
    ["x", "y", "z"].forEach((a, k) => {
      for (const S of [opts.raw, P]) {
        if (!S) continue;
        g.strokeStyle = COL[a]; g.lineWidth = S === P ? 1.8 : 1; g.globalAlpha = S === P ? 1 : 0.35;
        g.beginPath(); let pen = false, lastT = null;
        for (const i of idx) { const p = S.pts[i]; if (!valid(p) || (lastT !== null && S.t[i] - lastT > 0.5)) { pen = false; lastT = S.t[i]; if (!valid(p)) continue; } const x = X(S.t[i]), y = Y(p[k]); if (!pen) { g.moveTo(x, y); pen = true; } else g.lineTo(x, y); lastT = S.t[i]; }
        g.stroke();
      }
    });
    g.globalAlpha = 1;
  }

  // ---------------- cumulative length vs time ----------------
  function lengthTime(cv, t, series) {
    const {g, w, h} = setup(cv), L = 50, R = 8, T = 8, B = 22, pw = w - L - R, ph = h - T - B;
    g.font = "11px system-ui";
    if (!t || t.length < 2) { g.fillStyle = COL.dim; g.fillText("sem dados", w / 2 - 25, h / 2); return; }
    const t0 = t[0], t1 = Math.max(t[t.length - 1], t0 + 1e-3);
    let hi = 0.01; for (const s of series) for (const v of s.v) if (v > hi) hi = v;
    hi *= 1.08;
    const X = (x) => L + (x - t0) / (t1 - t0) * pw, Y = (v) => T + (1 - v / hi) * ph;
    g.strokeStyle = COL.line; g.lineWidth = 1; g.fillStyle = COL.dim;
    const ys = niceStep(hi, 5);
    for (let v = 0; v <= hi; v += ys) { const y = Y(v); g.beginPath(); g.moveTo(L, y); g.lineTo(w - R, y); g.stroke(); g.fillText(v.toFixed(Math.max(0, -Math.floor(Math.log10(ys)))), 4, y + 4); }
    const ts = niceStep(t1 - t0, 6);
    for (let x = Math.ceil(t0 / ts) * ts; x <= t1; x += ts) { const px = X(x); g.beginPath(); g.moveTo(px, T); g.lineTo(px, T + ph); g.stroke(); g.fillText((x - t0).toFixed(ts < 1 ? 1 : 0) + " s", px - 10, h - 6); }
    const idx = decimate(t.length, 3000);
    for (const s of series) { g.strokeStyle = s.color; g.lineWidth = s.width || 1.8; g.setLineDash(s.dash ? [5, 4] : []); g.beginPath(); idx.forEach((i, n) => { const x = X(t[i]), y = Y(s.v[i]); if (!n) g.moveTo(x, y); else g.lineTo(x, y); }); g.stroke(); }
    g.setLineDash([]);
  }

  root.PathPlot = {COL, VIEWS, setup, niceStep, tcol, bounds, View3D, proj2d, xyzTime, lengthTime};
})(window);
