// The plots of a Mallard run report: one SVG per series of the embedded data, drawn in the browser.
// Each series is a line through the last value of each bucket of progress rows and, where a bucket holds more
// than one row, a band between their least and largest values.
(function () {
  "use strict";
  var NS = "http://www.w3.org/2000/svg";
  var W = 460, H = 210, L = 62, R = 12, T = 12, B = 34;

  function el(name, attrs, parent) {
    var e = document.createElementNS(NS, name);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }

  function fmt(v) {
    if (v === null || !isFinite(v)) return "-";
    var a = Math.abs(v);
    if (a !== 0 && (a < 1e-3 || a >= 1e5)) return v.toExponential(2);
    return String(Number(v.toPrecision(4)));
  }

  function ticks(lo, hi, n) {
    if (!(hi > lo)) return [lo];
    var step = Math.pow(10, Math.floor(Math.log10((hi - lo) / n)));
    var err = (hi - lo) / n / step;
    step *= err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1;
    var out = [];
    for (var v = Math.ceil(lo / step) * step; v <= hi + 1e-9 * step; v += step) out.push(Math.abs(v) < 1e-12 * step ? 0 : v);
    return out;
  }

  function finite(a) { return a.filter(function (v) { return v !== null && isFinite(v); }); }

  function plot(s, parent) {
    var lo = s.lo || s.y, hi = s.hi || s.y;
    var ys = finite(lo.concat(hi)), ts = finite(s.t);
    if (!ys.length || !ts.length) return;
    var ymin = Math.min.apply(null, ys), ymax = Math.max.apply(null, ys);
    var log = ymin > 0 && ymax / ymin > 1e3;
    var f = log ? Math.log10 : function (v) { return v; };
    var a = f(ymin), b = f(ymax);
    if (a === b) { var d = a === 0 ? 1 : Math.abs(a) * 0.05; a -= d; b += d; }
    var t0 = Math.min.apply(null, ts), t1 = Math.max.apply(null, ts);
    if (t0 === t1) t1 = t0 + 1;
    var X = function (t) { return L + (t - t0) / (t1 - t0) * (W - L - R); };
    var Y = function (v) { return T + (b - f(v)) / (b - a) * (H - T - B); };

    var fig = document.createElement("figure");
    var cap = document.createElement("figcaption");
    cap.textContent = s.name + (s.unit && s.unit !== "-" ? " [" + s.unit + "]" : "") + (log ? ", log scale" : "");
    fig.appendChild(cap);
    var svg = el("svg", { viewBox: "0 0 " + W + " " + H, role: "img", "aria-label": cap.textContent }, fig);
    var yt = log ? ticks(Math.ceil(a), Math.floor(b), 4) : ticks(a, b, 4);
    yt.forEach(function (v) {
      var y = T + (b - v) / (b - a) * (H - T - B);
      el("line", { x1: L, x2: W - R, y1: y, y2: y, "class": "grid" }, svg);
      el("text", { x: L - 6, y: y + 4, "class": "tick", "text-anchor": "end" }, svg).textContent = fmt(log ? Math.pow(10, v) : v);
    });
    ticks(t0, t1, 4).forEach(function (v) {
      var x = X(v);
      el("line", { x1: x, x2: x, y1: T, y2: H - B, "class": "grid" }, svg);
      el("text", { x: x, y: H - B + 15, "class": "tick", "text-anchor": "middle" }, svg).textContent = fmt(v);
    });
    el("text", { x: (L + W - R) / 2, y: H - 4, "class": "tick", "text-anchor": "middle" }, svg).textContent = "t [" + (s.tunit || "s") + "]";
    el("rect", { x: L, y: T, width: W - L - R, height: H - T - B, "class": "frame" }, svg);

    var ok = function (i) { return s.t[i] !== null && lo[i] !== null && hi[i] !== null && (!log || lo[i] > 0); };
    if (s.lo) {
      var top = [], bottom = [];
      for (var i = 0; i < s.t.length; i++) {
        if (!ok(i)) continue;
        top.push(X(s.t[i]).toFixed(1) + "," + Y(hi[i]).toFixed(1));
        bottom.unshift(X(s.t[i]).toFixed(1) + "," + Y(lo[i]).toFixed(1));
      }
      el("polygon", { points: top.concat(bottom).join(" "), "class": "band" }, svg);
    }
    var pts = [];
    for (var j = 0; j < s.t.length; j++) {
      if (s.t[j] !== null && s.y[j] !== null && (!log || s.y[j] > 0)) pts.push(X(s.t[j]).toFixed(1) + "," + Y(s.y[j]).toFixed(1));
    }
    el("polyline", { points: pts.join(" "), "class": "line" }, svg);

    var cursor = el("line", { y1: T, y2: H - B, "class": "cursor", visibility: "hidden" }, svg);
    var readout = document.createElement("div");
    readout.className = "readout";
    readout.textContent = " ";
    fig.appendChild(readout);
    svg.addEventListener("mousemove", function (ev) {
      var r = svg.getBoundingClientRect();
      var t = t0 + ((ev.clientX - r.left) / r.width * W - L) / (W - L - R) * (t1 - t0);
      var best = -1, dist = Infinity;
      for (var k = 0; k < s.t.length; k++) {
        if (s.t[k] === null) continue;
        var dd = Math.abs(s.t[k] - t);
        if (dd < dist) { dist = dd; best = k; }
      }
      if (best < 0) return;
      cursor.setAttribute("x1", X(s.t[best]));
      cursor.setAttribute("x2", X(s.t[best]));
      cursor.setAttribute("visibility", "visible");
      var text = "step " + s.step[best] + ", t = " + fmt(s.t[best]) + ": " + fmt(s.y[best]);
      if (s.lo && lo[best] !== hi[best]) text += " (" + fmt(lo[best]) + " to " + fmt(hi[best]) + " over the bucket)";
      readout.textContent = text;
    });
    svg.addEventListener("mouseleave", function () {
      cursor.setAttribute("visibility", "hidden");
      readout.textContent = " ";
    });
    parent.appendChild(fig);
  }

  var node = document.getElementById("mallard-series");
  if (!node) return;
  var data = JSON.parse(node.textContent);
  data.groups.forEach(function (g, i) {
    var parent = document.getElementById("plots-" + i);
    if (!parent) return;
    g.series.forEach(function (s) {
      var axis = data.axes[s.axis];
      s.t = axis.t;
      s.step = axis.step;
      s.tunit = data.tunit;
      plot(s, parent);
    });
  });
  var note = document.getElementById("plots-nojs");
  if (note) note.hidden = true;
})();
