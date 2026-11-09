// Ashi Liquid Glass FULL engine. Source: github.com/the-Ashkanian/ashi-liquid-glass
// Chromium: Snell refraction, RGB dispersion, rim light. Others: blur + rim.
(() => {
  const IOR = 1.5, BEZEL = 30, THICK = 40, DISP = 0.10, BLUR = 1.5;
  const BIG = 60000;
  const chromium = /Chrome\//.test(navigator.userAgent);
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.style.cssText = 'position:absolute;width:0;height:0'; document.body.prepend(svg);
  const PLAIN = `blur(${BLUR}px) saturate(1.8) brightness(1.12)`;

  const shiftFor = bz => {
    const th = THICK * bz / BEZEL;
    const f = t => { const u = 1 - t, q = 1 - u ** 4, a1 = Math.atan(th / bz * q ** -.75 * u ** 3), a2 = Math.asin(Math.sin(a1) / IOR); return (th * .3 + th * q ** .25) * Math.tan(a1 - a2); };
    let max = 0; for (let i = 1; i < 200; i++) max = Math.max(max, f(i / 200));
    return { f, max };
  };

  const toCanvas = d => { const c = document.createElement('canvas'); c.width = d.width; c.height = d.height; c.getContext('2d').putImageData(d, 0, 0); return c; };
  const toURL = c => new Promise((ok, no) => c.toBlob(b => {
    if (!b) return no(new Error('toBlob failed'));
    const fr = new FileReader(); fr.onload = () => ok(fr.result); fr.onerror = no; fr.readAsDataURL(b);
  }));

  function scan(w, h, r, k, fn) {
    const W = w * k, H = h * k, hw = w / 2 - r, hh = h / 2 - r;
    for (let j = 0; j < H; j++) for (let i = 0; i < W; i++) {
      const px = (i + .5) / k - w / 2, py = (j + .5) / k - h / 2;
      const qx = Math.abs(px) - hw, qy = Math.abs(py) - hh, sx = px < 0 ? -1 : 1, sy = py < 0 ? -1 : 1;
      let d, nx, ny;
      if (qx > 0 && qy > 0) { const l = Math.hypot(qx, qy); d = l - r; nx = sx * qx / l; ny = sy * qy / l; }
      else if (qx > qy) { d = qx - r; nx = sx; ny = 0; } else { d = qy - r; nx = 0; ny = sy; }
      fn((j * W + i) * 4, -d, nx, ny, px, py);
    }
  }

  const entries = new Map(), pending = new Map(); let uid = 0;
  function build(key, sp) {
    if (pending.has(key)) return pending.get(key);
    const p = (async () => {
      const { w, h, r, zoom, bz } = sp, big = w * h > BIG, rk = big ? 1 : 2;
      const rim = new ImageData(w * rk, h * rk), rd = rim.data;
      scan(w, h, r, rk, (o, depth, nx, ny) => {
        if (depth >= 40) return;
        const dot = Math.abs(nx * -.55 + ny * -.83);
        const a = (Math.exp(-depth) * (.3 + .85 * dot ** 1.4) + Math.exp(-depth / 7) * dot * .28) * Math.min(1, Math.max(0, depth * 2 + .5));
        rd[o] = rd[o + 1] = rd[o + 2] = 255; rd[o + 3] = a * 255;
      });
      let map = null, M = 1;
      if (chromium) {
        const S = shiftFor(bz), Z = 1 - 1 / zoom; M = S.max + Math.max(w, h) / 2 * Z;
        map = new ImageData(w, h); const md = map.data;
        scan(w, h, r, 1, (o, depth, nx, ny, px, py) => {
          const m = depth > 0 && depth < bz ? S.f(depth / bz) : 0;
          md[o] = 128 + 127 * (-m * nx - px * Z) / M; md[o + 1] = 128 + 127 * (-m * ny - py * Z) / M; md[o + 2] = 128; md[o + 3] = 255;
        });
      }
      const [rimU, mapU] = await Promise.all([toURL(toCanvas(rim)), map ? toURL(toCanvas(map)) : null]);
      let id = null, el = null;
      if (chromium) {
        id = 'lg' + ++uid; const s = M * 2;
        const ch = (sc, mt, n) => `<feDisplacementMap in="SourceGraphic" in2="m" scale="${sc}" xChannelSelector="R" yChannelSelector="G"/><feColorMatrix type="matrix" values="${mt}" result="${n}"/>`;
        svg.insertAdjacentHTML('beforeend', `<filter id="${id}" x="0" y="0" width="100%" height="100%" color-interpolation-filters="sRGB"><feImage href="${mapU}" x="0" y="0" width="${w}" height="${h}" preserveAspectRatio="none" result="m"/>${ch(s * (1 + DISP), '1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0', 'r')}${ch(s, '0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1 0', 'g')}${ch(s * (1 - DISP), '0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0', 'b')}<feBlend in="r" in2="g" mode="screen" result="rg"/><feBlend in="rg" in2="b" mode="screen"/></filter>`);
        el = svg.lastElementChild;
      }
      const res = { id, rim: rimU, el }; entries.set(key, res); return res;
    })();
    pending.set(key, p);
    const done = () => pending.delete(key); p.then(done, done);
    return p;
  }

  function apply(host) {
    const L = host._lg, res = host._res; if (!L || !res) return;
    const v = host._vis && res.id ? `url(#${res.id}) ${PLAIN}` : PLAIN;
    if (host._cur === v) return; host._cur = v;
    L.style.backdropFilter = L.style.webkitBackdropFilter = v;
  }

  async function init(host) {
    const w = host.offsetWidth, h = host.offsetHeight; if (!w || !h) return;
    const cs = getComputedStyle(host), g = n => parseFloat(cs.getPropertyValue(n));
    const rs = getComputedStyle(document.documentElement), za = parseFloat(rs.getPropertyValue('--lg-zoom-all')), ba = parseFloat(rs.getPropertyValue('--lg-bezel-all'));
    const r = Math.min(parseFloat(cs.borderTopLeftRadius) || 0, w / 2, h / 2);
    const zoom = za || g('--lg-zoom') || 1, bz = Math.min(ba || g('--lg-bezel') || BEZEL, Math.min(w, h) / 2);
    const key = [w, h, r.toFixed(1), zoom.toFixed(3), bz.toFixed(1)].join('|');
    host._w = w; host._h = h;
    const tok = host._tok = (host._tok || 0) + 1;
    let res = entries.get(key);
    if (!res) { try { res = await build(key, { w, h, r, zoom, bz }); } catch (e) { return; } if (tok !== host._tok) return; }
    host._res = res;
    host._lg.style.setProperty('--hl', `url(${res.rim})`);
    apply(host); gc();
  }

  let gct;
  function gc() {
    clearTimeout(gct);
    gct = setTimeout(() => {
      const used = new Set(); document.querySelectorAll('.lg[data-lg]').forEach(h => used.add(h._res));
      entries.forEach((res, key) => { if (!used.has(res)) { if (res.el) res.el.remove(); entries.delete(key); } });
    }, 1500);
  }

  const ro = new ResizeObserver(es => es.forEach(e => {
    const h = e.target; cancelAnimationFrame(h._raf);
    h._raf = requestAnimationFrame(() => {
      if (h._w && Math.abs(h.offsetWidth - h._w) <= 1 && Math.abs(h.offsetHeight - h._h) <= 1) return;
      init(h);
    });
  }));
  const io = new IntersectionObserver(es => es.forEach(e => { e.target._vis = e.isIntersecting; apply(e.target); }), { rootMargin: '300px 0px' });

  const refresh = (root = document) => root.querySelectorAll('.lg:not([data-lg])').forEach(h => {
    h.dataset.lg = 1; const L = document.createElement('i'); L.className = 'lg-layer'; h._lg = L; h.prepend(L); ro.observe(h); io.observe(h);
  });
  let rid = 0;
  const rebuild = () => { const id = ++rid, q = [...document.querySelectorAll('.lg[data-lg]')]; (function step() { if (id !== rid) return; q.splice(0, 6).forEach(h => init(h)); if (q.length) requestAnimationFrame(step); })(); };
  refresh(); window.LiquidGlass = { refresh, rebuild };
})();
