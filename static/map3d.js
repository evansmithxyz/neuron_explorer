// 3D view of the network: layers run left to right, each drawn as a small slab of mixer heads followed by a
// sheet of its MLP neurons (one glowing point per neuron).  Traced pathways are drawn as curves with particles
// flowing from cause to effect.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const GAP = 1.7, SHEET = 4, MLP_DX = 0.22, HEAD_DX = -0.55;
const POS = [1.0, 0.58, 0.16], NEG = [0.22, 0.52, 1.0], IDLE = [0.05, 0.065, 0.09];

function pointMaterial(ring) {
  return new THREE.ShaderMaterial({
    uniforms: { uScale: { value: 600 }, uTime: { value: 0 } },
    vertexShader: `
      attribute vec3 aColor; attribute float aSize; uniform float uScale; varying vec3 vC;
      void main() {
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        gl_Position = projectionMatrix * mv;
        float px = aSize * uScale / -mv.z;
        vC = aColor * clamp(px * px, 0.3, 1.0);   // points smaller than a pixel dim instead of vanishing
        gl_PointSize = max(px, 1.0);
      }`,
    fragmentShader: ring ? `
      varying vec3 vC; uniform float uTime;
      void main() {
        float r = length(gl_PointCoord - 0.5) * 2.0;
        float ringA = smoothstep(0.16, 0.0, abs(r - 0.78 + 0.06 * sin(uTime * 3.0)));
        float core = smoothstep(0.45, 0.0, r);
        gl_FragColor = vec4(vC * (ringA + core * 0.9), 1.0);
      }` : `
      varying vec3 vC;
      void main() {
        float r = length(gl_PointCoord - 0.5);
        if (r > 0.5) discard;
        gl_FragColor = vec4(vC * smoothstep(0.5, 0.12, r), 1.0);
      }`,
    transparent: true, depthWrite: false, depthTest: false, blending: THREE.AdditiveBlending,
  });
}

export function createMap3D(root, info, cb) {
  const L = info.n_layers, N = info.d_mlp, G = Math.ceil(Math.sqrt(N));
  const canvas = document.createElement("canvas");
  const labels = document.createElement("div");
  labels.className = "labels3d";
  root.append(canvas, labels);

  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setClearColor(0x04060a);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(34, 1, 0.05, 2000);
  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true; controls.dampingFactor = 0.12; controls.zoomToCursor = true;
  controls.minDistance = 1.5; controls.maxDistance = 400;
  const center = new THREE.Vector3((L - 1) * GAP / 2 - 2.2, SHEET / 2 - 0.3, 0); // a little left of the middle: earlier words sit past the input
  function resetView() {
    controls.target.copy(center);
    // far enough back that the whole length of the network fits the current width
    // back off until both ends (the earlier-words column on the left, the output word on the right) are on screen
    const dir = new THREE.Vector3(-0.5, 0.28, 1).normalize(), p = new THREE.Vector3();
    const ends = [[-GAP * 1.5 - 4.5, SHEET / 2], [-GAP * 1.5 - 4.5, SHEET + 1], [(L - 1) * GAP + GAP * 1.7 + 1.5, SHEET / 2], [(L - 1) * GAP + GAP * 1.7 + 1.5, -0.6]];
    camera.updateProjectionMatrix();
    for (let d = 12, k = 0; k < 70; k++, d *= 1.06) {
      camera.position.copy(center).addScaledVector(dir, d); camera.lookAt(center); camera.updateMatrixWorld();
      if (ends.every(([x, y]) => { p.set(x, y, 0).project(camera); return Math.abs(p.x) < 0.93 && Math.abs(p.y) < 0.93; })) break;
    }
    controls.update();
  }
  resetView();

  const layerX = l => l * GAP;
  const neuronPos = (l, n, out = new THREE.Vector3()) =>
    out.set(layerX(l) + MLP_DX, (1 - Math.floor(n / G) / (G - 1)) * SHEET, ((n % G) / (G - 1) - 0.5) * SHEET);
  const headPos = (l, h, out = new THREE.Vector3()) => {
    const rows = info.heads[l] / 4, s = 0.42;
    return out.set(layerX(l) + HEAD_DX, SHEET / 2 + (rows / 2 - 0.5 - Math.floor(h / 4)) * s, ((h % 4) - 1.5) * s);
  };
  const IN_POS = new THREE.Vector3(-GAP * 1.5, SHEET / 2, 0), OUT_POS = new THREE.Vector3((L - 1) * GAP + GAP * 1.7, SHEET / 2, 0);
  const nodePos = n => n.kind === "neuron" ? neuronPos(n.layer, n.index) : n.kind === "head" ? headPos(n.layer, n.index)
    : n.kind === "embed" ? IN_POS.clone() : OUT_POS.clone();

  // ---- every neuron, one draw call
  const nPos = new Float32Array(L * N * 3), nCol = new Float32Array(L * N * 3), nSize = new Float32Array(L * N);
  const v = new THREE.Vector3();
  for (let l = 0; l < L; l++) for (let n = 0; n < N; n++) { neuronPos(l, n, v).toArray(nPos, (l * N + n) * 3); }
  const nGeo = new THREE.BufferGeometry();
  nGeo.setAttribute("position", new THREE.BufferAttribute(nPos, 3));
  nGeo.setAttribute("aColor", new THREE.BufferAttribute(nCol, 3));
  nGeo.setAttribute("aSize", new THREE.BufferAttribute(nSize, 1));
  const mat = pointMaterial(false), ringMat = pointMaterial(true);
  const neurons = new THREE.Points(nGeo, mat); neurons.frustumCulled = false; scene.add(neurons);

  // ---- heads
  const headIndex = []; // flat list of [layer, head]
  for (let l = 0; l < L; l++) for (let h = 0; h < info.heads[l]; h++) headIndex.push([l, h]);
  const hPos = new Float32Array(headIndex.length * 3), hCol = new Float32Array(headIndex.length * 3), hSize = new Float32Array(headIndex.length).fill(0.2);
  headIndex.forEach(([l, h], i) => headPos(l, h, v).toArray(hPos, i * 3));
  const hGeo = new THREE.BufferGeometry();
  hGeo.setAttribute("position", new THREE.BufferAttribute(hPos, 3));
  hGeo.setAttribute("aColor", new THREE.BufferAttribute(hCol, 3));
  hGeo.setAttribute("aSize", new THREE.BufferAttribute(hSize, 1));
  const heads = new THREE.Points(hGeo, mat); heads.frustumCulled = false; scene.add(heads);

  // ---- structure: sheet outlines, head slabs, the residual stream running underneath
  {
    const seg = [], col = [];
    const rect = (x, y0, y1, z0, z1, c) => {
      const p = [[x, y0, z0], [x, y1, z0], [x, y1, z1], [x, y0, z1]];
      for (let i = 0; i < 4; i++) { seg.push(...p[i], ...p[(i + 1) % 4]); col.push(...c, ...c); }
    };
    for (let l = 0; l < L; l++) {
      const full = info.layer_types[l] === "full_attention";
      rect(layerX(l) + MLP_DX, -0.06, SHEET + 0.06, -SHEET / 2 - 0.06, SHEET / 2 + 0.06, [0.16, 0.23, 0.3]);
      const rows = info.heads[l] / 4, hh = rows * 0.21 + 0.12;
      rect(layerX(l) + HEAD_DX, SHEET / 2 - hh, SHEET / 2 + hh, -0.98, 0.98, full ? [0.3, 0.22, 0.42] : [0.1, 0.2, 0.26]);
    }
    seg.push(IN_POS.x, -0.5, 0, OUT_POS.x, -0.5, 0); col.push(0.1, 0.16, 0.2, 0.1, 0.16, 0.2);
    for (let l = 0; l < L; l++) for (const dx of [HEAD_DX, MLP_DX]) { seg.push(layerX(l) + dx, -0.5, 0, layerX(l) + dx, -0.06, 0); col.push(0.07, 0.11, 0.14, 0.07, 0.11, 0.14); }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(seg, 3));
    g.setAttribute("color", new THREE.Float32BufferAttribute(col, 3));
    scene.add(new THREE.LineSegments(g, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, depthTest: false, blending: THREE.AdditiveBlending })));
  }

  // ---- layer labels
  const layerLabels = [];
  for (let l = 0; l < L; l++) {
    const d = document.createElement("div"); d.className = "ll" + (info.layer_types[l] === "full_attention" ? " full" : ""); d.textContent = l;
    d.title = `Layer ${l} (${info.layer_types[l] === "full_attention" ? "attention" : "delta-net"}) — click to enlarge`;
    d.onclick = () => cb.onLayer && cb.onLayer(l);
    labels.append(d); layerLabels.push(d);
  }

  // ---- values
  function setValues(map, scales, gain) {
    for (let l = 0; l < L; l++) {
      const sc = map ? scales[l] / gain : 1;
      for (let n = 0; n < N; n++) {
        const i = l * N + n, x = map ? map[i] / sc : 0, a = Math.min(1, Math.abs(x)), k = i * 3;
        if (a < 0.1) { nCol[k] = IDLE[0]; nCol[k + 1] = IDLE[1]; nCol[k + 2] = IDLE[2]; nSize[i] = 0.03; continue; }
        const c = x > 0 ? POS : NEG, b = 0.2 + 1.8 * a * a;
        nCol[k] = c[0] * b; nCol[k + 1] = c[1] * b; nCol[k + 2] = c[2] * b; nSize[i] = 0.04 + 0.1 * a;
      }
    }
    nGeo.attributes.aColor.needsUpdate = true; nGeo.attributes.aSize.needsUpdate = true;
  }
  function setHeads(hv, gain) {
    let i = 0;
    for (let l = 0; l < L; l++) {
      const vals = hv && hv[l], mx = vals ? Math.max(...vals.map(Math.abs), 1e-9) : 1;
      for (let h = 0; h < info.heads[l]; h++, i++) {
        const x = vals ? vals[h] / mx * Math.min(1.5, gain) : 0, a = Math.min(1, Math.abs(x)), c = x >= 0 ? POS : NEG, b = 0.06 + 1.1 * a * a;
        hCol[i * 3] = vals ? c[0] * b : 0.05; hCol[i * 3 + 1] = vals ? c[1] * b : 0.08; hCol[i * 3 + 2] = vals ? c[2] * b : 0.1;
        hSize[i] = 0.14 + 0.14 * a;
      }
    }
    hGeo.attributes.aColor.needsUpdate = true; hGeo.attributes.aSize.needsUpdate = true;
  }

  // ---- markers (selection, edits) and pathway graph
  const dyn = new THREE.Group(); scene.add(dyn);
  let marks = { sel: null, edits: [] }, graph = null, curves = [], flow = null, nodeLabels = [], pickables = [];
  function pointsFrom(list, material) { // list of {p:Vector3, c:[r,g,b], s:size}
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(list.flatMap(o => o.p.toArray()), 3));
    g.setAttribute("aColor", new THREE.Float32BufferAttribute(list.flatMap(o => o.c), 3));
    g.setAttribute("aSize", new THREE.Float32BufferAttribute(list.map(o => o.s), 1));
    const pts = new THREE.Points(g, material); pts.frustumCulled = false; return pts;
  }
  function rebuild() {
    for (const o of [...dyn.children]) { dyn.remove(o); o.geometry.dispose(); if (o.material !== mat && o.material !== ringMat) o.material.dispose(); }
    for (const l of nodeLabels) l.el.remove();
    nodeLabels = []; curves = []; flow = null; pickables = [];
    const rings = [];
    for (const e of marks.edits) if (e.enabled && e.kind === "group") {
      const s = e.members.length > 60 ? 0.16 : 0.34; // big groups get small rings so they read as a patch
      for (const [l, i] of e.members) rings.push({ p: neuronPos(l, i), c: [0.25, 0.9, 0.55], s });
    } else if (e.enabled && e.kind !== "steer")
      rings.push({ p: e.kind === "neuron" ? neuronPos(e.layer, e.index) : headPos(e.layer, e.index), c: [0.25, 0.9, 0.55], s: 0.5 });
    if (marks.sel) rings.push({ p: marks.sel.kind === "neuron" ? neuronPos(marks.sel.layer, marks.sel.index) : headPos(marks.sel.layer, marks.sel.index), c: [1, 1, 1], s: 0.7 });

    if (graph) {
      const byId = new Map(graph.nodes.map(n => [n.id, n]));
      const nodeDots = [];
      // earlier words that heads were reading: a column on the far left, in the order they were said
      const words = graph.nodes.filter(n => n.kind === "word").sort((a, b) => a.index - b.index);
      const span = Math.max(SHEET + 0.8, words.length * 1.5); // tall enough that the labels do not sit on each other
      words.forEach((n, i) => { n.p = new THREE.Vector3(IN_POS.x - 1.6, SHEET / 2 + span / 2 - (i + 0.5) * span / words.length, 0); });
      for (const n of graph.nodes) {
        if (n.kind !== "word") n.p = nodePos(n);
        const big = n.kind === "out" || n.kind === "embed", word = n.kind === "word";
        const c = n.kind === "out" ? [0.5, 1, 0.75] : n.kind === "embed" ? [0.8, 0.8, 1] : word ? [0.8, 0.55, 1] : n.kind === "head" ? [0.85, 0.6, 1] : n.value >= 0 ? [1, 0.75, 0.4] : [0.45, 0.7, 1];
        nodeDots.push({ p: n.p, c, s: big ? 0.9 : word ? 0.55 : 0.28 + 0.4 * n.weight });
        pickables.push(n);
        if (big || word || n.weight > 0.22) {
          const el = document.createElement("div");
          el.className = "nl " + n.kind;
          const tag = n.kind === "neuron" ? `L${n.layer}·${n.index}` : n.kind === "head" ? `L${n.layer} head ${n.index}` : n.kind === "embed" ? "input word" : word ? `earlier word · ${n.index}` : "writes";
          const txt = n.label === "<|image_pad|>" ? "🖼 picture patch" : n.label ? n.label.replace(/\n/g, "↵") : "";
          const bold = document.createElement("b"); bold.textContent = txt;
          const small = document.createElement("span"); small.textContent = tag;
          if (txt) el.append(bold); el.append(small);
          if (n.kind === "neuron" || n.kind === "head") el.onclick = () => cb.onPick({ kind: n.kind, layer: n.layer, index: n.index });
          if (word) { el.title = "Click to jump to this word"; el.onclick = () => cb.onWord && cb.onWord(n.index); }
          labels.append(el); nodeLabels.push({ el, p: n.p, w: big ? 2 : word ? 1.5 : n.weight });
        }
      }
      dyn.add(pointsFrom(nodeDots, ringMat));
      // edges: faint curve + particles flowing toward the effect
      const linePos = [], lineCol = [], parts = [];
      for (const e of graph.edges) {
        const a = byId.get(e.src), b = byId.get(e.dst); if (!a || !b) continue;
        const dx = (b.p.x - a.p.x) * 0.45, lift = 0.25 + Math.abs(b.p.x - a.p.x) * 0.035;
        const curve = new THREE.CubicBezierCurve3(a.p, new THREE.Vector3(a.p.x + dx, a.p.y + lift, a.p.z), new THREE.Vector3(b.p.x - dx, b.p.y + lift, b.p.z), b.p);
        const c = e.value >= 0 ? POS : NEG, k = 0.1 + 0.5 * e.w, pts = curve.getPoints(36);
        for (let i = 0; i < pts.length - 1; i++) { linePos.push(...pts[i].toArray(), ...pts[i + 1].toArray()); lineCol.push(c[0] * k, c[1] * k, c[2] * k, c[0] * k, c[1] * k, c[2] * k); }
        const count = 2 + Math.round(e.w * 9);
        for (let i = 0; i < count; i++) parts.push({ curve, phase: (i + Math.random() * 0.6) / count, c: c.map(x => x * (0.5 + 0.9 * e.w)), s: 0.07 + 0.11 * e.w });
        curves.push(curve);
      }
      const lg = new THREE.BufferGeometry();
      lg.setAttribute("position", new THREE.Float32BufferAttribute(linePos, 3));
      lg.setAttribute("color", new THREE.Float32BufferAttribute(lineCol, 3));
      dyn.add(new THREE.LineSegments(lg, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, depthTest: false, blending: THREE.AdditiveBlending })));
      if (parts.length) {
        const pts = pointsFrom(parts.map(p => ({ p: p.curve.getPoint(p.phase), c: p.c, s: p.s })), mat);
        dyn.add(pts); flow = { parts, attr: pts.geometry.attributes.position };
      }
    }
    if (rings.length) dyn.add(pointsFrom(rings, ringMat));
    nodeLabels.sort((a, b) => b.w - a.w);
  }

  // ---- picking
  const ray = new THREE.Raycaster(), ndc = new THREE.Vector2(), tmp = new THREE.Vector3();
  function screen(p, rect) { tmp.copy(p).project(camera); return [(tmp.x + 1) / 2 * rect.width, (1 - tmp.y) / 2 * rect.height, tmp.z]; }
  function pick(ev) {
    const rect = canvas.getBoundingClientRect(), mx = ev.clientX - rect.left, my = ev.clientY - rect.top;
    let best = null, bd = 10;
    for (const n of pickables) if (n.kind === "neuron" || n.kind === "head") { const [x, y, z] = screen(n.p, rect); const d = Math.hypot(x - mx, y - my); if (z < 1 && d < bd) { bd = d; best = { kind: n.kind, layer: n.layer, index: n.index }; } }
    if (best) return best;
    bd = 9;
    headIndex.forEach(([l, h], i) => { tmp.fromArray(hPos, i * 3); const [x, y, z] = screen(tmp.clone(), rect); const d = Math.hypot(x - mx, y - my); if (z < 1 && d < bd) { bd = d; best = { kind: "head", layer: l, index: h }; } });
    if (best) return best;
    ndc.set(mx / rect.width * 2 - 1, -(my / rect.height) * 2 + 1); ray.setFromCamera(ndc, camera);
    const o = ray.ray.origin, d = ray.ray.direction; let bt = Infinity;
    for (let l = 0; l < L; l++) {
      const t = (layerX(l) + MLP_DX - o.x) / d.x; if (!(t > 0) || t > bt) continue;
      const y = o.y + d.y * t, z = o.z + d.z * t;
      const row = Math.round((1 - y / SHEET) * (G - 1)), colI = Math.round((z / SHEET + 0.5) * (G - 1));
      if (row < 0 || row >= G || colI < 0 || colI >= G || row * G + colI >= N) continue;
      bt = t; best = { kind: "neuron", layer: l, index: row * G + colI };
    }
    return best;
  }
  let down = null, hoverQueued = false, lastMove = null;
  canvas.addEventListener("pointerdown", e => { down = [e.clientX, e.clientY]; });
  canvas.addEventListener("pointerup", e => { if (down && Math.hypot(e.clientX - down[0], e.clientY - down[1]) < 4) { const p = pick(e); if (p) cb.onPick(p); } down = null; });
  canvas.addEventListener("pointermove", e => { lastMove = e; if (hoverQueued || e.buttons) return; hoverQueued = true; requestAnimationFrame(() => { hoverQueued = false; cb.onHover(pick(lastMove), lastMove); }); });
  canvas.addEventListener("pointerleave", () => cb.onHover(null));

  // ---- render loop
  let active = true, W = 0, H = 0;
  function resize() {
    const w = root.clientWidth, h = root.clientHeight; if (!w || !h || (w === W && h === H)) return;
    W = w; H = h; renderer.setSize(w, h, false); canvas.style.width = w + "px"; canvas.style.height = h + "px";
    camera.aspect = w / h; camera.updateProjectionMatrix();
  }
  new ResizeObserver(resize).observe(root);
  function place(el, p, dy = 0) {
    tmp.copy(p).project(camera);
    if (tmp.z > 1 || Math.abs(tmp.x) > 1.05 || Math.abs(tmp.y) > 1.05) { el.style.display = "none"; return null; }
    const x = (tmp.x + 1) / 2 * W, y = (1 - tmp.y) / 2 * H + dy;
    el.style.display = ""; el.style.transform = `translate(${x.toFixed(1)}px,${y.toFixed(1)}px) translate(-50%,-50%)`; return x;
  }
  function frame() {
    requestAnimationFrame(frame);
    if (!active || !W) return;
    resize(); controls.update();
    const t = performance.now() / 1000;
    const scale = H * renderer.getPixelRatio() / (2 * Math.tan(camera.fov * Math.PI / 360));
    mat.uniforms.uScale.value = ringMat.uniforms.uScale.value = scale; ringMat.uniforms.uTime.value = t;
    if (flow) { flow.parts.forEach((p, i) => { p.curve.getPoint((p.phase + t * 0.22) % 1, tmp); flow.attr.setXYZ(i, tmp.x, tmp.y, tmp.z); }); flow.attr.needsUpdate = true; }
    renderer.render(scene, camera);
    let prev = -1e9;
    for (let l = 0; l < L; l++) { const x = place(layerLabels[l], v.set(layerX(l) + MLP_DX, -0.5, SHEET / 2 + 0.25)); if (x !== null) { if (Math.abs(x - prev) < 20) layerLabels[l].style.display = "none"; else prev = x; } }
    const placed = []; // strongest first; hide labels that would sit on top of one already shown
    for (const n of nodeLabels) {
      const x = place(n.el, n.p, -16); if (x === null) continue;
      tmp.copy(n.p).project(camera); const y = (1 - tmp.y) / 2 * H - 16, w = n.el.offsetWidth / 2 + 4;
      if (placed.some(q => Math.abs(q[0] - x) < q[2] + w && Math.abs(q[1] - y) < 18)) n.el.style.display = "none"; else placed.push([x, y, w]);
    }
    if (rec) compose(rec.c);
  }

  // ---- pictures and video: the WebGL canvas plus the HTML labels, flattened onto one 2D canvas
  let rec = null;
  function compose(target) {
    if (target.width !== canvas.width || target.height !== canvas.height) { target.width = canvas.width; target.height = canvas.height; }
    const ctx = target.getContext("2d"), pr = canvas.width / W, base = canvas.getBoundingClientRect();
    ctx.drawImage(canvas, 0, 0);
    ctx.textBaseline = "middle";
    for (const d of labels.children) {
      if (d.style.display === "none") continue;
      const r = d.getBoundingClientRect(), x = (r.left - base.left) * pr, y = (r.top - base.top) * pr, w = r.width * pr, h = r.height * pr;
      if (d.classList.contains("nl")) { ctx.fillStyle = "rgba(0,0,0,0.66)"; ctx.fillRect(x, y, w, h); ctx.strokeStyle = getComputedStyle(d).borderColor; ctx.lineWidth = pr; ctx.strokeRect(x, y, w, h); }
      for (const part of d.children.length ? d.children : [d]) {
        const cs = getComputedStyle(part), pb = part.getBoundingClientRect();
        ctx.font = `${cs.fontWeight} ${parseFloat(cs.fontSize) * pr}px ${cs.fontFamily}`; ctx.fillStyle = cs.color;
        ctx.fillText(part.textContent, (pb.left - base.left) * pr, y + h / 2);
      }
    }
  }
  function snapshot() {
    renderer.render(scene, camera);
    const c = document.createElement("canvas"); compose(c);
    return new Promise(res => c.toBlob(res, "image/png"));
  }
  function startRecording() {
    const c = document.createElement("canvas"); renderer.render(scene, camera); compose(c);
    const mime = ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm"].find(m => MediaRecorder.isTypeSupported(m));
    const mr = new MediaRecorder(c.captureStream(30), { mimeType: mime, videoBitsPerSecond: 10e6 }), chunks = [];
    mr.ondataavailable = e => { if (e.data.size) chunks.push(e.data); };
    mr.start(250); rec = { c, mr, chunks };
  }
  function stopRecording() {
    return new Promise(res => { const r = rec; rec = null; r.mr.onstop = () => res(new Blob(r.chunks, { type: "video/webm" })); r.mr.stop(); });
  }
  let firstSize = true;
  new ResizeObserver(() => { if (firstSize && root.clientWidth) { firstSize = false; resize(); resetView(); } }).observe(root);
  resize(); setValues(null, null, 1); setHeads(null, 1); frame();

  return {
    setValues, setHeads, resetView, resize,
    setMarks(sel, edits) { marks = { sel, edits }; rebuild(); },
    setGraph(g) { graph = g; rebuild(); },
    setActive(a) { active = a; if (a) resize(); },
    snapshot, startRecording, stopRecording, isRecording: () => !!rec,
  };
}
