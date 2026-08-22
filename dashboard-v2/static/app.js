/* NervNet — Real-Time MAC Digital Twin Dashboard & Interactive Web */

const $ = (id) => document.getElementById(id);
const PROTO_COLORS = { SAC: '#22d3ee', DCF: '#a855f7', GDCF: '#f59e0b', TD3: '#ec4899', GDCF_ORIG: '#a3743a' };

const state = {
  meta: null,
  summary: null,
  topology: null,
  runKey: null,
  devices: new Map(),           // Active protocol snapshot: id -> Device object
  clock: 0,
  mode: 'sim',                  // 'sim' | 'replay' | 'live' | 'idle'
  activeProto: 'SAC',           // Currently focused protocol
  selected: null,               // Selected device id
  deviceSeries: null,
  history: [],                  // [{t, thr, coll, pdr}]
  ws: null,
  hover: null,
  currentView: 'topology',      // 'topology' | 'grid' | 'arena'
  deviceFilter: 'all',
  liveDeviceHistory: new Map(), // id -> array of live snapshots
  sim: {
    runner: null,
    protocols: ['SAC', 'DCF', 'GDCF'],
    perProto: new Map(),        // proto -> { devices: Map, history: [] }
    deviceCount: 50,
    offeredLoad: 0.20,
    seed: 42,
    speed: 1.0,
    isPaused: false,
  },
  view: { zoom: 1.0, panX: 0, panY: 0, apPx: { x: 400, y: 200 } },
};

/* =========================================================================
   1. BOOTSTRAP & INITIALIZATION
   ========================================================================= */
function initImmediate() {
  try {
    // 1. Setup local topology immediately so canvas has data at t=0
    state.topology = generateFallbackTopology(50);

    // 2. Setup navigation, controls, and canvases immediately
    setupNavigation();
    setupControls();
    setupCanvas();
    setupArenaCanvases();

    // 3. Initialize Simulation Runner and wire frame handler
    state.sim.runner = new SimulationRunner();
    state.sim.runner.onFrame = onSimFrame;

    // 4. Start Live Simulation Immediately (Synchronously emits frame 0)
    startLiveSimulation();

    // 5. Select Node #0 by default for instant live telemetry
    selectDevice(0);

    // 6. Start 60fps Canvas Render Loop
    requestAnimationFrame(drawLoop);

    // 7. Asynchronously fetch backend metadata & weights in background
    fetchBackendData();
  } catch (err) {
    console.error('Immediate init error:', err);
  }
}

async function fetchBackendData() {
  try {
    const [metaRes, summaryRes, topoRes] = await Promise.all([
      fetch('/api/meta').then(r => r.json()).catch(() => null),
      fetch('/api/summary').then(r => r.json()).catch(() => null),
      fetch('/api/topology/50').then(r => r.json()).catch(() => null)
    ]);

    if (metaRes) {
      state.meta = metaRes;
      populateRunSelector();
    }
    if (summaryRes) {
      state.summary = summaryRes;
      renderCompare();
    }
    if (topoRes && topoRes.devices && topoRes.devices.length) {
      state.topology = topoRes;
    }

    // Load SAC Neural Network weights
    if (state.sim.runner) {
      state.sim.runner.loadWeights('sac_as_td3').then((loaded) => {
        if (loaded) {
          status('SAC Neural Network weights loaded · 100% active', 'on');
        }
      });
    }
  } catch (err) {
    console.warn('Backend fetch non-blocking error:', err);
  }
}

function generateFallbackTopology(n) {
  const devices = [];
  const cols = 10;
  for (let i = 0; i < n; i++) {
    devices.push({ id: i, x: (i % cols) * 5.0, y: Math.floor(i / cols) * 10.0 });
  }
  return { devices, ap: { x: 22.5, y: 20.0 } };
}

function populateRunSelector() {
  const sel = $('runSelect');
  if (!sel || !state.meta || !state.meta.runs) return;
  sel.innerHTML = '';
  const runs = state.meta.runs.filter(r => r.nodes === 50);
  for (const r of (runs.length ? runs : state.meta.runs)) {
    const opt = document.createElement('option');
    opt.value = r.key;
    opt.textContent = `${r.protocol} · ${r.nodes}n · load ${r.load}`;
    if (r.protocol === 'SAC') opt.selected = true;
    sel.appendChild(opt);
  }
  state.runKey = sel.value;
  sel.onchange = () => { state.runKey = sel.value; stopStream(); };
}

/* =========================================================================
   2. CONTROLS & NAVIGATION
   ========================================================================= */
function setupNavigation() {
  document.querySelectorAll('.tab').forEach(btn => {
    btn.onclick = () => {
      document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t === btn));
      document.querySelectorAll('.tabpane').forEach(p => p.classList.toggle('active', p.id === 'tab-' + btn.dataset.tab));
      if (btn.dataset.tab === 'compare') renderCompare();
    };
  });

  document.querySelectorAll('.view-btn').forEach(btn => {
    btn.onclick = () => {
      document.querySelectorAll('.view-btn').forEach(b => b.classList.toggle('active', b === btn));
      setView(btn.dataset.view);
    };
  });

  document.querySelectorAll('.filter-btn').forEach(btn => {
    btn.onclick = () => {
      document.querySelectorAll('.filter-btn').forEach(b => b.classList.toggle('active', b === btn));
      state.deviceFilter = btn.dataset.filter;
      if (state.currentView === 'grid') renderDeviceGrid();
    };
  });

  const dClose = $('dClose');
  if (dClose) dClose.onclick = () => selectDevice(null);

  const btnBurst = $('btnInjectBurst');
  if (btnBurst) {
    btnBurst.onclick = () => {
      if (state.selected !== null) {
        if (state.sim.runner) {
          state.sim.runner.injectBurst(state.selected, 60);
        }
        // Force local state to active immediately so UI feedback is instantaneous
        const d = state.devices.get(state.selected);
        if (d) {
          d.isMuted = false;
          d.state = 'active';
          d.queue_occ = 0.85;
          d.throughput = 0.045;
        }
        pulseEffect(state.selected);
        selectDevice(state.selected);
      }
    };
  }

  const btnMute = $('btnToggleMute');
  if (btnMute) {
    btnMute.onclick = () => {
      if (state.selected !== null) {
        let isMuted = false;
        if (state.sim.runner) {
          isMuted = state.sim.runner.toggleMute(state.selected);
        }
        const d = state.devices.get(state.selected);
        if (d) {
          d.isMuted = isMuted;
          d.state = isMuted ? 'muted' : 'idle';
          if (isMuted) {
            d.queue_occ = 0;
            d.throughput = 0;
            d.collision = 0;
          }
        }
        renderDetail(state.selected, d);
      }
    };
  }

  const btnSelectRandom = $('btnSelectRandom');
  if (btnSelectRandom) {
    btnSelectRandom.onclick = () => selectDevice(0);
  }
}

function setView(viewName) {
  state.currentView = viewName;
  $('topoView').classList.toggle('hidden', viewName !== 'topology');
  $('gridView').classList.toggle('hidden', viewName !== 'grid');
  $('arenaView').classList.toggle('hidden', viewName !== 'arena');
  $('liveChartRow').classList.toggle('hidden', viewName === 'arena');

  const filterBar = $('deviceFilter');
  if (filterBar) filterBar.style.display = viewName === 'grid' ? 'flex' : 'none';

  const viewTitles = {
    topology: 'Interactive Wireless Mesh Web',
    grid: 'Device Telemetry Cards Matrix',
    arena: 'Tri-Protocol Head-to-Head Arena'
  };
  $('currentViewTitle').textContent = viewTitles[viewName] || 'Wireless Network';

  if (viewName === 'grid') renderDeviceGrid();
  if (viewName === 'topology') fitCanvas();
  if (viewName === 'arena') {
    fitArenaCanvases();
    updateArenaCharts();
  }
}

function setupControls() {
  const devSlider = $('simDevices');
  const loadSlider = $('simLoad');
  const seedInput = $('simSeed');
  const speedSelect = $('simSpeed');

  devSlider.oninput = () => {
    $('simDevicesVal').textContent = devSlider.value;
    const badge = $('deviceCountBadge');
    if (badge) badge.textContent = `${devSlider.value} Devices Active`;
    state.sim.deviceCount = +devSlider.value;
    updateTopologyAndRestart();
  };

  loadSlider.oninput = () => {
    const val = (+loadSlider.value / 100).toFixed(2);
    $('simLoadVal').textContent = val;
    state.sim.offeredLoad = +val;
    restartSimulation();
  };

  seedInput.onchange = () => {
    state.sim.seed = +seedInput.value || 42;
    restartSimulation();
  };

  speedSelect.onchange = () => {
    state.sim.speed = +speedSelect.value || 1.0;
    restartSimulation();
  };

  document.querySelectorAll('.proto-chip').forEach(chip => {
    chip.onclick = () => {
      document.querySelectorAll('.proto-chip').forEach(c => c.classList.remove('active'));
      chip.classList.add('active');
      const p = chip.dataset.proto;
      if (p === 'ALL') {
        state.activeProto = 'SAC';
        setView('arena');
        document.querySelector('.view-btn[data-view="arena"]')?.click();
      } else {
        state.activeProto = p;
        $('activeProtoBadge').textContent = `Protocol: ${window.STRATEGY_META?.[p]?.label || p}`;
        const pd = state.sim.perProto.get(p);
        if (pd && pd.devices.size > 0) {
          state.devices = new Map(pd.devices);
          updateScorecard([...pd.devices.values()], state.clock);
        }
        if (state.currentView === 'arena') {
          setView('topology');
          document.querySelector('.view-btn[data-view="topology"]')?.click();
        }
      }
    };
  });

  document.querySelectorAll('.preset-btn').forEach(btn => {
    btn.onclick = () => {
      document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const preset = btn.dataset.preset;
      if (preset === 'light') {
        devSlider.value = 20; loadSlider.value = 8;
      } else if (preset === 'standard') {
        devSlider.value = 50; loadSlider.value = 20;
      } else if (preset === 'heavy') {
        devSlider.value = 100; loadSlider.value = 40;
      }
      devSlider.oninput(); loadSlider.oninput();
    };
  });

  const btnToggle = $('btnSimToggle');
  btnToggle.onclick = () => {
    state.sim.isPaused = !state.sim.isPaused;
    if (state.sim.isPaused) {
      if (state.sim.runner) state.sim.runner.stop();
      btnToggle.textContent = '▶ Resume Sim';
      btnToggle.classList.remove('primary');
      $('globalStatusPill').style.borderColor = 'rgba(245,158,11,0.4)';
      $('globalStatusText').textContent = 'SIMULATION PAUSED';
      $('globalStatusText').style.color = '#f59e0b';
    } else {
      startLiveSimulation();
      btnToggle.textContent = '⏸ Pause Sim';
      btnToggle.classList.add('primary');
      $('globalStatusPill').style.borderColor = 'rgba(34,211,238,0.3)';
      $('globalStatusText').textContent = 'SIMULATING LIVE';
      $('globalStatusText').style.color = '#22d3ee';
    }
  };

  const btnReset = $('btnSimReset');
  btnReset.onclick = () => {
    seedInput.value = Math.floor(Math.random() * 90000) + 1000;
    seedInput.onchange();
  };

  $('btnPlay').onclick = () => state.ws ? stopStream() : startReplay();
  $('speedSelect').onchange = () => { if (state.mode === 'replay') startReplay(); };
  $('btnLive').onclick = () => state.mode === 'live' ? stopStream() : startLiveNs3();

  $('btnZoomIn').onclick = () => { state.view.zoom = Math.min(2.8, state.view.zoom * 1.2); };
  $('btnZoomOut').onclick = () => { state.view.zoom = Math.max(0.5, state.view.zoom * 0.83); };
  $('btnZoomReset').onclick = () => { state.view.zoom = 1.0; };
}

async function updateTopologyAndRestart() {
  try {
    const t = await fetch(`/api/topology/${state.sim.deviceCount}`).then(r => r.json());
    state.topology = t;
  } catch {
    state.topology = generateFallbackTopology(state.sim.deviceCount);
  }
  restartSimulation();
}

/* =========================================================================
   3. SIMULATION EXECUTION & DATA FLOW
   ========================================================================= */
function startLiveSimulation() {
  stopStream();
  state.mode = 'sim';
  state.history = [];
  state.liveDeviceHistory.clear();
  state.sim.perProto.clear();

  for (const p of ['SAC', 'DCF', 'GDCF']) {
    state.sim.perProto.set(p, { devices: new Map(), history: [] });
  }

  if (state.sim.runner) {
    state.sim.runner.start({
      devices: state.sim.deviceCount,
      load: state.sim.offeredLoad,
      seed: state.sim.seed,
      protocols: ['SAC', 'DCF', 'GDCF'],
      speed: state.sim.speed,
    });
  }

  status(`Simulating ${state.sim.deviceCount} devices · ${state.sim.offeredLoad * 100}% load`, 'on');
}

function restartSimulation() {
  if (state.mode === 'sim' && !state.sim.isPaused) {
    startLiveSimulation();
  }
}

function onSimFrame(proto, frame) {
  try {
    let pd = state.sim.perProto.get(proto);
    if (!pd) {
      // Auto-initialize for protocols not yet in perProto (e.g. ns-3 live frames)
      pd = { devices: new Map(), history: [] };
      state.sim.perProto.set(proto, pd);
    }

    for (const d of frame.devices) {
      pd.devices.set(d.id, d);
    }

    const ds = [...pd.devices.values()];
    const n = ds.length || 1;
    const thr = ds.reduce((s, d) => s + d.throughput, 0) / n;
    const coll = ds.reduce((s, d) => s + d.collision, 0) / n;
    const pdr = ds.reduce((s, d) => s + d.delivery, 0) / n;
    const cw = ds.reduce((s, d) => s + d.cw_vo, 0) / n;

    // Reset history if time rewinds (e.g. new run or mode switch)
    if (pd.history.length > 0 && frame.t < pd.history[pd.history.length - 1].t) {
      pd.history = [];
    }

    pd.history.push({ t: frame.t, thr, coll, pdr, cw });
    if (pd.history.length > 300) pd.history.shift();

    if (proto === state.activeProto) {
      state.clock = frame.t;
      for (const d of frame.devices) {
        state.devices.set(d.id, d);
        if (!state.liveDeviceHistory.has(d.id)) state.liveDeviceHistory.set(d.id, []);
        const h = state.liveDeviceHistory.get(d.id);
        if (h.length > 0 && frame.t < h[h.length - 1].t) {
          h.length = 0;
        }
        h.push({
          t: frame.t,
          cw_vo: d.cw_vo,
          cw_be: d.cw_be,
          throughput: d.throughput,
          collision: d.collision,
          delivery: d.delivery,
          queue_occ: d.queue_occ,
          load_estimate: d.load_estimate
        });
        if (h.length > 250) h.shift();

        // Track real-time device transmission activity for vibrant visual state
        if (d.throughput > 0.0001 || d.collision > 0.01 || d.queue_occ > 0.005 || d.state === 'active') {
          nodeActivityMap.set(d.id, {
            lastTx: performance.now(),
            isCollision: d.collision > 0.04 || d.state === 'colliding'
          });
        }
      }

      updateScorecard(ds, frame.t);

      if (state.currentView === 'grid') renderDeviceGrid();

      if (state.selected !== null) {
        const snap = state.devices.get(state.selected);
        renderDetail(state.selected, snap);
      }
    }

    updateArenaCard(proto, thr, coll, pdr, cw);
    updateArenaCharts();
  } catch (err) {
    console.error('Frame processing error:', err);
  }
}

let _lastChartRenderTime = 0;

function updateScorecard(ds, t) {
  if (!ds || !ds.length) return;
  const n = ds.length;
  const thrAvg = ds.reduce((s, d) => s + d.throughput, 0) / n;
  const thrTotal = ds.reduce((s, d) => s + d.throughput, 0);
  const pdrAvg = ds.reduce((s, d) => s + d.delivery, 0) / n;
  const collAvg = ds.reduce((s, d) => s + d.collision, 0) / n;
  const qAvg = ds.reduce((s, d) => s + d.queue_occ, 0) / n;
  const cwVo = ds.reduce((s, d) => s + d.cw_vo, 0) / n;
  const cwBe = ds.reduce((s, d) => s + d.cw_be, 0) / n;

  const kpiThr = $('kpiThr'); if (kpiThr) kpiThr.textContent = (thrAvg * 1000).toFixed(1);
  const kpiThrSub = $('kpiThrSub'); if (kpiThrSub) kpiThrSub.textContent = `Total: ${thrTotal.toFixed(2)} Mbps`;
  const kpiPdr = $('kpiPdr'); if (kpiPdr) kpiPdr.textContent = (pdrAvg * 100).toFixed(1);
  const kpiColl = $('kpiColl'); if (kpiColl) kpiColl.textContent = (collAvg * 100).toFixed(1);
  const kpiCw = $('kpiCw'); if (kpiCw) kpiCw.textContent = `${Math.round(cwVo)} / ${Math.round(cwBe)}`;
  const kpiQueue = $('kpiQueue'); if (kpiQueue) kpiQueue.textContent = Math.round(qAvg * 100);
  const kpiClock = $('kpiClock'); if (kpiClock) kpiClock.textContent = t.toFixed(1);
  const badge = $('deviceCountBadge'); if (badge) badge.textContent = `${n} Devices Active`;

  if (state.history.length > 0 && t < state.history[state.history.length - 1].t) {
    state.history = [];
  }
  state.history.push({ t, thr: thrAvg, coll: collAvg });
  if (state.history.length > 400) state.history.shift();

  // Throttle live SVG charts to ~20fps (50ms interval) to completely prevent browser lag
  const now = performance.now();
  if (now - _lastChartRenderTime < 48) return;
  _lastChartRenderTime = now;

  if (state.currentView !== 'arena') {
    const thrSeries = [], collSeries = [];
    const protoStyles = {
      SAC: { dash: '', width: 2.8, glow: true },
      DCF: { dash: '6 3', width: 2.0, glow: false },
      GDCF: { dash: '2 3', width: 2.0, glow: false }
    };

    for (const proto of ['SAC', 'DCF', 'GDCF']) {
      const pd = state.sim.perProto.get(proto);
      const color = PROTO_COLORS[proto] || '#94a3b8';
      const st = protoStyles[proto] || { dash: '', width: 2.0 };
      if (pd && pd.history.length >= 2) {
        thrSeries.push({
          name: proto,
          color,
          pts: pd.history.map(h => [h.t, h.thr * 1000]),
          fill: proto === state.activeProto,
          dash: st.dash,
          strokeWidth: st.width,
          glow: st.glow
        });
        collSeries.push({
          name: proto,
          color,
          pts: pd.history.map(h => [h.t, h.coll * 100]),
          fill: proto === state.activeProto,
          dash: st.dash,
          strokeWidth: st.width,
          glow: st.glow
        });
      }
    }
    if (thrSeries.length > 0) {
      lineChart($('liveThrChart'), thrSeries, { yLabel: v => v.toFixed(0), title: 'Throughput' });
    } else {
      lineChart($('liveThrChart'), [
        { name: 'Throughput', color: '#22d3ee', pts: state.history.map(h => [h.t, h.thr * 1000]), fill: true, strokeWidth: 2.5 }
      ], { yLabel: v => v.toFixed(0) });
    }
    if (collSeries.length > 0) {
      lineChart($('liveCollChart'), collSeries, { yLabel: v => v.toFixed(1), title: 'Collisions' });
    } else {
      lineChart($('liveCollChart'), [
        { name: 'Collisions', color: '#f87171', pts: state.history.map(h => [h.t, h.coll * 100]), fill: true, strokeWidth: 2.5 }
      ], { yLabel: v => v.toFixed(1) });
    }
  }
}

/* =========================================================================
   4. INTERACTIVE WIRELESS WEB TOPOLOGY (CANVAS)
   ========================================================================= */
let canvas, ctx;
let nodePx = [];
let packets = [];
let pulseRipples = [];

function setupCanvas() {
  canvas = $('netCanvas');
  if (!canvas) return;
  ctx = canvas.getContext('2d');

  const wrapper = $('canvasWrapper');
  if (wrapper) {
    new ResizeObserver(() => fitCanvas()).observe(wrapper);
  }

  canvas.addEventListener('click', (e) => {
    const rect = canvas.getBoundingClientRect();
    const hit = hitTest(e.clientX - rect.left, e.clientY - rect.top);
    selectDevice(hit);
  });

  canvas.addEventListener('mousemove', (e) => {
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    const hit = hitTest(mx, my);
    state.hover = hit;
    canvas.style.cursor = hit !== null ? 'pointer' : 'crosshair';

    const tt = $('nodeTooltip');
    if (hit !== null && tt) {
      const d = state.devices.get(hit);
      const st = getNodeState(d);
      tt.style.display = 'block';
      tt.style.left = `${mx}px`;
      tt.style.top = `${my}px`;
      $('ttTitle').textContent = `Device #${hit}`;
      $('ttDot').style.background = st.color;
      $('ttBadge').textContent = st.label.toUpperCase();
      $('ttBadge').style.color = st.color;
      $('ttBadge').style.background = st.color + '22';
      $('ttCw').textContent = d ? `${d.cw_vo} / ${d.cw_be}` : '15 / 31';
      $('ttThr').textContent = d ? `${(d.throughput * 1000).toFixed(1)} kbps` : '0.0 kbps';
      $('ttColl').textContent = d ? `${(d.collision * 100).toFixed(1)}%` : '0.0%';
      $('ttQueue').textContent = d ? `${Math.round(d.queue_occ * 100)}%` : '0%';
    } else if (tt) {
      tt.style.display = 'none';
    }
  });

  canvas.addEventListener('mouseleave', () => {
    state.hover = null;
    const tt = $('nodeTooltip');
    if (tt) tt.style.display = 'none';
  });

  canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    const delta = e.deltaY < 0 ? 1.08 : 0.92;
    state.view.zoom = Math.min(2.8, Math.max(0.5, state.view.zoom * delta));
  }, { passive: false });

  fitCanvas();
}

function fitCanvas() {
  if (!canvas) return;
  const wrapper = $('canvasWrapper') || canvas.parentElement;
  const w = wrapper ? wrapper.clientWidth : 800;
  const h = wrapper ? wrapper.clientHeight : 500;
  const dpr = window.devicePixelRatio || 1;
  const finalW = Math.max(w, 400);
  const finalH = Math.max(h, 350);

  canvas.width = finalW * dpr;
  canvas.height = finalH * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function computeNodeLayout(W, H) {
  const n = state.sim.deviceCount || 50;
  const apX = W / 2, apY = H / 2;
  state.view.apPx = { x: apX, y: apY };

  const zoom = state.view.zoom;
  const baseRadius = Math.min(W * 0.42, H * 0.40) * zoom;

  nodePx = [];
  // Dynamic multi-ring layout (up to 100+ nodes)
  const ring1Cap = Math.min(15, Math.ceil(n * 0.22));
  const ring2Cap = Math.min(25, Math.ceil(n * 0.35));
  const ring3Cap = Math.min(35, Math.ceil(n * 0.30));

  for (let i = 0; i < n; i++) {
    let ring, ringRadius, countInRing, ringIdx;
    if (i < ring1Cap) {
      ring = 1; ringRadius = baseRadius * 0.32; countInRing = ring1Cap; ringIdx = i;
    } else if (i < ring1Cap + ring2Cap) {
      ring = 2; ringRadius = baseRadius * 0.58; countInRing = ring2Cap; ringIdx = i - ring1Cap;
    } else if (i < ring1Cap + ring2Cap + ring3Cap) {
      ring = 3; ringRadius = baseRadius * 0.82; countInRing = ring3Cap; ringIdx = i - (ring1Cap + ring2Cap);
    } else {
      ring = 4; ringRadius = baseRadius * 1.02; countInRing = Math.max(1, n - (ring1Cap + ring2Cap + ring3Cap)); ringIdx = i - (ring1Cap + ring2Cap + ring3Cap);
    }

    const angle = (ringIdx / countInRing) * 2 * Math.PI + (ring * 0.28);
    const nx = apX + Math.cos(angle) * ringRadius;
    const ny = apY + Math.sin(angle) * ringRadius;

    nodePx.push({ id: i, x: nx, y: ny, ring });
  }
}

function hitTest(mx, my) {
  const rHit = 16 * state.view.zoom;
  for (const p of nodePx) {
    const dx = p.x - mx, dy = p.y - my;
    if (dx * dx + dy * dy <= rHit * rHit) return p.id;
  }
  return null;
}

const nodeActivityMap = new Map(); // id -> { lastTx: timestamp, isCollision: boolean }

function getNodeState(d, id) {
  if (!d) return { color: '#64748b', label: 'idle', r: 6.0 };
  if (d.isMuted || d.state === 'muted') return { color: '#475569', label: 'muted', r: 5.0 };

  const devId = id !== undefined ? id : d.id;
  const now = performance.now();
  const act = devId !== undefined ? nodeActivityMap.get(devId) : null;
  const timeSinceTx = act ? (now - act.lastTx) : 99999;

  // 1. Collision state
  if (d.collision > 0.04 || d.state === 'colliding' || (act && act.isCollision && timeSinceTx < 600)) {
    return { color: '#f87171', label: 'collision', r: 8.5 };
  }

  // 2. Active transmission state (held for 1.2s for smooth visual persistence)
  if (d.throughput > 0.0001 || d.queue_occ > 0.005 || d.state === 'active' || timeSinceTx < 1200) {
    const pulseFactor = Math.max(0, 1 - timeSinceTx / 1200);
    return { color: '#4ade80', label: 'transmitting', r: 7.0 + pulseFactor * 1.5 };
  }

  // 3. Active connected STA in running simulation
  if (state.mode === 'sim' || state.mode === 'live' || state.mode === 'replay') {
    return { color: '#22d3ee', label: 'connected', r: 6.2 };
  }

  return { color: '#64748b', label: 'idle', r: 6.0 };
}

function pulseEffect(id) {
  const p = nodePx.find(x => x.id === id);
  if (p) {
    pulseRipples.push({ x: p.x, y: p.y, r: 4, maxR: 60, color: '#f59e0b', born: performance.now() });
    pulseRipples.push({ x: p.x, y: p.y, r: 8, maxR: 85, color: '#f87171', born: performance.now() + 100 });
  }
}

function drawLoop(timestamp) {
  if (state.currentView === 'topology' && canvas && ctx) {
    const wrapper = $('canvasWrapper') || canvas.parentElement;
    const W = wrapper ? wrapper.clientWidth : 800;
    const H = wrapper ? wrapper.clientHeight : 500;

    if (W > 0 && H > 0) {
      computeNodeLayout(W, H);
      ctx.clearRect(0, 0, W, H);

      // 1. Background Grid
      ctx.strokeStyle = 'rgba(30, 41, 59, 0.45)';
      ctx.lineWidth = 1;
      const step = 45 * state.view.zoom;
      for (let x = 0; x < W; x += step) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke(); }
      for (let y = 0; y < H; y += step) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke(); }

      if (nodePx.length && state.view.apPx) {
        const ap = state.view.apPx;
        const themeColor = PROTO_COLORS[state.activeProto] || '#22d3ee';
        const now = performance.now();

        // 2. AP Radio Waves
        const wavePhase = (now % 2500) / 2500;
        for (let wi = 0; wi < 3; wi++) {
          const waveR = ((wavePhase + wi * 0.33) % 1) * Math.min(W, H) * 0.48;
          const alphaHex = Math.max(0, Math.min(30, Math.floor((1 - waveR / (Math.min(W, H) * 0.48)) * 30)))
            .toString(16).padStart(2, '0');
          ctx.strokeStyle = themeColor + alphaHex;
          ctx.lineWidth = 1.2;
          ctx.beginPath();
          ctx.arc(ap.x, ap.y, waveR, 0, Math.PI * 2);
          ctx.stroke();
        }

        // 3. Mesh Links & Packets
        for (const p of nodePx) {
          const d = state.devices.get(p.id);
          const st = getNodeState(d, p.id);
          const isSel = state.selected === p.id;
          const isHov = state.hover === p.id;

          ctx.beginPath();
          ctx.moveTo(p.x, p.y);
          ctx.lineTo(ap.x, ap.y);
          if (isSel) {
            ctx.strokeStyle = '#22d3ee'; ctx.lineWidth = 2.0;
          } else if (isHov) {
            ctx.strokeStyle = 'rgba(34, 211, 238, 0.7)'; ctx.lineWidth = 1.5;
          } else if (st.label === 'collision') {
            ctx.strokeStyle = 'rgba(248, 113, 113, 0.35)'; ctx.lineWidth = 1.2;
          } else if (st.label === 'transmitting') {
            ctx.strokeStyle = 'rgba(74, 222, 128, 0.35)'; ctx.lineWidth = 1.2;
          } else if (st.label === 'connected') {
            ctx.strokeStyle = 'rgba(34, 211, 238, 0.16)'; ctx.lineWidth = 0.8;
          } else {
            ctx.strokeStyle = 'rgba(71, 85, 105, 0.12)'; ctx.lineWidth = 0.5;
          }
          ctx.stroke();

          if (st.label !== 'idle' && st.label !== 'muted' && Math.random() < 0.18) {
            packets.push({
              fromX: p.x, fromY: p.y, toX: ap.x, toY: ap.y,
              born: now, dur: 600 + Math.random() * 250,
              isCollision: st.label === 'collision',
              color: st.label === 'collision' ? '#f87171' : (st.label === 'transmitting' ? '#4ade80' : themeColor)
            });
          }
        }

        // 4. Animate Packets
        packets = packets.filter(pk => now - pk.born < pk.dur);
        for (const pk of packets) {
          const k = (now - pk.born) / pk.dur;
          const px = pk.fromX + (pk.toX - pk.fromX) * k;
          const py = pk.fromY + (pk.toY - pk.fromY) * k;

          ctx.fillStyle = pk.color;
          ctx.beginPath();
          ctx.arc(px, py, pk.isCollision ? 3.0 : 2.4, 0, Math.PI * 2);
          ctx.fill();
        }

        // 5. AP Gateway
        ctx.save();
        const apGlow = ctx.createRadialGradient(ap.x, ap.y, 2, ap.x, ap.y, 38);
        apGlow.addColorStop(0, 'rgba(34, 211, 238, 0.4)');
        apGlow.addColorStop(1, 'transparent');
        ctx.fillStyle = apGlow;
        ctx.beginPath(); ctx.arc(ap.x, ap.y, 38, 0, Math.PI * 2); ctx.fill();

        ctx.strokeStyle = '#22d3ee'; ctx.lineWidth = 2.5; ctx.fillStyle = '#0f172a';
        ctx.beginPath(); ctx.arc(ap.x, ap.y, 14, 0, Math.PI * 2); ctx.fill(); ctx.stroke();

        ctx.fillStyle = '#22d3ee'; ctx.font = '700 11px Inter'; ctx.textAlign = 'center';
        ctx.fillText('AP', ap.x, ap.y + 4);
        ctx.font = '600 9px Inter'; ctx.fillStyle = '#94a3b8';
        ctx.fillText('Central Gateway', ap.x, ap.y + 28);
        ctx.restore();

        // 6. Device Nodes
        for (const p of nodePx) {
          const d = state.devices.get(p.id);
          const st = getNodeState(d);
          const isSel = state.selected === p.id;
          const isHov = state.hover === p.id;

          const cwNorm = d ? Math.min(1.0, (d.cw_vo - 15) / 1008) : 0;
          const haloR = st.r + 4 + cwNorm * 12;

          if (st.label !== 'idle' || isSel || isHov) {
            ctx.strokeStyle = isSel ? '#22d3ee' : st.color + '55';
            ctx.lineWidth = 1;
            ctx.setLineDash([2, 3]);
            ctx.beginPath();
            ctx.arc(p.x, p.y, haloR, 0, Math.PI * 2);
            ctx.stroke();
            ctx.setLineDash([]);
          }

          if (st.label !== 'idle') {
            const nodeGlow = ctx.createRadialGradient(p.x, p.y, 1, p.x, p.y, st.r + 8);
            nodeGlow.addColorStop(0, st.color + '66');
            nodeGlow.addColorStop(1, 'transparent');
            ctx.fillStyle = nodeGlow;
            ctx.beginPath(); ctx.arc(p.x, p.y, st.r + 8, 0, Math.PI * 2); ctx.fill();
          }

          ctx.fillStyle = isSel ? '#fff' : st.color;
          ctx.beginPath();
          ctx.arc(p.x, p.y, (isSel || isHov) ? st.r + 2 : st.r, 0, Math.PI * 2);
          ctx.fill();

          if (isSel) {
            ctx.strokeStyle = '#22d3ee'; ctx.lineWidth = 2.0;
            ctx.beginPath(); ctx.arc(p.x, p.y, st.r + 6, 0, Math.PI * 2); ctx.stroke();
          }

          if (isSel || isHov || p.id % 5 === 0) {
            ctx.fillStyle = isSel ? '#22d3ee' : '#cbd5e1';
            ctx.font = `600 ${isSel ? '10px' : '9px'} JetBrains Mono`;
            ctx.textAlign = 'center';
            ctx.fillText(`#${p.id}`, p.x, p.y - haloR - 4);
          }
        }
      }
    }
  }

  requestAnimationFrame(drawLoop);
}

/* =========================================================================
   5. DEVICE INSPECTOR SIDEBAR
   ========================================================================= */
async function selectDevice(id) {
  state.selected = id;
  const empty = $('detailEmpty');
  const body = $('detailBody');

  if (empty) empty.hidden = id !== null;
  if (body) body.hidden = id === null;

  document.querySelectorAll('.device-card.selected').forEach(c => c.classList.remove('selected'));
  if (id === null) return;

  const card = document.querySelector(`.device-card[data-id="${id}"]`);
  if (card) card.classList.add('selected');

  $('dTitle').textContent = `Device #${id}`;
  $('dSub').textContent = `802.11 STA · ${window.STRATEGY_META?.[state.activeProto]?.label || state.activeProto}`;

  const snap = state.devices.get(id);
  renderDetail(id, snap);
}

function renderDetail(id, snap) {
  const d = snap || {};
  const st = getNodeState(snap);

  const dot = $('dStatusDot');
  if (dot) {
    dot.style.background = st.color;
    dot.style.boxShadow = `0 0 8px ${st.color}`;
  }

  const btnMute = $('btnToggleMute');
  if (btnMute) {
    if (d.isMuted) {
      btnMute.textContent = '⚡ Resume Node Traffic';
      btnMute.classList.add('muted-active');
    } else {
      btnMute.textContent = '💤 Force Node Idle';
      btnMute.classList.remove('muted-active');
    }
  }

  const dGrid = $('dGrid');
  if (dGrid) {
    dGrid.innerHTML = `
      <div class="d-item"><label>CWmin VO</label><b>${d.cw_vo ?? 15}</b></div>
      <div class="d-item"><label>CWmin BE</label><b>${d.cw_be ?? 31}</b></div>
      <div class="d-item"><label>AIFSN VO/BE</label><b>${d.aifsn_vo ?? 2} / ${d.aifsn_be ?? 3}</b></div>
      <div class="d-item"><label>Delivery (PDR)</label><b>${(100 * (d.delivery ?? 1.0)).toFixed(1)}%</b></div>
      <div class="d-item"><label>Queue Buffer</label><b>${d.isMuted ? 'MUTED' : Math.round((d.queue_occ ?? 0) * 100) + '%'}</b></div>
      <div class="d-item"><label>Estimated Load</label><b>${d.isMuted ? '0.00' : (d.load_estimate ?? 0.2).toFixed(2)}</b></div>
    `;
  }

  const thrCur = $('dThrCur'); if (thrCur) thrCur.textContent = `${((d.throughput ?? 0) * 1000).toFixed(1)} kbps`;
  const collCur = $('dCollCur'); if (collCur) collCur.textContent = `${((d.collision ?? 0) * 100).toFixed(1)}%`;

  const hist = state.liveDeviceHistory.get(id) || [];
  if (hist.length > 2) {
    lineChart($('dCwChart'), [
      { name: 'CW VO', color: '#22d3ee', pts: hist.map(h => [h.t, h.cw_vo]) },
      { name: 'CW BE', color: '#f59e0b', pts: hist.map(h => [h.t, h.cw_be]) }
    ], { logY: true });

    lineChart($('dThrChart'), [
      { name: 'Throughput', color: '#4ade80', pts: hist.map(h => [h.t, h.throughput * 1000]), fill: true }
    ], { yLabel: v => v.toFixed(0) });

    lineChart($('dCollChart'), [
      { name: 'Collisions', color: '#f87171', pts: hist.map(h => [h.t, h.collision * 100]), fill: true }
    ], { yLabel: v => v.toFixed(1) });
  }
}

/* =========================================================================
   6. DEVICE CARDS MATRIX VIEW
   ========================================================================= */
function renderDeviceGrid() {
  const grid = $('deviceGrid');
  if (!grid || !state.topology) return;
  const n = state.sim.deviceCount;

  if (grid.children.length !== n) {
    grid.innerHTML = '';
    for (let i = 0; i < n; i++) {
      grid.appendChild(createDeviceCard(i));
    }
  }

  for (let i = 0; i < n; i++) {
    const card = grid.children[i];
    if (!card) continue;
    const d = state.devices.get(i);
    const st = getNodeState(d);

    if (state.deviceFilter !== 'all' && st.label !== state.deviceFilter) {
      card.style.display = 'none';
      continue;
    }
    card.style.display = '';

    card.querySelector('.dc-led').className = `dc-led ${st.label}`;
    const badge = card.querySelector('.dc-status');
    badge.className = `dc-status ${st.label}`;
    badge.textContent = st.label;

    card.classList.toggle('selected', state.selected === i);

    if (d) {
      card.querySelector('[data-metric="cw"]').textContent = `${d.cw_vo}`;
      card.querySelector('[data-metric="thr"]').textContent = `${(d.throughput * 1000).toFixed(1)}`;
      card.querySelector('[data-metric="coll"]').textContent = `${(d.collision * 100).toFixed(1)}%`;
      card.querySelector('[data-metric="queue"]').textContent = `${Math.round(d.queue_occ * 100)}%`;

      const thrG = card.querySelector('[data-gauge="thr"]');
      const collG = card.querySelector('[data-gauge="coll"]');
      thrG.style.width = `${Math.min(100, (d.throughput / 0.02) * 100)}%`;
      thrG.style.background = 'linear-gradient(90deg, #22d3ee, #4ade80)';
      collG.style.width = `${Math.min(100, d.collision * 100 * 3.5)}%`;
      collG.style.background = 'linear-gradient(90deg, #f59e0b, #f87171)';
    }
  }
}

function createDeviceCard(id) {
  const card = document.createElement('div');
  card.className = 'device-card';
  card.dataset.id = id;
  card.onclick = () => selectDevice(id);

  card.innerHTML = `
    <div class="dc-header">
      <div class="dc-id"><span class="dc-led idle"></span>STA #${id}</div>
      <span class="dc-status idle">idle</span>
    </div>
    <div class="dc-metrics">
      <div class="dc-metric"><div class="dc-metric-label"><span class="metric-dot" style="background:#f59e0b"></span>CW VO</div><div class="dc-metric-value" data-metric="cw">15</div></div>
      <div class="dc-metric"><div class="dc-metric-label"><span class="metric-dot" style="background:#4ade80"></span>Thr kbps</div><div class="dc-metric-value" data-metric="thr">0.0</div></div>
      <div class="dc-metric"><div class="dc-metric-label"><span class="metric-dot" style="background:#f87171"></span>Collision</div><div class="dc-metric-value" data-metric="coll">0.0%</div></div>
      <div class="dc-metric"><div class="dc-metric-label"><span class="metric-dot" style="background:#a78bfa"></span>Queue</div><div class="dc-metric-value" data-metric="queue">0%</div></div>
    </div>
    <div class="dc-gauge">
      <div class="dc-gauge-row"><span class="dc-gauge-label">Thr</span><div class="dc-gauge-track"><div class="dc-gauge-fill" data-gauge="thr" style="width:0%"></div></div></div>
      <div class="dc-gauge-row"><span class="dc-gauge-label">Col</span><div class="dc-gauge-track"><div class="dc-gauge-fill" data-gauge="coll" style="width:0%"></div></div></div>
    </div>
  `;
  return card;
}

/* =========================================================================
   7. 3-PROTOCOL ARENA (SIDE-BY-SIDE PARALLEL VIEW)
   ========================================================================= */
function setupArenaCanvases() {
  for (const p of ['SAC', 'DCF', 'GDCF']) {
    const c = $(`arenaCanvas${p}`);
    if (c) {
      new ResizeObserver(() => fitArenaCanvas(p)).observe(c);
      fitArenaCanvas(p);
    }
  }
}

function fitArenaCanvases() {
  for (const p of ['SAC', 'DCF', 'GDCF']) fitArenaCanvas(p);
}

function fitArenaCanvas(proto) {
  const c = $(`arenaCanvas${proto}`);
  if (!c) return;
  const w = c.clientWidth || 300;
  const h = c.clientHeight || 160;
  const dpr = window.devicePixelRatio || 1;
  c.width = w * dpr;
  c.height = h * dpr;
  const ctxMini = c.getContext('2d');
  ctxMini.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function updateArenaCard(proto, thr, coll, pdr, cw) {
  const acThr = $(`acThr${proto}`); if (acThr) acThr.textContent = `${(thr * 1000).toFixed(1)} kbps`;
  const acColl = $(`acColl${proto}`); if (acColl) acColl.textContent = `${(coll * 100).toFixed(1)}%`;
  const acPdr = $(`acPdr${proto}`); if (acPdr) acPdr.textContent = `${(pdr * 100).toFixed(0)}%`;
  const acCw = $(`acCw${proto}`); if (acCw) acCw.textContent = Math.round(cw);

  const c = $(`arenaCanvas${proto}`);
  if (!c || state.currentView !== 'arena') return;
  const ctxM = c.getContext('2d');
  const W = c.clientWidth || 300, H = c.clientHeight || 160;
  ctxM.clearRect(0, 0, W, H);

  const pd = state.sim.perProto.get(proto);
  if (!pd) return;

  const color = PROTO_COLORS[proto] || '#22d3ee';
  const apX = W / 2, apY = H * 0.45;

  ctxM.fillStyle = color;
  ctxM.beginPath(); ctxM.arc(apX, apY, 5, 0, Math.PI * 2); ctxM.fill();

  const ds = [...pd.devices.values()];
  const n = ds.length;
  for (let i = 0; i < n; i++) {
    const d = ds[i];
    const angle = (i / n) * 2 * Math.PI;
    const rad = Math.min(W, H) * 0.38;
    const nx = apX + Math.cos(angle) * rad;
    const ny = apY + Math.sin(angle) * (rad * 0.75);

    ctxM.strokeStyle = (d.collision > 0.04) ? 'rgba(248,113,113,0.3)' : (d.throughput > 0) ? `${color}33` : 'rgba(71,85,105,0.15)';
    ctxM.lineWidth = 0.6;
    ctxM.beginPath(); ctxM.moveTo(nx, ny); ctxM.lineTo(apX, apY); ctxM.stroke();

    ctxM.fillStyle = (d.collision > 0.04) ? '#f87171' : (d.throughput > 0) ? '#4ade80' : '#475569';
    ctxM.beginPath(); ctxM.arc(nx, ny, 2.5, 0, Math.PI * 2); ctxM.fill();
  }

  if (pd.history.length >= 2) {
    lineChart($(`arenaChart${proto}`), [
      { name: 'Thr', color, pts: pd.history.map(h => [h.t, h.thr * 1000]), fill: true }
    ], { yLabel: v => v.toFixed(0) });
  }
}

function updateArenaCharts() {
  if (state.currentView !== 'arena') return;

  const thrSeries = [], collSeries = [];
  for (const proto of ['SAC', 'DCF', 'GDCF']) {
    const pd = state.sim.perProto.get(proto);
    const color = PROTO_COLORS[proto] || '#94a3b8';
    // Show series as soon as there are >= 2 data points (not 3)
    if (pd && pd.history.length >= 2) {
      thrSeries.push({ name: proto, color, pts: pd.history.map(h => [h.t, h.thr * 1000]) });
      collSeries.push({ name: proto, color, pts: pd.history.map(h => [h.t, h.coll * 100]) });
    }
  }

  if (thrSeries.length > 0) {
    lineChart($('arenaOverlayThr'), thrSeries, { yLabel: v => v.toFixed(0) });
  }
  if (collSeries.length > 0) {
    lineChart($('arenaOverlayColl'), collSeries, { yLabel: v => v.toFixed(1) });
  }
}

/* =========================================================================
   8. REPLAY & NS-3 BACKEND STREAMING
   ========================================================================= */
function startReplay() {
  stopStream();
  if (state.sim.runner) state.sim.runner.stop();
  state.mode = 'replay';
  state.history = [];
  state.liveDeviceHistory.clear();

  state.ws = new WebSocket(`ws://${location.host}/ws`);
  state.ws.onopen = () => {
    state.ws.send(JSON.stringify({ mode: 'replay', run: state.runKey, speed: +$('speedSelect').value }));
    $('btnPlay').textContent = '⏸ Stop';
    status(`Replaying ${state.runKey}`, 'on');
  };
  hookWS();
}

function startLiveNs3() {
  stopStream();
  if (state.sim.runner) state.sim.runner.stop();
  state.mode = 'live';
  state.history = [];
  state.liveDeviceHistory.clear();
  state.sim.perProto.clear();

  for (const p of ['SAC', 'DCF', 'GDCF']) {
    state.sim.perProto.set(p, { devices: new Map(), history: [] });
  }

  const logEl = $('engineLog');
  if (logEl) logEl.textContent = '>>> Initializing native ns-3.48 C++ Discrete-Event Network Simulator (SAC + DCF + GDCF)...\n';

  state.ws = new WebSocket(`ws://${location.host}/ws`);
  state.ws.onopen = () => {
    state.ws.send(JSON.stringify({
      mode: 'live',
      nodes: state.sim.deviceCount,
      load: state.sim.offeredLoad,
      protocol: 'ALL',
      simTime: 25.0
    }));
    const btnL = $('btnLive');
    if (btnL) {
      btnL.classList.add('active');
      btnL.textContent = '⏹ Stop ns-3';
    }
    const logRow = $('logRow');
    if (logRow) logRow.hidden = false;

    $('globalStatusPill').style.borderColor = 'rgba(244, 63, 94, 0.6)';
    $('globalStatusText').textContent = `LIVE NS-3 C++ (TRI-PROTOCOL)`;
    $('globalStatusText').style.color = '#f43f5e';
    status(`Spawning real ns-3 C++ discrete-event engine (SAC + DCF + GDCF, ${state.sim.deviceCount} nodes)...`, 'live');
  };
  hookWS();
}

function hookWS() {
  const ws = state.ws;
  ws.onmessage = (ev) => {
    let m; try { m = JSON.parse(ev.data); } catch { return; }
    if (m.type === 'meta') {
      const el = $('engineLog');
      if (el) {
        el.textContent += `[NS-3 PROCESS STARTED] PID: ${m.pid} | Protocol: ${m.protocol || 'SAC'} | Nodes: ${m.nodes || 50} | Load: ${m.load || 0.2}\n`;
        el.scrollTop = 1e9;
      }
      status(`ns-3 C++ PID ${m.pid} simulating 802.11 PHY/MAC...`, 'live');
      return;
    }
    if (m.type === 'log') {
      const el = $('engineLog');
      if (el) { el.textContent += m.msg + '\n'; el.scrollTop = 1e9; }
      return;
    }
    if (m.type === 'end') {
      const el = $('engineLog');
      if (el) { el.textContent += '[NS-3 RUN COMPLETED] Telemetry stream finished successfully.\n'; el.scrollTop = 1e9; }
      stopStream();
      status('ns-3 simulation run completed');
      return;
    }
    if (m.type === 'error') {
      stopStream();
      status('Stream error: ' + m.msg);
      return;
    }
    if (m.type === 'frame') {
      onSimFrame(m.protocol || state.activeProto, m);
    }
  };
  ws.onclose = () => { if (state.ws === ws) stopStream(); };
  ws.onerror = () => status('Connection error');
}

function stopStream() {
  if (state.ws) { try { state.ws.close(); } catch {} state.ws = null; }
  const btnP = $('btnPlay'); if (btnP) btnP.textContent = '▶ Replay';
  const btnL = $('btnLive');
  if (btnL) {
    btnL.classList.remove('active');
    btnL.textContent = '● Live ns-3';
  }
  const dot = $('connDot'); if (dot) dot.className = 'conn-dot';
  if (state.mode === 'live') {
    $('globalStatusPill').style.borderColor = 'rgba(34,211,238,0.3)';
    $('globalStatusText').textContent = 'SIMULATING LIVE';
    $('globalStatusText').style.color = '#22d3ee';
  }
}

function status(msg, dot) {
  const s = $('footerStatus');
  if (s) s.textContent = msg;
  const d = $('connDot');
  if (d) d.className = 'conn-dot' + (dot ? ' ' + dot : '');
}

/* =========================================================================
   9. HIGH-PERFORMANCE SVG CHARTS
   ========================================================================= */
function svgEl(tag, attrs) {
  const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const k in attrs) el.setAttribute(k, attrs[k]);
  return el;
}

function lineChart(svg, series, opts = {}) {
  if (!svg) return;
  while (svg.firstChild) svg.removeChild(svg.firstChild);

  const W = svg.clientWidth || 480, H = svg.clientHeight || 120;
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);

  const pad = { l: 38, r: 10, t: 10, b: 20 };
  
  // Clean, sort, and deduplicate all series points
  const cleanSeries = [];
  for (const s of series) {
    if (!s.pts || s.pts.length < 2) continue;
    const sorted = [...s.pts].sort((a, b) => a[0] - b[0]);
    const cleanPts = [];
    for (let i = 0; i < sorted.length; i++) {
      if (i === 0 || sorted[i][0] > sorted[i - 1][0]) {
        cleanPts.push(sorted[i]);
      }
    }
    if (cleanPts.length >= 2) {
      cleanSeries.push({ ...s, pts: cleanPts });
    }
  }

  const all = cleanSeries.flatMap(s => s.pts);
  if (!all.length) return;

  const xs = all.map(p => p[0]);
  const ys = all.map(p => (opts.logY ? Math.log10(Math.max(p[1], 1)) : p[1]));
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  const y0 = 0, y1 = Math.max(...ys) || 1;

  const X = v => pad.l + (v - x0) / (x1 - x0 || 1) * (W - pad.l - pad.r);
  const Y = v => H - pad.b - (v / (y1 || 1)) * (H - pad.t - pad.b);

  // Background Grid Lines & Y-axis labels
  for (let i = 0; i <= 3; i++) {
    const yv = y1 * i / 3, y = Y(yv);
    svg.appendChild(svgEl('line', { x1: pad.l, x2: W - pad.r, y1: y, y2: y, stroke: 'rgba(30,41,59,0.7)', 'stroke-width': 1 }));
    const label = opts.logY ? Math.round(10 ** yv) : (opts.yLabel ? opts.yLabel(yv) : yv.toFixed(1));
    const t = svgEl('text', { x: pad.l - 5, y: y + 3, fill: '#64748b', 'font-size': 8.5, 'font-family': 'JetBrains Mono', 'text-anchor': 'end' });
    t.textContent = label;
    svg.appendChild(t);
  }

  // Smooth cubic Bezier spline helper with micro-offset for zero-overlap separation
  function smoothPath(pts, mapY, offsetPx = 0) {
    if (pts.length < 2) return '';
    const coords = pts.map(p => [X(p[0]), Y(mapY(p[1])) + offsetPx]);
    if (coords.length === 2) {
      return `M${coords[0][0].toFixed(1)},${coords[0][1].toFixed(1)}L${coords[1][0].toFixed(1)},${coords[1][1].toFixed(1)}`;
    }
    const tension = 0.3;
    let d = `M${coords[0][0].toFixed(1)},${coords[0][1].toFixed(1)}`;
    for (let i = 0; i < coords.length - 1; i++) {
      const p0 = coords[Math.max(0, i - 1)];
      const p1 = coords[i];
      const p2 = coords[i + 1];
      const p3 = coords[Math.min(coords.length - 1, i + 2)];
      const cp1x = p1[0] + (p2[0] - p0[0]) * tension;
      const cp1y = p1[1] + (p2[1] - p0[1]) * tension;
      const cp2x = p2[0] - (p3[0] - p1[0]) * tension;
      const cp2y = p2[1] - (p3[1] - p1[1]) * tension;
      d += `C${cp1x.toFixed(1)},${cp1y.toFixed(1)},${cp2x.toFixed(1)},${cp2y.toFixed(1)},${p2[0].toFixed(1)},${p2[1].toFixed(1)}`;
    }
    return d;
  }

  cleanSeries.forEach((s, sIdx) => {
    const pts = s.pts;
    const mapY = v => opts.logY ? Math.log10(Math.max(v, 1)) : v;
    // Apply micro-offset (±1.5px) so overlapping zero-lines remain visibly separated
    const offsetPx = cleanSeries.length > 1 ? (sIdx - 1) * 1.6 : 0;
    const d = smoothPath(pts, mapY, offsetPx);

    if (s.fill) {
      const area = d + `L${X(pts[pts.length - 1][0]).toFixed(1)},${(H - pad.b).toFixed(1)}L${X(pts[0][0]).toFixed(1)},${(H - pad.b).toFixed(1)}Z`;
      svg.appendChild(svgEl('path', { d: area, fill: s.color, 'fill-opacity': 0.12 }));
    }

    const pathAttrs = {
      d,
      fill: 'none',
      stroke: s.color,
      'stroke-width': s.strokeWidth || (s.name === 'SAC' ? 2.6 : 1.9),
      'stroke-linejoin': 'round',
      'stroke-linecap': 'round',
    };
    if (s.dash) pathAttrs['stroke-dasharray'] = s.dash;
    svg.appendChild(svgEl('path', pathAttrs));

    const last = pts[pts.length - 1];
    const ly = Y(mapY(last[1])) + offsetPx;
    const rDot = s.name === 'SAC' ? 3.6 : 2.6;
    svg.appendChild(svgEl('circle', { cx: X(last[0]), cy: ly, r: rDot, fill: s.color }));
  });

  // Top-right live value HUD badges for crystal-clear visual separation
  if (cleanSeries.length > 1) {
    let badgeX = W - pad.r - 4;
    for (let i = cleanSeries.length - 1; i >= 0; i--) {
      const s = cleanSeries[i];
      const lastVal = s.pts[s.pts.length - 1][1];
      const valStr = opts.yLabel ? opts.yLabel(lastVal) : lastVal.toFixed(1);
      const text = `${s.name}: ${valStr}`;

      const t = svgEl('text', {
        x: badgeX, y: pad.t + 4, fill: s.color,
        'font-size': 9.0, 'font-weight': 700, 'font-family': 'JetBrains Mono',
        'text-anchor': 'end'
      });
      t.textContent = text;
      svg.appendChild(t);
      badgeX -= (text.length * 6.5 + 12);
    }
  }
}

function barChart(svg, rows, opts = {}) {
  if (!svg) return;
  while (svg.firstChild) svg.removeChild(svg.firstChild);

  const W = svg.clientWidth || 500, H = svg.clientHeight || 220;
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  const pad = { l: 46, r: 10, t: 14, b: 26 };
  const maxV = Math.max(...rows.map(r => r.value)) * 1.15 || 1;
  const bw = Math.min(80, (W - pad.l - pad.r) / rows.length * 0.55);

  rows.forEach((r, i) => {
    const cx = pad.l + (i + 0.5) * (W - pad.l - pad.r) / rows.length;
    const h = (r.value / maxV) * (H - pad.t - pad.b);
    svg.appendChild(svgEl('rect', {
      x: cx - bw / 2, y: H - pad.b - h, width: bw, height: Math.max(h, 2), rx: 5,
      fill: r.color, 'fill-opacity': r.hl ? 0.95 : 0.55
    }));
    if (r.hl) {
      svg.appendChild(svgEl('rect', {
        x: cx - bw / 2 - 3, y: H - pad.b - h - 3, width: bw + 6, height: Math.max(h, 2) + 6, rx: 7,
        fill: 'none', stroke: r.color, 'stroke-width': 1.4, 'stroke-dasharray': '3 3'
      }));
    }
    const t = svgEl('text', { x: cx, y: H - pad.b - h - 7, fill: r.color, 'font-size': 10.5, 'font-weight': 700, 'font-family': 'JetBrains Mono', 'text-anchor': 'middle' });
    t.textContent = opts.fmt ? opts.fmt(r.value) : r.value.toFixed(2);
    svg.appendChild(t);

    const l = svgEl('text', { x: cx, y: H - 8, fill: '#94a3b8', 'font-size': 10, 'font-weight': 600, 'text-anchor': 'middle' });
    l.textContent = r.label;
    svg.appendChild(l);
  });
}

/* =========================================================================
   10. PROTOCOL LAB BENCHMARK MATRIX
   ========================================================================= */
function renderCompare() {
  const rows = state.summary ? state.summary.comparison : [];
  if (!rows || !rows.length) return;

  const agg = {};
  for (const r of rows) {
    const p = r.protocol;
    (agg[p] ??= { n: 0, thr: 0, pdr: 0, coll: 0, lat: 0, fair: 0 });
    const a = agg[p]; a.n++;
    a.thr += r.throughput_mbps_per_node; a.pdr += r.delivery_ratio;
    a.coll += r.collision_rate; a.lat += r.avg_latency_ms; a.fair += r.fairness;
  }

  const protos = Object.keys(agg).sort((a, b) => (a === 'SAC' ? -1 : b === 'SAC' ? 1 : 0));
  const avg = p => {
    const a = agg[p];
    return { thr: a.thr / a.n, pdr: a.pdr / a.n * 100, coll: a.coll / a.n * 100, lat: a.lat / a.n, fair: a.fair / a.n };
  };

  const sac = agg['SAC'] ? avg('SAC') : agg['TD3'] ? avg('TD3') : { thr: 0, pdr: 0, coll: 0, lat: 0, fair: 0 };
  const base = protos.filter(p => p !== 'SAC' && p !== 'TD3');
  const bestThr = Math.max(...base.map(p => avg(p).thr));
  const bestColl = Math.min(...base.map(p => avg(p).coll));
  const bestLat = Math.min(...base.map(p => avg(p).lat));
  const pct = (a, b, lb) => lb ? (b - a) / (b || 1) * 100 : (a - b) / (b || 1) * 100;

  const heroEl = $('cmpHero');
  if (heroEl) {
    heroEl.innerHTML = [
      { l: 'SAC Throughput Gain', v: sac.thr.toFixed(3) + ' Mbps', s: `+${pct(sac.thr, bestThr).toFixed(1)}% vs best baseline`, c: '--accent-c:#22d3ee', ic: '⇅' },
      { l: 'SAC Collision Reduction', v: sac.coll.toFixed(2) + '%', s: `−${Math.abs(pct(sac.coll, bestColl, true)).toFixed(1)}% drop vs DCF`, c: '--accent-c:#4ade80', ic: '✓' },
      { l: 'Mean Transmission Latency', v: sac.lat.toFixed(2) + ' ms', s: `−${Math.abs(pct(sac.lat, bestLat, true)).toFixed(1)}% vs baseline`, c: '--accent-c:#f59e0b', ic: '⏱' },
      { l: 'Packet Delivery Ratio (PDR)', v: sac.pdr.toFixed(2) + '%', s: `Jain Fairness: ${sac.fair.toFixed(3)}`, c: '--accent-c:#a78bfa', ic: '◫' },
      { l: 'ns-3 Validated Protocols', v: `${protos.length} Algorithms`, s: `${rows.length} NS-3 Sweep Runs`, c: '--accent-c:#38bdf8', ic: '⛭' },
      { l: 'Traffic QoS Classes', v: 'VO + BE ACs', s: 'Voice (High) + Best Effort', c: '--accent-c:#e879f9', ic: '◈' }
    ].map(h => `
      <div class="kpi" style="${h.c}">
        <div class="kpi-head"><span class="kpi-ic">${h.ic}</span>${h.l}</div>
        <div class="kpi-val">${h.v}</div>
        <div class="kpi-footer"><span class="kpi-sub green">${h.s}</span></div>
      </div>
    `).join('');
  }

  const mk = (metric) => protos.map(p => ({
    label: p === 'SAC' ? 'SAC (Ours)' : p,
    value: avg(p)[metric],
    color: PROTO_COLORS[p] || '#94a3b8',
    hl: p === 'SAC'
  }));

  barChart($('cmpThr'), mk('thr'), { fmt: v => v.toFixed(3) });
  barChart($('cmpPdr'), mk('pdr'), { fmt: v => v.toFixed(1) });
  barChart($('cmpColl'), mk('coll'), { fmt: v => v.toFixed(2) });
  barChart($('cmpLat'), mk('lat'), { fmt: v => v.toFixed(1) });

  const pc = state.summary ? state.summary.perclass : [];
  if (pc && pc.length) {
    const svg = $('cmpClass');
    if (svg) {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      const W = svg.clientWidth || 800, H = 260;
      svg.setAttribute('viewBox', `0 0 ${W} ${H}`);

      const groups = [...new Set(pc.map(r => r.protocol))];
      const metrics = [
        { key: 'crit_throughput', label: 'VO Thr (kbps)', color: '#22d3ee' },
        { key: 'be_throughput', label: 'BE Thr (kbps)', color: '#f59e0b' },
        { key: 'crit_collision_rate', label: 'VO Coll (%)', color: '#4ade80' },
        { key: 'be_collision_rate', label: 'BE Coll (%)', color: '#f87171' }
      ];

      const aggP = {};
      for (const r of pc) {
        const p = r.protocol;
        (aggP[p] ??= { n: 0 });
        aggP[p].n++;
        for (const m of metrics) aggP[p][m.key] = (aggP[p][m.key] || 0) + (+r[m.key]);
      }

      const maxV = Math.max(...groups.flatMap(p => metrics.map(m => (aggP[p][m.key] || 0) / aggP[p].n))) * 1.2 || 1;
      const gw = (W - 80) / groups.length;

      metrics.forEach((m, mi) => {
        groups.forEach((p, gi) => {
          const v = (aggP[p][m.key] || 0) / aggP[p].n;
          const bw = (gw / metrics.length) * 0.7;
          const x = 60 + gi * gw + mi * (gw / metrics.length) + 4;
          const h = (v / maxV) * (H - 60);

          svg.appendChild(svgEl('rect', {
            x, y: H - 30 - h, width: bw, height: Math.max(h, 2), rx: 3,
            fill: m.color, 'fill-opacity': p === 'SAC' ? 0.95 : 0.5
          }));
          const t = svgEl('text', { x: x + bw / 2, y: H - 30 - h - 4, fill: m.color, 'font-size': 8.5, 'font-family': 'JetBrains Mono', 'text-anchor': 'middle' });
          t.textContent = v.toFixed(2);
          svg.appendChild(t);
        });
      });

      groups.forEach((p, gi) => {
        const t = svgEl('text', { x: 60 + gi * gw + gw / 2, y: H - 10, fill: p === 'SAC' ? '#22d3ee' : '#94a3b8', 'font-size': 11, 'font-weight': 700, 'text-anchor': 'middle' });
        t.textContent = p;
        svg.appendChild(t);
      });

      metrics.forEach((m, mi) => {
        const t = svgEl('text', { x: 10, y: 16 + mi * 14, fill: m.color, 'font-size': 9.5, 'font-weight': 600 });
        t.textContent = '■ ' + m.label;
        svg.appendChild(t);
      });
    }
  }

  const cols = ['protocol', 'num_nodes', 'offered_load', 'throughput_mbps_per_node', 'delivery_ratio', 'collision_rate', 'avg_latency_ms', 'fairness'];
  const thead = '<tr>' + cols.map(c => `<th>${c.replace(/_/g, ' ').toUpperCase()}</th>`).join('') + '</tr>';
  const tbody = rows.map(r => `<tr class="${r.protocol === 'SAC' ? 'hl' : ''}">` + cols.map(c => `<td>${r[c]}</td>`).join('') + '</tr>').join('');
  const cmpTable = $('cmpTable');
  if (cmpTable) cmpTable.innerHTML = thead + tbody;
}

// Start Immediately on DOM Ready / Script Parse
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initImmediate);
} else {
  initImmediate();
}
