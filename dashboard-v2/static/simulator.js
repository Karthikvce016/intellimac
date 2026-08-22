/**
 * NervNet MAC Simulator — High-Fidelity Slotted Wireless Channel Engine
 * Implements DCF (802.11 BEB), GDCF (Adaptive Backoff), and SAC (Deep RL Continuous Controller)
 */

// ========================= DETERMINISTIC PRNG =========================
function createRNG(seed) {
  let s = (seed | 0) || 42;
  return function () {
    s |= 0; s = s + 0x6D2B79F5 | 0;
    let t = Math.imul(s ^ s >>> 15, 1 | s);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

function gaussRand(rng) {
  const u1 = Math.max(rng(), 1e-9), u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

function poissonRand(rng, lambda) {
  if (lambda <= 0) return 0;
  if (lambda > 20) return Math.max(0, Math.round(lambda + Math.sqrt(lambda) * gaussRand(rng)));
  let L = Math.exp(-lambda), k = 0, p = 1;
  do { k++; p *= rng(); } while (p > L && k < 100);
  return k - 1;
}

// ========================= CW SNAP HELPER =========================
// Mirrors C++ SnapToValidCw exactly: snaps up to nearest 2^n-1 value
function snapToValidCw(cw) {
  let v = 1;
  while (v < cw) v = (v << 1) | 1;
  return Math.min(v, 1023);
}

// ========================= NEURAL NETWORK INFERENCE =========================
class NeuralNet {
  constructor(w) {
    this.w0 = w['mu.0.weight']; this.b0 = w['mu.0.bias'];
    this.w2 = w['mu.2.weight']; this.b2 = w['mu.2.bias'];
    this.w4 = w['mu.4.weight']; this.b4 = w['mu.4.bias'];
  }
  forward(x) {
    // Layer 1: Linear + ReLU (400 units)
    const h1 = new Float64Array(this.w0.length);
    for (let i = 0; i < this.w0.length; i++) {
      let s = this.b0[i]; const row = this.w0[i];
      for (let j = 0; j < x.length; j++) s += row[j] * x[j];
      h1[i] = s > 0 ? s : 0;
    }
    // Layer 2: Linear + ReLU (300 units)
    const h2 = new Float64Array(this.w2.length);
    for (let i = 0; i < this.w2.length; i++) {
      let s = this.b2[i]; const row = this.w2[i];
      for (let j = 0; j < h1.length; j++) s += row[j] * h1[j];
      h2[i] = s > 0 ? s : 0;
    }
    // Layer 3: Linear + Tanh (4 units)
    const out = new Float64Array(this.w4.length);
    for (let i = 0; i < this.w4.length; i++) {
      let s = this.b4[i]; const row = this.w4[i];
      for (let j = 0; j < h2.length; j++) s += row[j] * h2[j];
      out[i] = Math.tanh(s);
    }
    return out;
  }
}

// ========================= DEVICE AGENT =========================
const CW_MIN = 15, CW_MAX = 1023, MAX_QUEUE = 60;

class Device {
  constructor(id, loadMult) {
    this.id = id;
    this.loadMult = loadMult || 1.0;
    this.baseLoadMult = this.loadMult;
    this.isMuted = false;
    this.cwMinVo = 15; this.cwMaxVo = 31;  // track max for mult obs
    this.cwMinBe = 31; this.cwMaxBe = 127; // track max for mult obs
    this.aifsnVo = 2; this.aifsnBe = 3;
    this.activeCw = 15;
    this.backoff = 0;
    this.queue = 0;
    this._att = 0; this._suc = 0; this._col = 0; this._gen = 0;
    this.throughput = 0;       // Mbps
    this.collisionRate = 0;    // fraction 0..1
    this.deliveryRatio = 1.0;  // fraction 0..1
    this.queueOcc = 0;         // fraction 0..1
    this.loadEstimate = 0;
    this.state = 'idle';       // 'idle' | 'active' | 'colliding' | 'muted'
  }
}

// ========================= PROTOCOL STRATEGIES =========================
const Strategies = {
  DCF: {
    label: 'DCF (802.11)', color: '#a855f7',
    init(d) { d.cwMinVo = 15; d.cwMinBe = 31; d.activeCw = 15; },
    update(d) {
      if (d._col > 0) d.activeCw = Math.min(d.activeCw * 2 + 1, CW_MAX);
      if (d._suc > 0) d.activeCw = 15;
      d.cwMinVo = 15; d.cwMinBe = 31;
    }
  },
  GDCF: {
    label: 'GDCF (Adaptive)', color: '#f59e0b',
    init(d) { d.cwMinVo = 32; d.cwMinBe = 64; d.activeCw = 32; },
    update(d) {
      const c = d.collisionRate;
      if (c > 0.12) d.cwMinVo = Math.min(Math.ceil(d.cwMinVo * 1.35), CW_MAX);
      else if (c < 0.08) d.cwMinVo = Math.max(Math.floor(d.cwMinVo * 0.88), CW_MIN);
      d.cwMinBe = Math.min(d.cwMinVo * 2, CW_MAX);
      d.activeCw = d.cwMinVo;
    }
  },
  SAC: {
    label: 'SAC (Ours)', color: '#22d3ee', net: null,
    init(d) { d.cwMinVo = 63; d.cwMinBe = 127; d.activeCw = 63; },
    update(d) {
      if (!this.net) {
        const target = 32 + d.collisionRate * 850 + d.loadEstimate * 300 + d.queueOcc * 200;
        d.cwMinVo = Math.round(Math.max(CW_MIN, Math.min(CW_MAX, target)));
        d.cwMinBe = Math.min(d.cwMinVo * 2, CW_MAX);
        d.activeCw = d.cwMinVo;
        return;
      }
      // Observation exactly matches training (train_sac.py) and C++ (GetObservationWithVO)
      const multVo = (d.cwMaxVo || d.cwMinVo * 2) / Math.max(1, d.cwMinVo);
      const multBe = (d.cwMaxBe || d.cwMinBe * 2) / Math.max(1, d.cwMinBe);
      const obs = [
        Math.min(d.loadEstimate, 1.0),                          // slot 0: load
        Math.min(d.collisionRate, 1.0),                         // slot 1: collision_rate
        Math.min(d.deliveryRatio, 1.0),                         // slot 2: delivery_ratio
        Math.min(d.collisionRate * 10.0, 1.0),                  // slot 3: latency_norm
        Math.min(d.queueOcc, 1.0),                              // slot 4: queue_occ
        (d.cwMinVo - 3.0) / (1023.0 - 3.0),                   // slot 5: cw_min_vo_norm
        (d.cwMinBe - 16.0) / (1024.0 - 16.0),                 // slot 6: cw_min_be_norm
        Math.min(1.0, Math.max(0.0, (multVo - 1.0) / 3.0)),    // slot 7: mult_vo_norm
        Math.min(1.0, Math.max(0.0, (multBe - 1.0) / 3.0)),    // slot 8: mult_be_norm
      ];
      const a = this.net.forward(obs);
      // Decode action[0] → VO CWmin using same grid as training + C++
      const critGrid = [3, 7, 15, 31, 63, 127, 255, 511, 1023];
      const beGrid   = [15, 31, 63, 127, 255, 511, 1023];
      const critIdx = Math.max(0, Math.min(8, Math.floor((a[0] + 1.0) * 0.5 * 8.0 + 0.5)));
      d.cwMinVo = critGrid[critIdx];
      const voMult = 1.0 + (a[1] + 1.0) * 0.5 * 3.0;
      d.cwMaxVo = snapToValidCw(Math.round(d.cwMinVo * voMult));
      d.cwMaxVo = Math.max(d.cwMaxVo, d.cwMinVo);
      // Decode action[2] → BE CWmin
      const beIdx = Math.max(0, Math.min(6, Math.floor((a[2] + 1.0) * 0.5 * 6.0 + 0.5)));
      d.cwMinBe = beGrid[beIdx];
      const beMult = 1.0 + (a[3] + 1.0) * 0.5 * 3.0;
      d.cwMaxBe = snapToValidCw(Math.round(d.cwMinBe * beMult));
      d.cwMaxBe = Math.max(d.cwMaxBe, d.cwMinBe);
      d.activeCw = d.cwMinVo;
    }
  },
};

// ========================= MAC SLOTTED SIMULATOR =========================
const DT = 0.1;           // 100 ms per step
const SLOTS = 1100;       // ~100ms / 91μs slot
const PKT_BYTES = 1500;   // 1500 byte payloads

class MACSimulator {
  constructor(nDev, load, seed, proto, net) {
    this.n = nDev;
    this.load = Math.max(0.005, Math.min(load, 1.0));
    this.proto = proto;
    this.rng = createRNG(seed);
    this.strat = Object.create(Strategies[proto] || Strategies.SAC);
    if (proto === 'SAC' && net) this.strat.net = net;
    this.time = 0;
    this.tickCount = 0;
    this.devices = [];
    for (let i = 0; i < nDev; i++) {
      const d = new Device(i, 0.85 + this.rng() * 0.3);
      this.strat.init(d);
      this.devices.push(d);
    }
  }

  injectBurst(nodeId, packetCount) {
    const d = this.devices.find(x => x.id === nodeId);
    if (d) {
      d.isMuted = false;
      d.queue = Math.min(d.queue + (packetCount || 50), MAX_QUEUE);
      d.loadMult = 4.5;
      d.state = 'active';
    }
  }

  toggleMute(nodeId) {
    const d = this.devices.find(x => x.id === nodeId);
    if (d) {
      d.isMuted = !d.isMuted;
      if (d.isMuted) {
        d.queue = 0;
        d.loadMult = 0;
        d.state = 'muted';
        d.throughput = 0;
        d.collisionRate = 0;
      } else {
        d.loadMult = d.baseLoadMult || 1.0;
        d.state = 'idle';
      }
      return d.isMuted;
    }
    return false;
  }

  tick() {
    this.time += DT;
    this.tickCount++;
    const rng = this.rng;
    const baseArrivals = this.load * 14.0;

    // 1. Packet Arrivals
    for (const d of this.devices) {
      if (d.isMuted) {
        d._gen = 0;
        d.queue = 0;
        d.state = 'muted';
        continue;
      }
      const g = poissonRand(rng, baseArrivals * d.loadMult);
      d._gen = g;
      d.queue = Math.min(d.queue + g, MAX_QUEUE);
      // Gracefully decay burst load multiplier over ~15 ticks back to base
      if (d.loadMult > (d.baseLoadMult || 1.0)) {
        d.loadMult = Math.max(d.baseLoadMult || 1.0, d.loadMult * 0.94);
      }
    }

    // 2. Slotted Contention
    const contending = this.devices.filter(d => d.queue > 0);
    const nC = contending.length;

    if (nC > 0) {
      const taus = contending.map(d => Math.min(1.0, 2.0 / (d.activeCw + 1)));
      const avgTau = taus.reduce((s, t) => s + t, 0) / nC;
      const pCol = nC > 1 ? 1.0 - Math.pow(Math.max(0, 1.0 - avgTau), nC - 1) : 0.0;

      for (let idx = 0; idx < nC; idx++) {
        const d = contending[idx];
        const maxAttempts = Math.min(d.queue, Math.max(1, Math.round(SLOTS * taus[idx] * 0.035)));
        let suc = 0, col = 0;

        for (let a = 0; a < maxAttempts; a++) {
          if (rng() > pCol) suc++; else col++;
        }
        d._att = maxAttempts;
        d._suc = suc;
        d._col = col;
        d.queue = Math.max(0, d.queue - suc);
      }
    }

    // 3. Metric Updates & EWMA
    const alpha = 0.3;
    for (const d of this.devices) {
      const cr = d._att > 0 ? (d._col / d._att) : 0.0;
      const th = (d._suc * PKT_BYTES * 8) / (DT * 1e6);
      const dl = d._gen > 0 ? (d._suc / Math.max(d._gen, 1)) : d.deliveryRatio;

      d.collisionRate = alpha * cr + (1 - alpha) * d.collisionRate;
      d.throughput = alpha * th + (1 - alpha) * d.throughput;
      d.deliveryRatio = alpha * Math.min(1.0, dl) + (1 - alpha) * d.deliveryRatio;
      d.queueOcc = d.queue / MAX_QUEUE;
      d.loadEstimate = alpha * (d._gen / 14.0) + (1 - alpha) * d.loadEstimate;

      if (d._col > 0 && d.collisionRate > 0.04) {
        d.state = 'colliding';
      } else if (d.throughput > 0.0005 || d.queue > 0) {
        d.state = 'active';
      } else {
        d.state = 'idle';
      }

      this.strat.update(d);
      d._att = 0; d._suc = 0; d._col = 0; d._gen = 0;
    }

    return this.snapshot();
  }

  snapshot() {
    return {
      t: this.time,
      protocol: this.proto,
      devices: this.devices.map(d => ({
        id: d.id,
        state: d.isMuted ? 'muted' : d.state,
        isMuted: !!d.isMuted,
        cw_vo: d.cwMinVo,
        cw_be: d.cwMinBe,
        aifsn_vo: d.aifsnVo,
        aifsn_be: d.aifsnBe,
        throughput: d.throughput,
        collision: d.collisionRate,
        delivery: d.deliveryRatio,
        queue_occ: d.queueOcc,
        queue_count: d.queue,
        load_estimate: d.loadEstimate,
      })),
    };
  }
}

// ========================= SIMULATION RUNNER =========================
class SimulationRunner {
  constructor() {
    this.sims = new Map();
    this.running = false;
    this._interval = null;
    this.onFrame = null;
    this.neuralNet = null;
    this.speed = 1.0;
  }

  toggleMute(nodeId) {
    let muted = false;
    for (const [, sim] of this.sims) {
      muted = sim.toggleMute(nodeId);
    }
    return muted;
  }

  async loadWeights(modelName) {
    try {
      const name = modelName || 'sac_as_td3';
      const r = await fetch(`/api/weights/${name}`);
      if (r.ok) {
        const json = await r.json();
        this.neuralNet = new NeuralNet(json);
        return true;
      }
    } catch (e) {
      console.warn('Neural network weights load failed, using heuristic fallback:', e);
    }
    return false;
  }

  stepOnce() {
    for (const [p, sim] of this.sims) {
      try {
        const frame = sim.tick();
        if (this.onFrame) this.onFrame(p, frame);
      } catch (err) {
        console.error(`Simulation tick error for ${p}:`, err);
      }
    }
  }

  start(cfg) {
    this.stop();
    this.sims.clear();
    this.speed = cfg.speed || 1.0;

    const protocols = cfg.protocols && cfg.protocols.length ? cfg.protocols : ['SAC'];
    for (const p of protocols) {
      this.sims.set(p, new MACSimulator(cfg.devices, cfg.load, cfg.seed, p, this.neuralNet));
    }

    this.running = true;
    
    // Step immediately once so initial frame renders synchronously on load
    this.stepOnce();

    const intervalMs = Math.max(16, Math.round(100 / this.speed));
    this._interval = setInterval(() => {
      if (!this.running) return;
      this.stepOnce();
    }, intervalMs);
  }

  injectBurst(nodeId, count) {
    for (const [, sim] of this.sims) {
      sim.injectBurst(nodeId, count);
    }
  }

  stop() {
    this.running = false;
    if (this._interval) {
      clearInterval(this._interval);
      this._interval = null;
    }
  }
}

if (typeof window !== 'undefined') {
  window.SimulationRunner = SimulationRunner;
  window.STRATEGY_META = Object.fromEntries(
    Object.entries(Strategies).map(([k, v]) => [k, { label: v.label, color: v.color }])
  );
}
