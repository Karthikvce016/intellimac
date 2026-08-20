# NervNet — Real-Time Wireless MAC Digital Twin & RL Controller

Autonomous Deep Reinforcement Learning (Soft Actor-Critic) continuous contention window controller for IEEE 802.11 Wi-Fi networks in high-density MSME industrial environments.

---

## 🚀 Quick Start

Run the entire application with one command:

```bash
./run.sh
```

Then open your browser at:
👉 **[http://127.0.0.1:8600](http://127.0.0.1:8600)**

---

## 📁 Essential Project Structure

```text
important/
├── dashboard-v2/
│   ├── app/
│   │   └── main.py          # FastAPI backend (REST API + WebSocket stream)
│   ├── static/
│   │   ├── index.html       # Mission-control dashboard UI
│   │   ├── style.css        # Modern glassmorphic theme styling
│   │   ├── app.js           # Frontend controller, smooth SVG charts & telemetry
│   │   └── simulator.js     # Slotted MAC engine & in-browser neural net inference
│   └── run.sh               # Backend launcher script
│
├── ns-3.48/
│   ├── build/scratch/
│   │   └── ns3.48-msme_mac_comparison_v2-default  # Compiled ns-3 C++ binary
│   ├── scratch/
│   │   └── msme_mac_comparison_v2.cc             # C++ discrete-event MAC simulation
│   └── results/                                  # Validated benchmark CSV sweeps & traces
│
├── results/
│   └── models/
│       └── sac_as_td3_weights.json               # Trained SAC actor neural network weights
│
├── requirements.txt         # Minimal Python dependencies (fastapi, uvicorn, websockets)
├── run.sh                   # One-click startup script
└── README.md                # Project documentation
```

---

## ⚡ Key Features

1. **Live Wireless Mesh Web (Canvas):** Real-time animated 802.11 AP radio access network with interactive node inspector and packet burst injection.
2. **Tri-Protocol Head-to-Head Arena:** Simultaneous real-time comparison of **SAC (Ours)**, **DCF (802.11)**, and **GDCF (Adaptive)** with overlaid smooth telemetry scope curves.
3. **Native ns-3 Digital Twin Streaming:** Spawns C++ ns-3 discrete-event simulations in the background and pipes live telemetry to the browser via WebSocket.
4. **In-Browser Neural Net Inference:** Fully forward passes the 9D SAC actor model (`Linear(9, 400) -> ReLU -> Linear(400, 300) -> ReLU -> Linear(300, 4) -> Tanh`) inside JavaScript.
5. **Protocol Lab Benchmark Matrix:** Complete experimental evaluation sweeps covering throughput gain, collision reduction, latency, and Jain fairness index.
