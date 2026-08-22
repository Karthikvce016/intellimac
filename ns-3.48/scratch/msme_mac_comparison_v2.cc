/*
 * MSME MAC Layer Comparison v2: DCF vs GDCF vs SAC with Real-time Metrics
 * 
 * Improvements over v1:
 * - Real MAC metrics fed to SAC observation vector
 * - Extended action space: CWmin, CWmax, AIFS, TXOP limit
 * - MSME traffic patterns (periodic sensor + bursty alerts)
 */

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/wifi-module.h"
#include "ns3/applications-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/stats-module.h"
#include "ns3/yans-wifi-helper.h"
#include "ns3/wifi-tx-stats-helper.h"

#include <fstream>
#include <iomanip>
#include <map>
#include <vector>
#include <cmath>
#include <sstream>
#include <deque>
#include <functional>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("MsmeMacComparisonV2");

struct SimulationResult
{
    std::string protocol;
    uint32_t numNodes;
    double offeredLoad;
    uint32_t seed;
    
    // Aggregate (legacy)
    double throughput;
    double deliveryRatio;
    double collisionRate;
    double avgLatency;
    double fairness;
    
    // Per-class: Critical (periodic, port 9)
    double crit_throughput;
    double crit_deliveryRatio;
    double crit_collisionRate;
    double crit_avgLatency;
    double crit_fairness;
    
    // Per-class: Best-effort (bursty, port 10)
    double be_throughput;
    double be_deliveryRatio;
    double be_collisionRate;
    double be_avgLatency;
    double be_fairness;
    
    // CW parameters (average across nodes at end of simulation)
    double avg_cw_min;
    double avg_cw_max;
    double avg_aifsn;
    
    // Per-AC CW parameters (SAC per-class)
    double avg_cw_min_vo;
    double avg_cw_max_vo;
    double avg_cw_min_be;
    double avg_cw_max_be;
};

std::vector<SimulationResult> g_results;

// ============================================================
// Lightweight JSON Parser for SAC Weights
// ============================================================
class JsonParser
{
public:
    explicit JsonParser(const std::string& jsonStr) : m_json(jsonStr) {}

    std::vector<double> GetArray(const std::string& key) const
    {
        std::string searchKey = "\"" + key + "\"";
        size_t pos = m_json.find(searchKey);
        if (pos == std::string::npos) return {};
        pos = m_json.find('[', pos);
        if (pos == std::string::npos) return {};
        
        int bracketCount = 0;
        size_t endPos = pos;
        for (size_t i = pos; i < m_json.length(); ++i)
        {
            if (m_json[i] == '[') bracketCount++;
            else if (m_json[i] == ']') bracketCount--;
            if (bracketCount == 0)
            {
                endPos = i;
                break;
            }
        }
        if (bracketCount != 0) return {};

        std::string arrayStr = m_json.substr(pos + 1, endPos - pos - 1);
        std::vector<double> result;
        std::stringstream ss(arrayStr);
        std::string token;
        while (std::getline(ss, token, ','))
        {
            // Trim whitespace
            token.erase(0, token.find_first_not_of(" \t\n\r"));
            token.erase(token.find_last_not_of(" \t\n\r") + 1);
            if (!token.empty())
            {
                result.push_back(std::stod(token));
            }
        }
        return result;
    }

    std::vector<std::vector<double>> GetMatrix(const std::string& key, size_t rows, size_t cols) const
    {
        std::string searchKey = "\"" + key + "\"";
        size_t pos = m_json.find(searchKey);
        if (pos == std::string::npos) return {};

        // Find the opening bracket of the array
        pos = m_json.find('[', pos);
        if (pos == std::string::npos) return {};

        // Find the matching closing bracket for the outer array
        int bracketCount = 0;
        size_t endPos = pos;
        for (size_t i = pos; i < m_json.length(); ++i)
        {
            if (m_json[i] == '[') bracketCount++;
            else if (m_json[i] == ']') bracketCount--;
            if (bracketCount == 0)
            {
                endPos = i;
                break;
            }
        }
        if (bracketCount != 0) return {};

        std::string matrixStr = m_json.substr(pos + 1, endPos - pos - 1);
        std::vector<std::vector<double>> result(rows, std::vector<double>(cols));

        size_t rowStart = 0;
        for (size_t i = 0; i < rows; ++i)
        {
            // Find the next row (starts with '[')
            size_t rowBracketStart = matrixStr.find('[', rowStart);
            if (rowBracketStart == std::string::npos) break;
            
            // Find the end of this row (matching ']')
            int bracketCount = 0;
            size_t rowEnd = rowBracketStart;
            for (size_t j = rowBracketStart; j < matrixStr.length(); ++j)
            {
                if (matrixStr[j] == '[') bracketCount++;
                else if (matrixStr[j] == ']') bracketCount--;
                if (bracketCount == 0)
                {
                    rowEnd = j;
                    break;
                }
            }
            if (bracketCount != 0) break;

            std::string rowStr = matrixStr.substr(rowBracketStart + 1, rowEnd - rowBracketStart - 1);
            rowStart = rowEnd + 1;

            std::stringstream ss(rowStr);
            std::string token;
            size_t col = 0;
            while (std::getline(ss, token, ',') && col < cols)
            {
                // Trim whitespace
                token.erase(0, token.find_first_not_of(" \t\n\r"));
                token.erase(token.find_last_not_of(" \t\n\r") + 1);
                if (!token.empty())
                {
                    result[i][col++] = std::stod(token);
                }
            }
            if (col < cols)
            {
                NS_LOG_ERROR("GetMatrix: Row " << i << " has only " << col << " columns, expected " << cols << " for key " << key);
                return {};
            }
        }
        return result;
    }

private:
    std::string m_json;
};

// ============================================================
// MAC Metrics Collector - Tracks real-time MAC statistics
// ============================================================
class MacMetricsCollector
{
public:
    MacMetricsCollector() : m_lastUpdate(Seconds(0)), m_windowStart(Seconds(0)) {}

    void Install(Ptr<WifiNetDevice> device)
    {
        m_device = device;
        Ptr<WifiMac> mac = device->GetMac();

        // Trace MAC layer events for own-traffic metrics
        mac->TraceConnectWithoutContext("MacTx", MakeCallback(&MacMetricsCollector::OnMacTx, this));
        mac->TraceConnectWithoutContext("MacRx", MakeCallback(&MacMetricsCollector::OnMacRx, this));
        mac->TraceConnectWithoutContext("MacTxDrop", MakeCallback(&MacMetricsCollector::OnMacTxDrop, this));
        
        // Trace per-node TX success/failure (ACK/NACK/Drop)
        mac->TraceConnectWithoutContext("AckedMpdu", MakeCallback(&MacMetricsCollector::OnAckedMpdu, this));
        mac->TraceConnectWithoutContext("NAckedMpdu", MakeCallback(&MacMetricsCollector::OnNAckedMpdu, this));
        mac->TraceConnectWithoutContext("DroppedMpdu", MakeCallback(&MacMetricsCollector::OnDroppedMpdu, this));
    }

    void OnMacTx(Ptr<const Packet> packet)
    {
        m_macTxPackets++;
        m_macTxBytes += packet->GetSize();
    }

    void OnMacRx(Ptr<const Packet> packet)
    {
        m_macRxPackets++;
        m_macRxBytes += packet->GetSize();
    }

    void OnMacTxDrop(Ptr<const Packet> packet)
    {
        m_macTxDrops++;
    }

    void OnAckedMpdu(Ptr<const WifiMpdu> mpdu)
    {
        m_ackedMpdus++;
    }

    void OnNAckedMpdu(Ptr<const WifiMpdu> mpdu)
    {
        m_nackedMpdus++;
    }

    void OnDroppedMpdu(WifiMacDropReason reason, Ptr<const WifiMpdu> mpdu)
    {
        m_droppedMpdus++;
    }

    // Compute metrics over measurement window
    void UpdateMetrics(Time now)
    {
        if (m_windowStart.IsZero())
        {
            m_windowStart = now;
            return;
        }

        double windowSec = (now - m_windowStart).GetSeconds();
        if (windowSec < 0.05) return; // at least 50ms window

        // Collision rate & Throughput based on actual MAC TX attempts and drops
        uint64_t successTx = 0;
        if (m_ackedMpdus > 0)
        {
            successTx = m_ackedMpdus;
            uint64_t totalAttempts = m_ackedMpdus + m_nackedMpdus + m_droppedMpdus;
            m_collisionRate = totalAttempts > 0 ? (1.0 - static_cast<double>(m_ackedMpdus) / totalAttempts) : 0.0;
        }
        else if (m_macTxPackets > 0)
        {
            successTx = (m_macTxPackets >= m_macTxDrops) ? (m_macTxPackets - m_macTxDrops) : 0;
            m_collisionRate = static_cast<double>(m_macTxDrops) / m_macTxPackets;
        }
        else
        {
            m_collisionRate = 0.0;
        }

        // Delivery ratio
        if (m_macTxPackets > 0)
        {
            m_deliveryRatio = static_cast<double>(successTx) / m_macTxPackets;
        }
        else
        {
            m_deliveryRatio = 1.0;
        }

        // Throughput in Mbps
        if (m_macTxBytes > 0)
        {
            m_throughput = (m_macTxBytes * 8.0) / (windowSec * 1e6);
        }
        else if (successTx > 0)
        {
            m_throughput = (successTx * 128.0 * 8.0) / (windowSec * 1e6);
        }
        else
        {
            m_throughput = 0.0;
        }

        // Estimate load from queue occupancy (both VO and BE)
        if (m_device)
        {
            Ptr<WifiMac> mac = m_device->GetMac();
            if (mac)
            {
                Ptr<WifiMacQueue> queueVo = mac->GetTxopQueue(AC_VO);
                Ptr<WifiMacQueue> queueBe = mac->GetTxopQueue(AC_BE);
                uint32_t nPkts = (queueVo ? queueVo->GetNPackets() : 0) + (queueBe ? queueBe->GetNPackets() : 0);
                uint32_t maxPkts = (queueVo ? queueVo->GetMaxSize().GetValue() : 400) + (queueBe ? queueBe->GetMaxSize().GetValue() : 400);
                m_queueOccupancy = maxPkts > 0 ? (static_cast<double>(nPkts) / maxPkts) : 0.0;
            }
        }

        // Reset window counters
        m_windowStart = now;
        m_macTxPackets = 0;
        m_macTxBytes = 0;
        m_macTxDrops = 0;
        m_ackedMpdus = 0;
        m_nackedMpdus = 0;
        m_droppedMpdus = 0;
    }

    // Get observation vector for TD3: [load, collision_rate, delivery_ratio, latency_norm, queue_occ, cw_min_norm, mult_norm]
    std::array<double, 7> GetObservation(uint32_t cwMin, uint32_t cwMax, uint8_t aifsn) const
    {
        double loadEstimate = std::min(1.0, m_queueOccupancy + m_throughput / 54.0); // Normalize to 54 Mbps max
        double latencyNorm = std::min(1.0, m_collisionRate * 10.0); // Proxy for latency
        double cwMinNorm = (cwMin - 16.0) / (1024.0 - 16.0);
        double multNorm = 0.5; // Fixed for now

        return {
            loadEstimate,
            m_collisionRate,
            m_deliveryRatio,
            latencyNorm,
            m_queueOccupancy,
            cwMinNorm,
            multNorm
        };
    }

    // Extended observation with separate VO/BE CWmin and mult for per-class observation matching sim
    // 9D: [loadEstimate, collision_rate, delivery_ratio, latency_norm, queue_occ,
    //      cw_crit_norm, cw_be_norm, mult_crit_norm, mult_be_norm]
    std::array<double, 9> GetObservationWithVO(uint32_t cwMinBE, uint32_t cwMaxBE, uint8_t aifsnBE,
                                                uint32_t cwMinVO, uint32_t cwMaxVO, uint8_t aifsnVO) const
    {
        double loadEstimate = std::min(1.0, m_queueOccupancy + m_throughput / 54.0); // Normalize to 54 Mbps max
        double latencyNorm = std::min(1.0, m_collisionRate * 10.0); // Proxy for latency
        
        // VO/Critical CWmin norm (3-1023)
        double cwMinNormVO = (cwMinVO - 3.0) / (1023.0 - 3.0);
        // BE CWmin norm (16-1024)
        double cwMinNormBE = (cwMinBE - 16.0) / (1024.0 - 16.0);
        
        // Mult signals: use multiplier = CWmax/CWmin, normalized to [0,1] for range [1,4]
        double multVO = static_cast<double>(cwMaxVO) / std::max(1u, cwMinVO);
        double multBE = static_cast<double>(cwMaxBE) / std::max(1u, cwMinBE);
        double multNormVO = std::min(1.0, std::max(0.0, (multVO - 1.0) / 3.0));
        double multNormBE = std::min(1.0, std::max(0.0, (multBE - 1.0) / 3.0));

        return {
            loadEstimate,
            m_collisionRate,
            m_deliveryRatio,
            latencyNorm,
            m_queueOccupancy,
            cwMinNormVO,      // slot 5: VO/Critical CWmin
            cwMinNormBE,      // slot 6: BE CWmin
            multNormVO,       // slot 7: VO/Critical multiplier
            multNormBE        // slot 8: BE multiplier
        };
    }

    double GetCollisionRate() const { return m_collisionRate; }
    double GetDeliveryRatio() const { return m_deliveryRatio; }
    double GetThroughput() const { return m_throughput; }
    double GetQueueOccupancy() const { return m_queueOccupancy; }

private:
    Ptr<WifiNetDevice> m_device;
    Time m_lastUpdate;
    Time m_windowStart;

    // Window counters (per-node MAC-layer)
    uint64_t m_macTxPackets{0};
    uint64_t m_macRxPackets{0};
    uint64_t m_macTxBytes{0};
    uint64_t m_macRxBytes{0};
    uint64_t m_macTxDrops{0};
    uint64_t m_ackedMpdus{0};
    uint64_t m_nackedMpdus{0};
    uint64_t m_droppedMpdus{0};

    // Computed metrics
    double m_collisionRate{0.0};
    double m_deliveryRatio{0.0};
    double m_throughput{0.0};
    double m_queueOccupancy{0.0};
};
 
// ============================================================
// SAC Inference Engine (Actor Network: 9->400->300->4)
// Per-class observation: 9D
// ============================================================
class LocalTd3Inference
{
public:
    LocalTd3Inference() : m_loaded(false) {}

    bool LoadWeights(const std::string& weightsPath)
    {
        std::ifstream file(weightsPath);
        if (!file.is_open())
        {
            NS_LOG_ERROR("Failed to open weights file: " << weightsPath);
            return false;
        }

        std::stringstream buffer;
        buffer << file.rdbuf();
        file.close();

        JsonParser parser(buffer.str());

        m_w1 = parser.GetMatrix("mu.0.weight", 400, 9);
        m_b1 = parser.GetArray("mu.0.bias");
        m_w2 = parser.GetMatrix("mu.2.weight", 300, 400);
        m_b2 = parser.GetArray("mu.2.bias");
        m_w3 = parser.GetMatrix("mu.4.weight", 4, 300);
        m_b3 = parser.GetArray("mu.4.bias");

        if (m_w1.empty() || m_b1.empty() || m_w2.empty() || m_b2.empty() || m_w3.empty() || m_b3.empty())
        {
            NS_LOG_ERROR("Failed to parse weights from JSON");
            return false;
        }

        NS_LOG_INFO("SAC weights loaded successfully (4 outputs, 9D input)");
        m_loaded = true;
        return true;
    }

    // Returns: [cw_min_action, cw_max_action, aifs_action, txop_action] all in [-1, 1]
    std::array<double, 4> Predict(const std::array<double, 9>& observation) const
    {
        if (!m_loaded)
        {
            return {0.0, 0.0, 0.0, 0.0};
        }

        std::vector<double> x(observation.begin(), observation.end());

        // Layer 1: 9 -> 400, ReLU
        std::vector<double> h1 = MatVecMul(m_w1, x, m_b1);
        for (auto& v : h1) v = std::max(0.0, v);

        // Layer 2: 400 -> 300, ReLU
        std::vector<double> h2 = MatVecMul(m_w2, h1, m_b2);
        for (auto& v : h2) v = std::max(0.0, v);

        // Layer 3: 300 -> 4, Tanh
        std::vector<double> out = MatVecMul(m_w3, h2, m_b3);
        for (auto& v : out) v = std::tanh(v);

        return {out[0], out[1], out[2], out[3]};
    }

private:
    std::vector<std::vector<double>> m_w1, m_w2, m_w3;
    std::vector<double> m_b1, m_b2, m_b3;
    bool m_loaded;

    std::vector<double> MatVecMul(const std::vector<std::vector<double>>& w,
                                   const std::vector<double>& x,
                                   const std::vector<double>& b) const
    {
        std::vector<double> result(w.size(), 0.0);
        for (size_t i = 0; i < w.size(); ++i)
        {
            double sum = b[i];
            for (size_t j = 0; j < x.size(); ++j)
            {
                sum += w[i][j] * x[j];
            }
            result[i] = sum;
        }
        return result;
    }
};

// ============================================================
// Free helper: snap any integer to the nearest valid 2^n-1 CW value
// ============================================================
static inline uint32_t SnapToValidCw(uint32_t cw)
{
    uint32_t v = 1;
    while (v < cw) v = (v << 1) | 1;
    return std::min(v, 1023u);
}

// ============================================================
// SAC MAC Controller with Real-time Metrics (Per-Class: AC_VO + AC_BE)
// ============================================================
class LocalTd3MacController
{
public:
    LocalTd3MacController() : m_td3(std::make_unique<LocalTd3Inference>()), m_metrics(std::make_unique<MacMetricsCollector>()),
                              m_cwMinVo(3), m_cwMaxVo(7), m_cwMinBe(15), m_cwMaxBe(1024) {}

    void Setup(Ptr<Txop> txopVo, Ptr<Txop> txopBe, Ptr<WifiNetDevice> device, const std::string& weightsPath)
    {
        m_txopVo = txopVo;
        m_txopBe = txopBe;
        m_device = device;
        m_td3->LoadWeights(weightsPath);
        m_metrics->Install(device);
        m_cwMinVo = txopVo ? txopVo->GetMinCw() : 3;
        m_cwMaxVo = txopVo ? txopVo->GetMaxCw() : 7;
        m_cwMinBe = txopBe ? txopBe->GetMinCw() : 15;
        m_cwMaxBe = txopBe ? txopBe->GetMaxCw() : 1024;
        ScheduleInference();
        ScheduleMetricsUpdate();
    }

    uint32_t GetCurrentCwMinVo() const { return m_cwMinVo; }
    uint32_t GetCurrentCwMaxVo() const { return m_cwMaxVo; }
    uint32_t GetCurrentCwMinBe() const { return m_cwMinBe; }
    uint32_t GetCurrentCwMaxBe() const { return m_cwMaxBe; }
    uint8_t GetCurrentAifsnVo() const { return m_txopVo ? m_txopVo->GetAifsn() : 0; }
    uint8_t GetCurrentAifsnBe() const { return m_txopBe ? m_txopBe->GetAifsn() : 0; }

private:
    // SnapToValidCw is now a free function above

    void ScheduleInference()
    {
        // First inference at 1.5s (traffic starts at 1.0s + 200ms for first metrics window)
        static bool first = true;
        if (first) {
            first = false;
            Simulator::Schedule(MilliSeconds(1500), &LocalTd3MacController::RunInference, this);
        } else {
            Simulator::Schedule(MilliSeconds(100), &LocalTd3MacController::RunInference, this);
        }
    }

    void ScheduleMetricsUpdate()
    {
        Simulator::Schedule(MilliSeconds(100), &LocalTd3MacController::UpdateMetrics, this);
    }

    void UpdateMetrics()
    {
        m_metrics->UpdateMetrics(Simulator::Now());
        ScheduleMetricsUpdate();
    }

    void RunInference()
    {
        if (!m_txopVo || !m_txopBe || !m_device) return;

        // Get observation with both VO and BE CWmin for per-class observation
        auto obs = m_metrics->GetObservationWithVO(
            m_txopBe->GetMinCw(),
            m_txopBe->GetMaxCw(),
            m_txopBe->GetAifsn(),
            m_txopVo->GetMinCw(),
            m_txopVo->GetMaxCw(),
            m_txopVo->GetAifsn()
        );

        auto action = m_td3->Predict(obs);

        // 4D action: [crit_cwmin, crit_mult, be_cwmin, be_mult] all in [-1, 1]
        // CWmin uses a UNIFORM INDEX over the 2^n-1 grid, exactly matching
        // sim/mac_simulator_perclass.py::_action_to_cw (floor(x+0.5) rounding,
        // the same semantics as std::round). The previous linear mapping
        // (3 + (a+1)*510) sent a=0 to CW=1023 while the simulator mapped a=0
        // to CW=63 -- that decode mismatch is why SAC sat at CW 1023/1023.
        static const uint32_t critCwGrid[9] = {3, 7, 15, 31, 63, 127, 255, 511, 1023};
        static const uint32_t beCwGrid[7] = {15, 31, 63, 127, 255, 511, 1023};
        int critIdx = static_cast<int>(std::floor((action[0] + 1.0) * 0.5 * 8.0 + 0.5));
        critIdx = std::max(0, std::min(8, critIdx));
        uint32_t voCwMin = critCwGrid[critIdx];

        // Map action[1] -> AC_VO multiplier [1.0, 4.0]
        double voMult = 1.0 + (action[1] + 1.0) * 0.5 * (4.0 - 1.0);
        uint32_t voCwMax = SnapToValidCw(static_cast<uint32_t>(voCwMin * voMult));
        voCwMax = std::max(voCwMax, voCwMin);

        int beIdx = static_cast<int>(std::floor((action[2] + 1.0) * 0.5 * 6.0 + 0.5));
        beIdx = std::max(0, std::min(6, beIdx));
        uint32_t beCwMin = beCwGrid[beIdx];

        // Map action[3] -> AC_BE multiplier [1.0, 4.0]
        double beMult = 1.0 + (action[3] + 1.0) * 0.5 * (4.0 - 1.0);
        uint32_t beCwMax = SnapToValidCw(static_cast<uint32_t>(beCwMin * beMult));
        beCwMax = std::max(beCwMax, beCwMin);

        // Apply to both TXOPs
        m_txopVo->SetMinCw(voCwMin);
        m_txopVo->SetMaxCw(voCwMax);
        m_txopBe->SetMinCw(beCwMin);
        m_txopBe->SetMaxCw(beCwMax);

        // Store for logging
        m_cwMinVo = voCwMin;
        m_cwMaxVo = voCwMax;
        m_cwMinBe = beCwMin;
        m_cwMaxBe = beCwMax;

        NS_LOG_INFO("SAC: VO_CWmin=" << voCwMin << " VO_CWmax=" << voCwMax
                  << " BE_CWmin=" << beCwMin << " BE_CWmax=" << beCwMax
                  << " | Obs: load=" << obs[0] << " coll=" << obs[1] << " del=" << obs[2]
                  << " queue=" << obs[4]);

        ScheduleInference();
    }

    Ptr<Txop> m_txopVo;
    Ptr<Txop> m_txopBe;
    Ptr<WifiNetDevice> m_device;
    std::unique_ptr<LocalTd3Inference> m_td3;
    std::unique_ptr<MacMetricsCollector> m_metrics;
    uint32_t m_cwMinVo, m_cwMaxVo;
    uint32_t m_cwMinBe, m_cwMaxBe;
};

// ============================================================
// Common base for GDCF Managers
// ============================================================
class LocalGdcfManagerBase
{
public:
    LocalGdcfManagerBase() : m_cwMin(16), m_cwMax(1024), m_minCwMin(16), m_maxCwMax(1024),
                        m_targetCollisionRate(0.1), m_successCount(0), m_failureCount(0), m_numNodes(50) {}
    virtual ~LocalGdcfManagerBase() = default;
    virtual void Setup(Ptr<Txop> txop, Ptr<WifiMac> mac, uint32_t numNodes = 50) = 0;
    virtual void Start() = 0;
    virtual void ReportTxSuccess(Ptr<const Packet> packet) = 0;
    virtual void ReportTxFailure(Ptr<const Packet> packet) = 0;
    virtual uint32_t GetCurrentCwMin() const = 0;
    virtual uint32_t GetCurrentCwMax() const = 0;
    virtual uint8_t GetCurrentAifsn() const = 0;

protected:
    Ptr<Txop> m_txop;
    uint32_t m_cwMin, m_cwMax;
    uint32_t m_minCwMin, m_maxCwMax;
    double m_targetCollisionRate;
    uint32_t m_successCount, m_failureCount;
    uint32_t m_numNodes;
    EventId m_adjustmentEvent;
};

// ============================================================
// GDCF Manager (Original - collapses to CWmin=15)
// ============================================================
class LocalGdcfManagerOriginal : public LocalGdcfManagerBase
{
public:
    LocalGdcfManagerOriginal() = default;

    void Setup(Ptr<Txop> txop, Ptr<WifiMac> mac, uint32_t numNodes = 50) override
    { 
        m_txop = txop; 
        m_cwMin = txop->GetMinCw(); 
        m_cwMax = txop->GetMaxCw();
        // Original GDCF: minimum CW is 15 (allows collapse to 15)
        m_minCwMin = 15;
        m_maxCwMax = 1024;
        m_targetCollisionRate = 0.1;
        m_successCount = 0;
        m_failureCount = 0;
        
        // Connect trace callbacks for success/failure tracking
        mac->TraceConnectWithoutContext("MacTx", MakeCallback(&LocalGdcfManagerOriginal::ReportTxSuccess, this));
        mac->TraceConnectWithoutContext("MacTxDrop", MakeCallback(&LocalGdcfManagerOriginal::ReportTxFailure, this));
    }
    void Start() override { m_successCount = 0; m_failureCount = 0; ScheduleAdjustment(); }

    void ReportTxSuccess(Ptr<const Packet> packet) override { m_successCount++; }
    void ReportTxFailure(Ptr<const Packet> packet) override { m_failureCount++; }

    uint32_t GetCurrentCwMin() const override { return m_txop ? m_txop->GetMinCw() : 0; }
    uint32_t GetCurrentCwMax() const override { return m_txop ? m_txop->GetMaxCw() : 0; }
    uint8_t GetCurrentAifsn() const override { return m_txop ? m_txop->GetAifsn() : 0; }

private:
    void ScheduleAdjustment() { m_adjustmentEvent = Simulator::Schedule(MilliSeconds(100), &LocalGdcfManagerOriginal::AdjustContentionWindow, this); }

    void AdjustContentionWindow()
    {
        uint32_t total = m_successCount + m_failureCount;
        if (total == 0) { ScheduleAdjustment(); return; }

        double observedCollisionRate = static_cast<double>(m_failureCount) / total;
        double ratio = observedCollisionRate / m_targetCollisionRate;

        // Ensure CW never goes below minimum floor (15 for original GDCF)
        if (m_cwMin < m_minCwMin)
        {
            m_cwMin = m_minCwMin;
        }

        if (ratio > 1.2)
        {
            m_cwMin = SnapToValidCw(std::min(m_cwMax, static_cast<uint32_t>(m_cwMin * 1.5)));
            m_cwMax = SnapToValidCw(std::min(m_maxCwMax, static_cast<uint32_t>(m_cwMax * 1.2)));
        }
        else if (ratio < 0.8 && m_cwMin > m_minCwMin)
        {
            m_cwMin = SnapToValidCw(std::max(m_minCwMin, static_cast<uint32_t>(m_cwMin * 0.75)));
            m_cwMax = SnapToValidCw(std::max(m_cwMin, static_cast<uint32_t>(m_cwMax * 0.9)));
        }

        if (m_txop)
        {
            m_txop->SetMinCw(m_cwMin);
            m_txop->SetMaxCw(m_cwMax);
        }

        m_successCount = 0;
        m_failureCount = 0;
        ScheduleAdjustment();
    }

    Ptr<Txop> m_txop;
    uint32_t m_cwMin, m_cwMax;
    uint32_t m_minCwMin, m_maxCwMax;
    double m_targetCollisionRate;
    uint32_t m_successCount, m_failureCount;
    EventId m_adjustmentEvent;
};

class LocalGdcfManager : public LocalGdcfManagerBase
{
public:
    LocalGdcfManager() = default;

    void Setup(Ptr<Txop> txop, Ptr<WifiMac> mac, uint32_t numNodes = 50) override
    { 
        m_txop = txop; 
        m_cwMin = txop->GetMinCw(); 
        m_cwMax = txop->GetMaxCw();
        m_numNodes = numNodes;
        // Set adaptive minimum CW floor based on number of nodes
        // Higher node count = higher minimum CW to reduce collision probability
        if (m_numNodes >= 100) m_minCwMin = 127;
        else if (m_numNodes >= 50) m_minCwMin = 63;
        else m_minCwMin = 31;
        
        // Connect trace callbacks for success/failure tracking
        mac->TraceConnectWithoutContext("MacTx", MakeCallback(&LocalGdcfManager::ReportTxSuccess, this));
        mac->TraceConnectWithoutContext("MacTxDrop", MakeCallback(&LocalGdcfManager::ReportTxFailure, this));
    }
    void Start() override { m_successCount = 0; m_failureCount = 0; ScheduleAdjustment(); }

    void ReportTxSuccess(Ptr<const Packet> packet) override { m_successCount++; }
    void ReportTxFailure(Ptr<const Packet> packet) override { m_failureCount++; }

    uint32_t GetCurrentCwMin() const override { return m_txop ? m_txop->GetMinCw() : 0; }
    uint32_t GetCurrentCwMax() const override { return m_txop ? m_txop->GetMaxCw() : 0; }
    uint8_t GetCurrentAifsn() const override { return m_txop ? m_txop->GetAifsn() : 0; }

private:
    void ScheduleAdjustment() { m_adjustmentEvent = Simulator::Schedule(MilliSeconds(100), &LocalGdcfManager::AdjustContentionWindow, this); }

    void AdjustContentionWindow()
    {
        uint32_t total = m_successCount + m_failureCount;
        if (total == 0) { ScheduleAdjustment(); return; }

        double observedCollisionRate = static_cast<double>(m_failureCount) / total;
        double ratio = observedCollisionRate / m_targetCollisionRate;

        // Ensure CW never goes below adaptive minimum floor
        if (m_cwMin < m_minCwMin)
        {
            m_cwMin = m_minCwMin;
        }

        if (ratio > 1.2)
        {
            m_cwMin = SnapToValidCw(std::min(m_cwMax, static_cast<uint32_t>(m_cwMin * 1.5)));
            m_cwMax = SnapToValidCw(std::min(m_maxCwMax, static_cast<uint32_t>(m_cwMax * 1.2)));
        }
        else if (ratio < 0.8 && m_cwMin > m_minCwMin)
        {
            m_cwMin = SnapToValidCw(std::max(m_minCwMin, static_cast<uint32_t>(m_cwMin * 0.75)));
            m_cwMax = SnapToValidCw(std::max(m_cwMin, static_cast<uint32_t>(m_cwMax * 0.9)));
        }

        if (m_txop)
        {
            m_txop->SetMinCw(m_cwMin);
            m_txop->SetMaxCw(m_cwMax);
        }

        m_successCount = 0;
        m_failureCount = 0;
        ScheduleAdjustment();
    }
};

// ============================================================
// MSME Traffic Generator - Periodic + Bursty
// ============================================================
class MsmeTrafficGenerator
{
public:
    static void Install(NodeContainer& staNodes, Ipv4Address apAddress, uint16_t port, 
                        double periodicRateKbps, double burstRateMbps, double burstProbability)
    {
        // Periodic sensor traffic (e.g., 10 kbps per node)
        OnOffHelper periodic("ns3::UdpSocketFactory", Address(InetSocketAddress(apAddress, port)));
        periodic.SetConstantRate(DataRate(periodicRateKbps * 1000));
        periodic.SetAttribute("PacketSize", UintegerValue(100)); // Small sensor packets
        periodic.SetAttribute("OnTime", StringValue("ns3::ConstantRandomVariable[Constant=1]"));
        periodic.SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=0]"));

        // Bursty alert traffic (Poisson arrivals with high rate)
        OnOffHelper burst("ns3::UdpSocketFactory", Address(InetSocketAddress(apAddress, port + 1)));
        burst.SetConstantRate(DataRate(burstRateMbps * 1e6));
        burst.SetAttribute("PacketSize", UintegerValue(1500));
        burst.SetAttribute("OnTime", StringValue("ns3::ExponentialRandomVariable[Mean=0.1]"));
        burst.SetAttribute("OffTime", StringValue("ns3::ExponentialRandomVariable[Mean=1.0]"));

        for (uint32_t i = 0; i < staNodes.GetN(); ++i)
        {
            Ptr<Node> node = staNodes.Get(i);
            
            // Always install periodic traffic
            ApplicationContainer periodicApps = periodic.Install(node);
            periodicApps.Start(Seconds(1.0));
            periodicApps.Stop(Seconds(30.0));

            // Install burst traffic with probability
            if (burstProbability > 0)
            {
                // Use a random variable to decide
                Ptr<UniformRandomVariable> rng = CreateObject<UniformRandomVariable>();
                if (rng->GetValue() < burstProbability)
                {
                    ApplicationContainer burstApps = burst.Install(node);
                    burstApps.Start(Seconds(1.0));
                    burstApps.Stop(Seconds(30.0));
                }
            }
        }
    }
};


// Global managers (after all class definitions)
std::vector<std::unique_ptr<LocalGdcfManagerBase>> g_gdcfManagers;
std::vector<std::unique_ptr<LocalTd3MacController>> g_td3Controllers;

// ============================================================
// Simulation Runner
// ============================================================
void RunSimulation(const std::string& protocol, uint32_t nNodes, double offeredLoad, 
                   const std::string& sacWeightsPath, double simTime, bool useMsmeTraffic, uint32_t seed,
                   bool perDeviceCsv, bool liveStream)
{
    NS_LOG_INFO("Running " << protocol << " with " << nNodes << " nodes, load=" << offeredLoad);

    // Create nodes
    NodeContainer wifiStaNodes;
    wifiStaNodes.Create(nNodes);
    NodeContainer wifiApNode;
    wifiApNode.Create(1);

    // Mobility - Grid layout
    MobilityHelper mobility;
    mobility.SetPositionAllocator("ns3::GridPositionAllocator",
                                  "MinX", DoubleValue(0.0),
                                  "MinY", DoubleValue(0.0),
                                  "DeltaX", DoubleValue(5.0),
                                  "DeltaY", DoubleValue(10.0),
                                  "GridWidth", UintegerValue(10),
                                  "LayoutType", StringValue("RowFirst"));
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.Install(wifiStaNodes);
    mobility.Install(wifiApNode);

    // WiFi Channel & PHY
    YansWifiChannelHelper channel = YansWifiChannelHelper::Default();
    YansWifiPhyHelper phy;
    phy.SetChannel(channel.Create());
    phy.Set("TxPowerStart", DoubleValue(20.0));
    phy.Set("TxPowerEnd", DoubleValue(20.0));

    // WiFi MAC
    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211ax);
    wifi.SetRemoteStationManager("ns3::ConstantRateWifiManager",
                                 "DataMode", StringValue("HeMcs0"),
                                 "ControlMode", StringValue("HeMcs0"));

    WifiMacHelper mac;
    Ssid ssid = Ssid("msme-network");
    mac.SetType("ns3::ApWifiMac", "Ssid", SsidValue(ssid));
    NetDeviceContainer apDevice = wifi.Install(phy, mac, wifiApNode);

    mac.SetType("ns3::StaWifiMac", "Ssid", SsidValue(ssid));
    NetDeviceContainer staDevices = wifi.Install(phy, mac, wifiStaNodes);

    // Internet stack
    InternetStackHelper stack;
    stack.Install(wifiStaNodes);
    stack.Install(wifiApNode);

    Ipv4AddressHelper address;
    address.SetBase("10.1.1.0", "255.255.255.0");
    Ipv4InterfaceContainer staInterfaces = address.Assign(staDevices);
    Ipv4InterfaceContainer apInterface = address.Assign(apDevice);

    // Traffic generation
    uint16_t port = 9;
    
    if (useMsmeTraffic)
    {
        // MSME traffic scaled by offeredLoad: periodic = 10kbps * load/0.1, burst = 1Mbps * load/0.1
        double scale = offeredLoad / 0.1;
        MsmeTrafficGenerator::Install(wifiStaNodes, apInterface.GetAddress(0), port,
                                      10.0 * scale, 1.0 * scale, 0.1);
    }
    else
    {
        // Standard saturation traffic
        double dataRateBps = offeredLoad * 1e6;
        OnOffHelper onOff("ns3::UdpSocketFactory", InetSocketAddress(apInterface.GetAddress(0), port));
        onOff.SetConstantRate(DataRate(dataRateBps));
        onOff.SetAttribute("PacketSize", UintegerValue(1500));
        onOff.SetAttribute("OnTime", StringValue("ns3::ConstantRandomVariable[Constant=1]"));
        onOff.SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=0]"));

        for (uint32_t i = 0; i < nNodes; ++i)
        {
            onOff.Install(wifiStaNodes.Get(i)).Start(Seconds(1.0));
        }
    }

    // Packet sink on AP (both ports)
    PacketSinkHelper sink1("ns3::UdpSocketFactory", InetSocketAddress(Ipv4Address::GetAny(), port));
    PacketSinkHelper sink2("ns3::UdpSocketFactory", InetSocketAddress(Ipv4Address::GetAny(), port + 1));
    ApplicationContainer sinkApps = sink1.Install(wifiApNode.Get(0));
    sinkApps.Add(sink2.Install(wifiApNode.Get(0)));
    sinkApps.Start(Seconds(0.0));
    sinkApps.Stop(Seconds(simTime));

    // Flow Monitor
    FlowMonitorHelper flowmon;
    Ptr<FlowMonitor> monitor = flowmon.InstallAll();

    // Protocol-specific setup
if (protocol == "GDCF")
    {
        for (uint32_t i = 0; i < nNodes; ++i)
        {
            Ptr<NetDevice> dev = staDevices.Get(i);
            Ptr<WifiNetDevice> wifiDev = DynamicCast<WifiNetDevice>(dev);
            if (!wifiDev) continue;
            Ptr<WifiMac> macPtr = wifiDev->GetMac();
            if (!macPtr) continue;
            Ptr<Txop> dcf = macPtr->GetTxopFor(AC_BE);
            if (dcf)
            {
                auto gdcf = std::make_unique<LocalGdcfManager>();
                gdcf->Setup(dcf, wifiDev->GetMac(), nNodes);
                gdcf->Start();
                g_gdcfManagers.push_back(std::move(gdcf));
            }
        }
    }
    else if (protocol == "GDCF_ORIG")
    {
        for (uint32_t i = 0; i < nNodes; ++i)
        {
            Ptr<NetDevice> dev = staDevices.Get(i);
            Ptr<WifiNetDevice> wifiDev = DynamicCast<WifiNetDevice>(dev);
            if (!wifiDev) continue;
            Ptr<WifiMac> macPtr = wifiDev->GetMac();
            if (!macPtr) continue;
            Ptr<Txop> dcf = macPtr->GetTxopFor(AC_BE);
            if (dcf)
            {
                auto gdcf = std::make_unique<LocalGdcfManagerOriginal>();
                gdcf->Setup(dcf, wifiDev->GetMac(), nNodes);
                gdcf->Start();
                g_gdcfManagers.push_back(std::move(gdcf));
            }
        }
    }
    else if (protocol == "SAC")
    {
        for (uint32_t i = 0; i < nNodes; ++i)
        {
            Ptr<NetDevice> dev = staDevices.Get(i);
            Ptr<WifiNetDevice> wifiDev = DynamicCast<WifiNetDevice>(dev);
            if (!wifiDev) continue;
            Ptr<WifiMac> macPtr = wifiDev->GetMac();
            if (!macPtr) continue;
            Ptr<Txop> txopVo = macPtr->GetTxopFor(AC_VO);
            Ptr<Txop> txopBe = macPtr->GetTxopFor(AC_BE);
            NS_ASSERT_MSG(txopVo && txopBe, "QoS TXOPs not found - MAC not QoS-enabled?");
            NS_ASSERT_MSG(txopVo != txopBe, "VO and BE TXOP pointers identical - shared object!");
            if (txopVo && txopBe)
            {
                auto sacCtrl = std::make_unique<LocalTd3MacController>();
                sacCtrl->Setup(txopVo, txopBe, wifiDev, sacWeightsPath);
                g_td3Controllers.push_back(std::move(sacCtrl));
            }
}
    }

    // Per-device data collection callback
    auto perDeviceCallback = [&, perDeviceCsv, liveStream](uint32_t nodeId, const MacMetricsCollector* metrics,
                                  const LocalTd3MacController* sacCtrl,
                                  const Ptr<WifiNetDevice>& device, 
                                  double timestamp) -> void
    {
        if (!perDeviceCsv && !liveStream) return;
        
        // Keyed by full filename so successive protocol/config runs never
        // write into a previous run's open stream
        static std::map<std::string, std::ofstream> deviceFiles;

        std::string filename = "results/per_device_" + protocol + "_" + std::to_string(nNodes) + "_" +
                               std::to_string(offeredLoad) + "_" + std::to_string(seed) + "_node" + std::to_string(nodeId) + ".csv";
        if (perDeviceCsv)
        {
            auto fileIt = deviceFiles.find(filename);
            if (fileIt == deviceFiles.end())
            {
                fileIt = deviceFiles.emplace(filename, std::ofstream{}).first;
                fileIt->second.open(filename);
                if (fileIt->second.is_open())
                {
                    fileIt->second << "timestamp,node_id,cw_min_vo,cw_max_vo,cw_min_be,cw_max_be,"
                                      "aifsn_vo,aifsn_be,throughput_vo,throughput_be,"
                                      "latency_vo,latency_be,delivery_vo,delivery_be,"
                                      "collision_vo,collision_be,queue_occ_vo,queue_occ_be,load_estimate\n";
                }
            }
        }
        
        // Get CW from SAC controller if available
        uint32_t cwMinVo = 0, cwMaxVo = 0, cwMinBe = 0, cwMaxBe = 0;
        uint8_t aifsnVo = 0, aifsnBe = 0;
        
        if (sacCtrl)
        {
            cwMinVo = sacCtrl->GetCurrentCwMinVo();
            cwMaxVo = sacCtrl->GetCurrentCwMaxVo();
            cwMinBe = sacCtrl->GetCurrentCwMinBe();
            cwMaxBe = sacCtrl->GetCurrentCwMaxBe();
            aifsnVo = sacCtrl->GetCurrentAifsnVo();
            aifsnBe = sacCtrl->GetCurrentAifsnBe();
        }
        else
        {
            Ptr<WifiMac> mac = device->GetMac();
            if (mac)
            {
                Ptr<Txop> txopVo = mac->GetTxopFor(AC_VO);
                Ptr<Txop> txopBe = mac->GetTxopFor(AC_BE);
                if (txopVo) { cwMinVo = txopVo->GetMinCw(); cwMaxVo = txopVo->GetMaxCw(); aifsnVo = txopVo->GetAifsn(); }
                if (txopBe) { cwMinBe = txopBe->GetMinCw(); cwMaxBe = txopBe->GetMaxCw(); aifsnBe = txopBe->GetAifsn(); }
            }
        }
        
        // Get metrics
        double throughputVo = 0, throughputBe = 0;
        double latencyVo = 0, latencyBe = 0;
        double deliveryVo = 0, deliveryBe = 0;
        double collisionVo = 0, collisionBe = 0;
        double queueOccVo = 0, queueOccBe = 0;
        double loadEstimate = 0;
        
        if (metrics)
        {
            const_cast<MacMetricsCollector*>(metrics)->UpdateMetrics(Simulator::Now());
            throughputVo = throughputBe = metrics->GetThroughput();
            latencyVo = latencyBe = metrics->GetCollisionRate() * 100.0;
            deliveryVo = deliveryBe = metrics->GetDeliveryRatio();
            collisionVo = collisionBe = metrics->GetCollisionRate();
            
            Ptr<WifiMac> mac = device->GetMac();
            if (mac)
            {
                Ptr<WifiMacQueue> queueVo = mac->GetTxopQueue(AC_VO);
                Ptr<WifiMacQueue> queueBe = mac->GetTxopQueue(AC_BE);
                queueOccVo = queueVo ? static_cast<double>(queueVo->GetNPackets()) / queueVo->GetMaxSize().GetValue() : 0;
                queueOccBe = queueBe ? static_cast<double>(queueBe->GetNPackets()) / queueBe->GetMaxSize().GetValue() : 0;
            }
            
            loadEstimate = metrics->GetQueueOccupancy() + metrics->GetThroughput() / 54.0;
        }
        
        if (perDeviceCsv)
        {
            auto writeIt = deviceFiles.find(filename);
            if (writeIt != deviceFiles.end() && writeIt->second.is_open())
            {
                writeIt->second << std::fixed << std::setprecision(6)
                           << timestamp << "," << nodeId << ","
                           << cwMinVo << "," << cwMaxVo << ","
                           << cwMinBe << "," << cwMaxBe << ","
                           << (int)aifsnVo << "," << (int)aifsnBe << ","
                           << throughputVo << "," << throughputBe << ","
                           << latencyVo << "," << latencyBe << ","
                           << deliveryVo << "," << deliveryBe << ","
                           << collisionVo << "," << collisionBe << ","
                           << queueOccVo << "," << queueOccBe << ","
                           << loadEstimate << "\n";
                writeIt->second.flush();
            }
        }
        
        // Live stream JSONL
        if (liveStream)
        {
            std::cout << "{"
                      << "\"type\":\"device\","
                      << "\"protocol\":\"" << protocol << "\","
                      << "\"t\":" << timestamp << ","
                      << "\"id\":" << nodeId << ","
                      << "\"cw_vo\":" << cwMinVo << ","
                      << "\"cw_be\":" << cwMinBe << ","
                      << "\"aifsn_vo\":" << (int)aifsnVo << ","
                      << "\"aifsn_be\":" << (int)aifsnBe << ","
                      << "\"throughput\":" << throughputVo << ","
                      << "\"collision\":" << collisionVo << ","
                      << "\"delivery\":" << deliveryVo << ","
                      << "\"queue_occ\":" << queueOccBe << ","
                      << "\"load_estimate\":" << loadEstimate
                      << "}\n";
            std::cout.flush();
        }
    };

    // Schedule per-device collection every 100ms
    // Use std::function for recursive lambda
    std::function<void(uint32_t, const MacMetricsCollector*, const LocalTd3MacController*, const Ptr<WifiNetDevice>&)> schedulePerDeviceCollection;
    
    schedulePerDeviceCollection = [&, simTime](uint32_t nodeId, const MacMetricsCollector* metrics,
                                       const LocalTd3MacController* sacCtrl,
                                       const Ptr<WifiNetDevice>& device)
    {
        // Hold the recurring body in a shared_ptr so the lambda can reschedule
        // itself safely — capturing itself by reference dangles once this
        // function's frame returns (this was the SIGBUS).
        auto body = std::make_shared<std::function<void()>>();
        *body = [body, nodeId, metrics, sacCtrl, device, simTime, &perDeviceCallback]() {
            double currentTime = Simulator::Now().GetSeconds();
            perDeviceCallback(nodeId, metrics, sacCtrl, device, currentTime);

            if (currentTime < simTime - 1.0) // Stop 1s before end
            {
                Simulator::Schedule(MilliSeconds(100), [body]() { (*body)(); });
            }
        };

        // First collection at 1.5s (after traffic starts)
        Simulator::Schedule(Seconds(1.5), [body]() { (*body)(); });
    };

    // Schedule per-device collection for each node
    // Store metrics collectors to keep them alive during simulation
    std::vector<std::unique_ptr<MacMetricsCollector>> metricsCollectors;
    
    for (uint32_t i = 0; i < nNodes; ++i)
    {
        Ptr<NetDevice> dev = staDevices.Get(i);
        Ptr<WifiNetDevice> wifiDev = DynamicCast<WifiNetDevice>(dev);
        if (!wifiDev) continue;
        
        Ptr<WifiMac> macPtr = wifiDev->GetMac();
        if (!macPtr) continue;
        
        // Create metrics collector for this node and store it
        auto metricsCollector = std::make_unique<MacMetricsCollector>();
        metricsCollector->Install(wifiDev);
        metricsCollectors.push_back(std::move(metricsCollector));
        
        // Get SAC controller if available
        const LocalTd3MacController* sacCtrl = nullptr;
        if (protocol == "SAC")
        {
            Ptr<Txop> txopVo = macPtr->GetTxopFor(AC_VO);
            Ptr<Txop> txopBe = macPtr->GetTxopFor(AC_BE);
            if (txopVo && txopBe)
            {
                // Find the SAC controller for this node
                if (i < g_td3Controllers.size())
                {
                    sacCtrl = g_td3Controllers[i].get();
                }
            }
        }
        
        schedulePerDeviceCollection(i, metricsCollectors.back().get(), sacCtrl, wifiDev);
    }

    // Run simulation
    Simulator::Stop(Seconds(simTime));
    Simulator::Run();

    // Collect results
    monitor->CheckForLostPackets();
    FlowMonitor::FlowStatsContainer stats = monitor->GetFlowStats();
    Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier>(flowmon.GetClassifier());

    // Aggregate totals
    double totalRxBytes = 0;
    double totalRxPackets = 0;
    double totalTxPackets = 0;
    double totalDelay = 0;

    // Per-class totals
    struct ClassMetrics {
        double rxBytes = 0;
        double rxPackets = 0;
        double txPackets = 0;
        double delaySum = 0;
    };
    ClassMetrics crit, be;

    for (auto& flow : stats)
    {
        auto t = classifier->FindFlow(flow.first);
        
        // Classify by destination port
        // Port 9 = critical (periodic sensor), Port 10 = best-effort (bursty alerts)
        bool isCritical = (t.destinationPort == 9);
        ClassMetrics& cm = isCritical ? crit : be;

        totalRxBytes += flow.second.rxBytes;
        totalRxPackets += flow.second.rxPackets;
        totalTxPackets += flow.second.txPackets;
        totalDelay += flow.second.delaySum.GetSeconds();

        cm.rxBytes += flow.second.rxBytes;
        cm.rxPackets += flow.second.rxPackets;
        cm.txPackets += flow.second.txPackets;
        cm.delaySum += flow.second.delaySum.GetSeconds();
    }

    auto compute = [&](const ClassMetrics& cm) -> std::tuple<double,double,double,double,double> {
        double thr = (cm.rxBytes * 8.0) / (simTime * 1e6) / nNodes;
        double dr = cm.txPackets > 0 ? cm.rxPackets / cm.txPackets : 0.0;
        double cr = cm.txPackets > 0 ? (cm.txPackets - cm.rxPackets) / cm.txPackets : 0.0;
        double lat = cm.rxPackets > 0 ? (cm.delaySum / cm.rxPackets) * 1000.0 : 0.0;
        double fair = 1.0 / (1.0 + cr);
        return {thr, dr, cr, lat, fair};
    };

    auto [crit_thr, crit_dr, crit_cr, crit_lat, crit_fair] = compute(crit);
    auto [be_thr, be_dr, be_cr, be_lat, be_fair] = compute(be);

    // Collect CW parameters from controllers
    double sum_cw_min = 0, sum_cw_max = 0, sum_aifsn = 0;
    double sum_cw_min_vo = 0, sum_cw_max_vo = 0, sum_cw_min_be = 0, sum_cw_max_be = 0;
    uint32_t cw_count = 0;
    if (protocol == "GDCF")
    {
        for (const auto& gdcf : g_gdcfManagers)
        {
            sum_cw_min += gdcf->GetCurrentCwMin();
            sum_cw_max += gdcf->GetCurrentCwMax();
            sum_aifsn += gdcf->GetCurrentAifsn();
            cw_count++;
        }
    }
    else if (protocol == "GDCF_ORIG")
    {
        for (const auto& gdcf : g_gdcfManagers)
        {
            sum_cw_min += gdcf->GetCurrentCwMin();
            sum_cw_max += gdcf->GetCurrentCwMax();
            sum_aifsn += gdcf->GetCurrentAifsn();
            cw_count++;
        }
    }
    else if (protocol == "SAC")
    {
        for (const auto& sac : g_td3Controllers)
        {
            sum_cw_min_vo += sac->GetCurrentCwMinVo();
            sum_cw_max_vo += sac->GetCurrentCwMaxVo();
            sum_cw_min_be += sac->GetCurrentCwMinBe();
            sum_cw_max_be += sac->GetCurrentCwMaxBe();
            sum_aifsn += sac->GetCurrentAifsnBe();
            cw_count++;
        }
        sum_cw_min = sum_cw_min_be;
        sum_cw_max = sum_cw_max_be;
    }
    else  // DCF / HEURISTIC_EDCA / any baseline - read directly from TXOPs
    {
        for (uint32_t i = 0; i < nNodes; ++i)
        {
            Ptr<NetDevice> dev = staDevices.Get(i);
            Ptr<WifiNetDevice> wifiDev = DynamicCast<WifiNetDevice>(dev);
            if (!wifiDev) continue;
            Ptr<WifiMac> macPtr = wifiDev->GetMac();
            if (!macPtr) continue;
            Ptr<Txop> txopVo = macPtr->GetTxopFor(AC_VO);
            Ptr<Txop> txopBe = macPtr->GetTxopFor(AC_BE);
            if (txopVo && txopBe)
            {
                sum_cw_min_vo += txopVo->GetMinCw();
                sum_cw_max_vo += txopVo->GetMaxCw();
                sum_cw_min_be += txopBe->GetMinCw();
                sum_cw_max_be += txopBe->GetMaxCw();
                sum_aifsn += txopBe->GetAifsn();
                cw_count++;
            }
        }
        sum_cw_min = sum_cw_min_be;
        sum_cw_max = sum_cw_max_be;
    }
    double avg_cw_min = cw_count > 0 ? sum_cw_min / cw_count : 0;
    double avg_cw_max = cw_count > 0 ? sum_cw_max / cw_count : 0;
    double avg_aifsn = cw_count > 0 ? sum_aifsn / cw_count : 0;
    double avg_cw_min_vo = cw_count > 0 ? sum_cw_min_vo / cw_count : 0;
    double avg_cw_max_vo = cw_count > 0 ? sum_cw_max_vo / cw_count : 0;
    double avg_cw_min_be = cw_count > 0 ? sum_cw_min_be / cw_count : 0;
    double avg_cw_max_be = cw_count > 0 ? sum_cw_max_be / cw_count : 0;

    SimulationResult result;
    result.protocol = protocol;
    result.numNodes = nNodes;
    result.offeredLoad = offeredLoad;
    result.throughput = (totalRxBytes * 8.0) / (simTime * 1e6) / nNodes;
    result.deliveryRatio = totalTxPackets > 0 ? totalRxPackets / totalTxPackets : 0.0;
    result.collisionRate = totalTxPackets > 0 ? (totalTxPackets - totalRxPackets) / totalTxPackets : 0.0;
    result.avgLatency = totalRxPackets > 0 ? (totalDelay / totalRxPackets) * 1000.0 : 0.0;
    result.fairness = 1.0 / (1.0 + result.collisionRate);
    result.crit_throughput = crit_thr;
    result.crit_deliveryRatio = crit_dr;
    result.crit_collisionRate = crit_cr;
    result.crit_avgLatency = crit_lat;
    result.crit_fairness = crit_fair;
    result.be_throughput = be_thr;
    result.be_deliveryRatio = be_dr;
    result.be_collisionRate = be_cr;
    result.be_avgLatency = be_lat;
    result.be_fairness = be_fair;
    result.seed = seed;
    result.avg_cw_min = avg_cw_min;
    result.avg_cw_max = avg_cw_max;
    result.avg_aifsn = avg_aifsn;
    result.avg_cw_min_vo = avg_cw_min_vo;
    result.avg_cw_max_vo = avg_cw_max_vo;
    result.avg_cw_min_be = avg_cw_min_be;
    result.avg_cw_max_be = avg_cw_max_be;

    g_results.push_back(result);

    NS_LOG_INFO("Result: Throughput=" << result.throughput << " Mbps/node, "
                << "PDR=" << result.deliveryRatio << ", Collision=" << result.collisionRate
                << ", Latency=" << result.avgLatency << " ms");

    Simulator::Destroy();
    
    g_gdcfManagers.clear();
    g_td3Controllers.clear();
}

int main(int argc, char* argv[])
{
    std::string outputFile = "results/ns3_mac_comparison_v2.csv";
    std::string sacWeightsPath = "/Users/karthik/MSME-hackathon-backup/results/models/sac_as_td3_weights.json";
    double simTime = 30.0;
    uint32_t seed = 1;
    uint32_t nSeeds = 1;
    bool useMsmeTraffic = true;
    std::string nodeCountsStr = "50,100,200";
    std::string offeredLoadsStr = "0.01,0.02,0.04,0.06,0.08";
    std::string protocolsStr = "DCF,GDCF,GDCF_ORIG,SAC";
    bool perDeviceCsv = false;
    bool liveStream = false;

    CommandLine cmd(__FILE__);
    cmd.AddValue("output", "Output CSV file", outputFile);
    cmd.AddValue("sacWeights", "Path to SAC weights JSON", sacWeightsPath);
    cmd.AddValue("simTime", "Simulation time (seconds)", simTime);
    cmd.AddValue("seed", "Base RNG seed", seed);
    cmd.AddValue("nSeeds", "Number of seeds per config", nSeeds);
    cmd.AddValue("msmeTraffic", "Use MSME traffic patterns", useMsmeTraffic);
    cmd.AddValue("nodeCounts", "Comma-separated node counts", nodeCountsStr);
    cmd.AddValue("offeredLoads", "Comma-separated offered loads", offeredLoadsStr);
    cmd.AddValue("protocols", "Comma-separated protocols", protocolsStr);
    cmd.AddValue("perDeviceCsv", "Enable per-device CSV export", perDeviceCsv);
    cmd.AddValue("liveStream", "Enable live JSONL streaming to stdout", liveStream);
    cmd.Parse(argc, argv);

    LogComponentEnable("MsmeMacComparisonV2", LOG_LEVEL_INFO);

    // Parse comma-separated values
    auto parseUint = [](const std::string& str) -> std::vector<uint32_t> {
        std::vector<uint32_t> result;
        std::stringstream ss(str);
        std::string token;
        while (std::getline(ss, token, ','))
        {
            result.push_back(static_cast<uint32_t>(std::stoi(token)));
        }
        return result;
    };
    
    auto parseDouble = [](const std::string& str) -> std::vector<double> {
        std::vector<double> result;
        std::stringstream ss(str);
        std::string token;
        while (std::getline(ss, token, ','))
        {
            result.push_back(std::stod(token));
        }
        return result;
    };
    
    auto parseString = [](const std::string& str) -> std::vector<std::string> {
        std::vector<std::string> result;
        std::stringstream ss(str);
        std::string token;
        while (std::getline(ss, token, ','))
        {
            result.push_back(token);
        }
        return result;
    };

    std::vector<uint32_t> nodeCounts = parseUint(nodeCountsStr);
    std::vector<double> offeredLoads = parseDouble(offeredLoadsStr);
    std::vector<std::string> protocols = parseString(protocolsStr);

    NS_LOG_INFO("Starting MSME MAC comparison v2 campaign...");
    NS_LOG_INFO("MSME Traffic: " << (useMsmeTraffic ? "ON" : "OFF"));
    NS_LOG_INFO("Node counts: " << nodeCountsStr);
    NS_LOG_INFO("Offered loads: " << offeredLoadsStr);
    NS_LOG_INFO("Protocols: " << protocolsStr);
    NS_LOG_INFO("Seeds per config: " << nSeeds);

    // For each seed, we need to re-run the full campaign with different RNG runs
    // The base seed determines the base, Run determines the variation
for (uint32_t s = 0; s < nSeeds; ++s)
        {
            uint32_t runSeed = seed + s;
            RngSeedManager::SetSeed(runSeed);
            RngSeedManager::SetRun(1);
            
            for (uint32_t nNodes : nodeCounts)
            {
                for (double load : offeredLoads)
                {
                    for (const std::string& protocol : protocols)
                    {
                        RunSimulation(protocol, nNodes, load, sacWeightsPath, simTime, useMsmeTraffic, runSeed, perDeviceCsv, liveStream);
                    }
                }
            }
        }

    // Write results to CSV
    std::ofstream csv(outputFile);
    csv << "protocol,num_nodes,offered_load,seed,throughput_mbps_per_node,delivery_ratio,collision_rate,avg_latency_ms,fairness,"
           "crit_throughput,crit_delivery_ratio,crit_collision_rate,crit_avg_latency_ms,crit_fairness,"
           "be_throughput,be_delivery_ratio,be_collision_rate,be_avg_latency_ms,be_fairness,"
           "avg_cw_min,avg_cw_max,avg_aifsn,"
           "avg_cw_min_vo,avg_cw_max_vo,avg_cw_min_be,avg_cw_max_be\n";
    
    for (const auto& r : g_results)
    {
        csv << r.protocol << "," << r.numNodes << "," << r.offeredLoad << "," << r.seed << ","
            << std::fixed << std::setprecision(4) << r.throughput << ","
            << r.deliveryRatio << "," << r.collisionRate << ","
            << r.avgLatency << "," << r.fairness << ","
            << r.crit_throughput << "," << r.crit_deliveryRatio << "," << r.crit_collisionRate << ","
            << r.crit_avgLatency << "," << r.crit_fairness << ","
            << r.be_throughput << "," << r.be_deliveryRatio << "," << r.be_collisionRate << ","
            << r.be_avgLatency << "," << r.be_fairness << ","
            << r.avg_cw_min << "," << r.avg_cw_max << "," << r.avg_aifsn << ","
            << r.avg_cw_min_vo << "," << r.avg_cw_max_vo << "," << r.avg_cw_min_be << "," << r.avg_cw_max_be << "\n";
    }
    csv.close();

    NS_LOG_INFO("Results written to " << outputFile);
    NS_LOG_INFO("Total simulations: " << g_results.size());

    return 0;
}