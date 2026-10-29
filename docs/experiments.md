# Experimental Methodology, Evaluation, and Empirical Results

This document provides a comprehensive record of the experimental phases, benchmarking methodology, hyperparameter configurations, and empirical findings. All metric values presented below are directly verified from project experimental logs and serialized JSON outputs in [`results/`](file:///c:/Users/ADMIN/Desktop/Project_SW/results/).

---

## 1. Experimental Overview

The research evaluation is organized into six progressive phases across three distinct domains:

| Phase | Title | Dataset | Key Focus |
|---|---|---|---|
| **Phase 1** | Baseline Federated Learning | MedMNIST PneumoniaMNIST | Standard FedAvg over Non-IID Dirichlet partitions ($\alpha=0.5$). |
| **Phase 2** | Differential Privacy Analysis | MedMNIST PneumoniaMNIST | Empirical privacy-utility trade-off across $\epsilon \in \{\text{None}, 20.0, 8.0, 3.0, 1.0\}$. |
| **Phase 3** | Blockchain-Connected FL | MedMNIST PneumoniaMNIST | Integration with local Ethereum smart contract, SHA-256 parameter hashing, and gas auditing. |
| **Phase 4** | NACS Contribution Scoring & Multi-Seed Benchmark | MedMNIST PneumoniaMNIST | 4-configuration ablation study and multi-seed variance stabilization (5 random seeds). |
| **Phase 5** | Cross-Domain Generalizability | UCI HAR (Smartphone Sensor Data) | Wearable tabular domain validation with real subject-ID partitioning (5 random seeds). |
| **Phase 6** | Multi-Label Clinical Pathology Benchmark | NIH Chest X-ray | High-resolution multi-label thoracic pathology classification with patient-level splitting (10 random seeds). |

---

## 2. Experimental Phases & Verified Results

### Phase 1: Plain FL Baseline (FedAvg)
- **Dataset:** PneumoniaMNIST (28x28 grayscale chest radiographs, binary: Normal vs. Pneumonia)
- **Partitioning:** 5 simulated hospital clients partitioned via Dirichlet distribution ($\alpha=0.5$, random seed 42)
- **Model:** `MedicalCNN` (2 conv blocks with GroupNorm and MaxPool2d, 1 linear output layer)
- **Federated Hyperparameters:** 10 communication rounds, local epochs = 2, learning rate = 0.005, batch size = 32, optimizer = Adam
- **Central Test Set:** 624 test samples

#### Verified Metric Trajectory ([`results/phase1_baseline.json`](file:///c:/Users/ADMIN/Desktop/Project_SW/results/phase1_baseline.json))
| Round | Global Accuracy | Macro F1-Score | Precision | Recall | Test Loss |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **0 (Init)** | 0.6266 (62.66%) | 0.4009 | 0.5993 | 0.5047 | 0.6916 |
| **1** | 0.6394 (63.94%) | 0.4251 | 0.8171 | 0.5192 | 0.5750 |
| **2** | 0.8718 (87.18%) | 0.8561 | 0.8854 | 0.8419 | 0.3391 |
| **3** | 0.8269 (82.69%) | 0.7919 | 0.8790 | 0.7726 | 0.4558 |
| **4** | 0.8205 (82.05%) | 0.7828 | 0.8753 | 0.7641 | 0.5437 |
| **5** | 0.8301 (83.01%) | 0.7964 | 0.8808 | 0.7769 | 0.5229 |
| **6** | 0.8381 (83.81%) | 0.8081 | 0.8827 | 0.7885 | 0.6557 |
| **7** | 0.8574 (85.74%) | 0.8333 | 0.8967 | 0.8132 | 0.4739 |
| **8** | 0.8221 (82.21%) | 0.7837 | 0.8825 | 0.7645 | 0.7746 |
| **9** | 0.8365 (83.65%) | 0.8041 | 0.8902 | 0.7838 | 0.6598 |
| **10 (Final)** | **0.8478 (84.78%)** | **0.8205** | **0.8910** | **0.8004** | **0.7538** |

---

### Phase 2: Differential Privacy Epsilon Analysis (DP-SGD)
- **Engine:** Opacus `PrivacyEngine` calibrated across `total_rounds * local_epochs` (total 20 epochs)
- **Fixed Privacy Parameter:** Target $\delta = 10^{-5}$, max gradient clipping norm $C = 1.0$
- **Rényi DP Accountant:** Tracks cumulative monotonic privacy consumption across rounds.

#### Verified Privacy vs. Utility Trade-Off ([`results/phase2_dp_*.json`](file:///c:/Users/ADMIN/Desktop/Project_SW/results/))
| Target Budget $\epsilon$ | Privacy Guarantee | Actual Spent $\epsilon$ | Final Accuracy | Final F1-Score | Final Test Loss |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **None (No-DP)** | Baseline (No guarantee) | 0.0000 | **0.8558 (85.58%)** | **0.8340** | 1.0033 |
| **20.0** | Weak Privacy | 19.9996 | 0.8317 (83.17%) | 0.7999 | 1.1916 |
| **8.0** | Moderate Privacy | 7.9990 | 0.8029 (80.29%) | 0.7579 | 1.4385 |
| **3.0** | Strong Privacy | 2.9968 | 0.7644 (76.44%) | 0.6966 | 1.7273 |
| **1.0** | Very Strong Privacy | 0.9994 | 0.7596 (75.96%) | 0.6873 | 1.2209 |

> **Empirical Observation:**  
> The empirical results rigorously confirm the core differential privacy theorem: lower target $\epsilon$ (stronger privacy guarantee) requires injecting higher variance Gaussian perturbation $\sigma$ into the client gradients, resulting in a controlled utility trade-off (accuracy transitions from 85.58% down to 75.96%).

---

### Phase 3: Blockchain-Connected Audit Logging
- **Smart Contract:** [`ContributionLedger.sol`](file:///c:/Users/ADMIN/Desktop/Project_SW/blockchain/contracts/ContributionLedger.sol) deployed on local Hardhat node (`0x5FbDB2315678afecb367f032d93F642f64180aa3`)
- **Weight Fingerprint:** Serialized model weights (`pickle.dumps`) are hashed into raw 32-byte SHA-256 digests (`bytes32`) before transmission to `logUpdate()`.
- **Verified Gas Metrics:**
  - `logUpdate()`: ~46,000 to ~48,500 gas units per client update.
  - `recordReward()`: ~44,000 to ~46,200 gas units per contribution recording.
  - Verification: 100% of on-chain logged hashes retrieved via `getClientHistory()` match off-chain computed hashes.

---

### Phase 4: NACS Ablation Study & Multi-Seed Robustness (MedMNIST)
- **Experimental Setup:** Heterogeneous client privacy budgets:
  - Client 0: $\epsilon=1.0$ (High noise $\sigma$)
  - Client 1: $\epsilon=3.0$
  - Client 2: $\epsilon=8.0$
  - Client 3: $\epsilon=15.0$
  - Client 4: $\epsilon=20.0$ (Low noise $\sigma$)
- **Seeds Evaluated:** 5 distinct random seeds (`42, 43, 44, 45, 46`).

#### 4-Configuration Ablation Study Findings ([`results/ablation_study_comparison.png`](file:///c:/Users/ADMIN/Desktop/Project_SW/results/ablation_study_comparison.png))
1. **Config A (`size_only`):** Rewards solely by data quantity ($N_i / N_{\text{total}}$). Clients with large partitions dominate rankings even when submitting noisy or inferior updates.
2. **Config B (`size_quality`):** Combines data quantity and empirical loss improvement ($s_{\text{size}} + s_{\text{qual}})/2$. Highly unstable round-to-round for high-privacy clients due to DP noise fluctuations.
3. **Config C (`size_quality_noise`):** Scales raw combined score by trust weight $\alpha = 1 / (1 + \sigma)$.
4. **Config D (`full_nacs`):** Couples the trust weight with temporal Exponential Moving Average (EMA). Eliminates noise spikes while rewarding sustained model contributions.
5. **Ranking Adjustment:** In ablation evaluations, client rankings actively shift between Config A and Config D, verifying that NACS prevents clients from winning rewards purely through data volume.

#### Verified Multi-Seed Variance Reduction ([`results/phase4_nacs_multiseed.json`](file:///c:/Users/ADMIN/Desktop/Project_SW/results/phase4_nacs_multiseed.json))
| Client Index | Target Privacy $\epsilon$ | Baseline Variance | NACS Variance | Mean Variance Reduction (%) | Std Dev (%) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **Client 0** | 1.0 (Highest Noise) | Unsmoothed baseline | Smoothed EMA | **86.49%** | $\pm$ 0.60% |
| **Client 1** | 3.0 | Unsmoothed baseline | Smoothed EMA | **93.29%** | $\pm$ 1.18% |
| **Client 2** | 8.0 | Unsmoothed baseline | Smoothed EMA | **74.21%** | $\pm$ 1.17% |
| **Client 3** | 15.0 | Unsmoothed baseline | Smoothed EMA | **81.07%** | $\pm$ 2.77% |
| **Client 4** | 20.0 (Lowest Noise) | Unsmoothed baseline | Smoothed EMA | **82.89%** | $\pm$ 2.83% |
| **Mean Across All Clients** | — | — | — | **83.59%** | — |

---

### Phase 5: Cross-Domain Generalizability Benchmark (UCI HAR)
- **Domain:** Wearable mobile sensor data (561 pre-extracted continuous inertial sensor features, 6 human physical activity classes).
- **Partitioning:** Realistic real-world partition: split strictly by **Subject ID** across 5 heterogeneous privacy clients ($\epsilon \in [1.0, 3.0, 8.0, 15.0, 20.0]$).
- **Model:** `HARClassifier` (3-layer Multi-Layer Perceptron: 561 $\to$ 256 $\to$ 128 $\to$ 6 with Dropout).
- **Seeds Evaluated:** 5 seeds (`42, 43, 44, 45, 46`).

#### Verified Multi-Seed Variance Reduction ([`results/phase5_har_multiseed.json`](file:///c:/Users/ADMIN/Desktop/Project_SW/results/phase5_har_multiseed.json))
| Client Index | Target Privacy $\epsilon$ | Mean Variance Reduction (%) | Std Dev (%) | Per-Seed Values (%) |
|:---:|:---:|:---:|:---:|---|
| **Client 0** | 1.0 | **92.51%** | $\pm$ 2.98% | [94.38, 94.78, 94.86, 91.41, 87.13] |
| **Client 1** | 3.0 | **90.13%** | $\pm$ 4.31% | [90.67, 91.62, 88.25, 96.64, 83.48] |
| **Client 2** | 8.0 | **87.34%** | $\pm$ 5.52% | [79.48, 95.44, 90.60, 87.63, 83.56] |
| **Client 3** | 15.0 | **81.98%** | $\pm$ 3.05% | [84.68, 77.93, 86.09, 79.55, 81.67] |
| **Client 4** | 20.0 | **81.46%** | $\pm$ 3.17% | [79.51, 83.09, 83.64, 84.85, 76.22] |
| **Mean Across All Clients** | — | **86.68%** | — | — |

---

### Phase 6: Multi-Label Clinical Pathology Benchmark (NIH Chest X-ray)
- **Domain:** Multi-label thoracic pathology detection on high-resolution hospital radiograph archives.
- **Target Pathologies (6 classes):** Infiltration, Effusion, Atelectasis, Nodule, Mass, Pneumothorax (+ 'No Finding' encoded as zero vector).
- **Partitioning:** Partitioned strictly by **Patient ID** to ensure zero data leakage between client training subsets and central test partitions.
- **Model:** `NIHChestCNN` (4-block convolutional network with GroupNorm for Opacus DP compatibility, BCEWithLogitsLoss).
- **Seeds Evaluated:** 10 random seeds (`42` through `51`).

#### Verified Multi-Seed Variance Reduction ([`results/phase6_nih_multiseed.json`](file:///c:/Users/ADMIN/Desktop/Project_SW/results/phase6_nih_multiseed.json))
| Client Index | Target Privacy $\epsilon$ | Baseline Var (Mean $\pm$ Std) | NACS Var (Mean $\pm$ Std) | Mean Variance Reduction (%) | Std Dev (%) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **Client 0** | 1.0 | $120.26 \pm 26.61$ | $23.79 \pm 5.53$ | **80.24%** | $\pm$ 1.19% |
| **Client 1** | 3.0 | $134.98 \pm 25.16$ | $35.44 \pm 6.76$ | **73.76%** | $\pm$ 1.14% |
| **Client 2** | 8.0 | $152.91 \pm 28.14$ | $40.92 \pm 8.36$ | **73.34%** | $\pm$ 1.28% |
| **Client 3** | 15.0 | $148.62 \pm 36.65$ | $38.89 \pm 9.61$ | **73.82%** | $\pm$ 0.72% |
| **Client 4** | 20.0 | $155.05 \pm 30.72$ | $40.16 \pm 6.85$ | **73.90%** | $\pm$ 1.53% |
| **Mean Across All Clients** | — | — | — | **75.01%** | — |

---

## 3. Cross-Domain Variance Stabilization Summary

Across all three experimental modalities, NACS consistently reduces contribution scoring volatility caused by differential privacy noise:

| Dataset Domain | Modality | Sample Format | Mean Scoring Variance Reduction |
|---|---|---|:---:|
| **MedMNIST PneumoniaMNIST** | Medical Computer Vision | 28x28 Grayscale Radiographs | **83.59%** |
| **UCI HAR** | Wearable IoT Sensing | 561 Continuous Inertial Features | **86.68%** |
| **NIH Chest X-ray** | Clinical Hospital Records | 64x64 Grayscale Radiographs (Multi-Label) | **75.01%** |

---

## 4. How to Reproduce Experiments

Ensure your local blockchain node is running (`npm run node` & `npm run deploy`) before launching experiments.

```bash
# Run Phase 1: Baseline FL
python run_phase1.py

# Run Phase 2: Differential Privacy Comparison
python run_phase2_dp.py

# Run Phase 3: Connected FL + Blockchain
python run_phase3_connected.py

# Run Phase 4: Single-Seed NACS Experiment
python run_phase4_nacs.py

# Run Phase 4: Multi-Seed Robustness (Seeds 42..46)
python run_phase4_multiseed.py

# Run 4-Configuration Ablation Analysis
python analyze_ablation.py

# Run Phase 5: Cross-Domain UCI HAR Benchmark
python run_phase5_har.py

# Run Phase 6: NIH Chest X-ray Multi-Label Benchmark
python run_phase6_nih.py

# Run Full Sequential Pipeline (Phases 1 through 6)
python run_all.py
```
