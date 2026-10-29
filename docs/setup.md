# Installation and Setup Guide

This guide provides step-by-step instructions for configuring the local environment, installing dependencies, starting the local blockchain test network, and launching the application.

---

## Prerequisites

| Requirement | Supported Version | Notes |
|-------------|-------------------|-------|
| **Python**  | 3.10 – 3.14        | Core FL algorithms, PyTorch, Flower, Opacus, Streamlit |
| **Node.js** | 18+ (LTS)          | Hardhat Ethereum local test node and contract deployment |
| **npm**     | 9+                 | Package manager for Node dependencies |
| **Git**     | Any modern version | Version control |
| **CUDA** *(Optional)* | 11.8+ / 12.x | GPU acceleration for PyTorch (CPU fallback supported automatically) |

---

## 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/Project_SW.git
cd Project_SW
```

---

## 2. Python Environment Setup

It is strongly recommended to use a clean virtual environment:

### Windows (PowerShell)
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Linux / macOS (Bash)
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

> **PyTorch Acceleration Note:**  
> If using an NVIDIA GPU, install the CUDA-enabled PyTorch build for your system from [pytorch.org](https://pytorch.org/get-started/locally/) prior to running `pip install -r requirements.txt`. All scripts automatically detect CUDA availability and fall back to CPU if no GPU is found.

---

## 3. Blockchain Setup (Hardhat Local Node)

The project includes an Ethereum smart contract ([`ContributionLedger.sol`](file:///c:/Users/ADMIN/Desktop/Project_SW/blockchain/contracts/ContributionLedger.sol)) that records model weight update hashes and NACS reward scores.

### Step 3.1: Install Node Dependencies
You can install dependencies directly from the root (using npm workspaces / script delegation) or inside the `blockchain/` folder:

```bash
cd blockchain
npm install
cd ..
```

### Step 3.2: Start the Local Ethereum Test Node
Open a **dedicated terminal window** and run:

```bash
# From project root
npm run node

# Or directly from the blockchain directory
cd blockchain
npx hardhat node --port 8545
```

You should see:
```text
Started HTTP and WebSocket JSON-RPC server at http://127.0.0.1:8545/
Account #0: 0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266 (10000 ETH)
...
```

> **Keep this terminal running in the background** while conducting experiments or interacting with the UI.

### Step 3.3: Compile and Deploy the Smart Contract
In a **second terminal window** (with your Python virtual environment activated):

```bash
# Compile Solidity contracts
npm run compile

# Deploy ContributionLedger to the local node
npm run deploy
```

Upon successful deployment, the script outputs the contract address and writes it to [`blockchain/deployed_address.json`](file:///c:/Users/ADMIN/Desktop/Project_SW/blockchain/deployed_address.json):
```text
Deploying ContributionLedger to network: localhost
ContributionLedger deployed to: 0x5FbDB2315678afecb367f032d93F642f64180aa3
Saved deployment address to .../blockchain/deployed_address.json
```

---

## 4. Dataset Acquisition

The framework uses three datasets across different benchmarking phases:

### 1. PneumoniaMNIST (MedMNIST v3)
- **Automatic:** Downloaded automatically by `medmnist` into `./data/` on the first run of Phase 1, Phase 2, Phase 3, or Phase 4.
- **Fallback:** If an offline environment is detected, [`src/dataset.py`](file:///c:/Users/ADMIN/Desktop/Project_SW/src/dataset.py) automatically falls back to an internal synthetic medical dataset to allow syntax and pipeline validation without network access.

### 2. UCI Human Activity Recognition (UCI HAR)
- **Automatic:** [`src/dataset_har.py`](file:///c:/Users/ADMIN/Desktop/Project_SW/src/dataset_har.py) will attempt to download and extract the dataset from the official UCI repository (`https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip`) into `./data/`.
- **Manual Download (Alternative):**
  1. Download the zip archive from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones).
  2. Place `uci_har.zip` or extract the folder `UCI HAR Dataset` into `./data/`.

### 3. NIH Chest X-ray Dataset (Kaggle / NIH Clinical Center)
- **Manual Acquisition:**
  1. Download the NIH Chest X-ray dataset metadata (`Data_Entry_2017.csv`) and image archives from [Kaggle: NIH Chest X-rays](https://www.kaggle.com/datasets/nih-chest-xrays/data).
  2. Extract images into subfolders `images_001` through `images_012` alongside `Data_Entry_2017.csv` in an `archive/` or `archive (1)/` directory in the repository root.
  3. Preprocessing and patient-level splitting are automatically handled by [`src/dataset_nih.py`](file:///c:/Users/ADMIN/Desktop/Project_SW/src/dataset_nih.py), which caches resized 64x64 arrays to `./data/nih_cache/` for accelerated repeat experimentation.

---

## 5. Verification Commands

Run a quick pipeline sanity check before executing full training loops:

```bash
# 1. Verify contract interactions via dummy test script
npm run test:dummy

# 2. Verify Python Web3 and FL pipeline integration (1 client, 1 round)
python run_phase3_connected.py --sanity
```

---

## Troubleshooting

| Problem | Cause | Resolution |
|---------|-------|------------|
| `ConnectionError: Could not connect to Hardhat Ethereum node at http://127.0.0.1:8545` | Hardhat node is not running. | Run `npm run node` in a separate terminal. |
| `FileNotFoundError: Contract ABI artifact not found` | Contracts were not compiled. | Run `npm run compile`. |
| `FileNotFoundError: Deployed address file not found` | Contract was not deployed. | Run `npm run deploy` with the node running. |
| `ModuleNotFoundError: No module named 'src'` | PYTHONPATH is not set or script run from wrong directory. | Always execute scripts from the project root (`Project_SW/`). |
| `CUDA out of memory` | High batch size on limited GPU VRAM. | Scripts default to batch size 32. Adjust batch size in runner arguments or set `CUDA_VISIBLE_DEVICES=""` to use CPU. |
| `npm audit warnings` | Upstream Hardhat 2 development tool advisories. | Expected for Hardhat 2 local test tooling. Do **not** run `npm audit fix --force` as it can introduce breaking changes to Hardhat. |
