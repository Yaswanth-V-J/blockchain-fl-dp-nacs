# Phase 3 — Blockchain Contribution Ledger

On-chain audit log for federated-learning update hashes and per-client reward scores. This folder is standalone: it does **not** connect to the Python FL pipeline yet.

## Prerequisites

- Node.js 18+ and npm
- Blockchain commands can be run from `Project_SW/` (project root) **or** from `Project_SW/blockchain/`

## One-time setup

```bash
cd blockchain
npm install
cd ..
```

## Network: Hardhat local node on port **8545**

Hardhat is configured to use **`http://127.0.0.1:8545`** via the built-in Hardhat node (`npm run node`).

If you prefer the **Ganache GUI** desktop app instead, start it and use the `ganacheGui` network (port **7545**) by adding `--network ganacheGui` to deploy/test commands below.

---

## Step-by-step (clean terminal)

### 1. Start the local blockchain node

**Option A — Hardhat node (recommended, port 8545):**

Open a terminal and leave it running:

```bash
# From Project_SW/ (project root)
npm run node

# Or from blockchain/
cd blockchain
npm run node
```

You should see `Started HTTP and WebSocket JSON-RPC server at http://127.0.0.1:8545/`.

**Option B — Ganache GUI (port 7545):**

1. Open the Ganache desktop app.
2. Create or open a workspace (Quickstart is fine).
3. Confirm the RPC server shows `HTTP://127.0.0.1:7545`.
4. Use `--network ganacheGui` in steps 3–4 below.

### 2. Compile the contract

In a **second** terminal:

```bash
cd blockchain
npx hardhat compile
```

Expected: `Compiled 1 Solidity file successfully`.

### 3. Deploy to the local network

**Hardhat node (8545):**

```bash
npx hardhat run scripts/deploy.js --network localhost
```

**Ganache GUI (7545):**

```bash
npx hardhat run scripts/deploy.js --network ganacheGui
```

Expected output includes a deployed contract address, e.g. `ContributionLedger deployed to: 0x...`

### 4. Run the dummy test (deploy + log + rewards + history)

**Hardhat node (8545):**

```bash
npx hardhat run scripts/test_dummy.js --network localhost
```

**Ganache GUI (7545):**

```bash
npx hardhat run scripts/test_dummy.js --network ganacheGui
```

This script:

- Deploys `ContributionLedger`
- Calls `logUpdate()` and `recordReward()` with 4 rows of dummy data
- Calls `getClientHistory()` for clients 0, 1, and 3
- Prints transaction hashes, gas used, and elapsed time per transaction

---

## npm scripts (shortcuts)

| Command | Description |
|---------|-------------|
| `npm run node` | Start Hardhat local node on port **8545** |
| `npm run compile` | Compile Solidity contracts |
| `npm run deploy` | Deploy to localhost (8545) |
| `npm run test:dummy` | Run full dummy test on localhost (8545) |

---

## Contract API

| Function | Description |
|----------|-------------|
| `logUpdate(uint256 round, uint256 clientId, bytes32 updateHash)` | Store an FL update hash; emits `UpdateLogged` |
| `recordReward(uint256 round, uint256 clientId, uint256 score)` | Store a contribution score; emits `RewardRecorded` |
| `getClientHistory(uint256 clientId)` | View all updates and rewards for one client |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ECONNREFUSED 127.0.0.1:8545` | Start the node: `npm run node` |
| `ECONNREFUSED 127.0.0.1:7545` | Start Ganache GUI, or switch to `--network localhost` |
| `Insufficient funds` | Restart the local node for fresh accounts |
| Compile errors | Run `npx hardhat clean` then `npx hardhat compile` |
| npm audit warnings | Expected for Hardhat 2; they are in local-only tooling. Do **not** run `npm audit fix --force` (breaks Hardhat). Install is configured with `audit-level=critical` so non-critical advisories stay quiet. |
