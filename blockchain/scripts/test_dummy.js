const hre = require("hardhat");

const DUMMY_UPDATES = [
  { round: 1, clientId: 0, updateHash: "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" },
  { round: 1, clientId: 1, updateHash: "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb" },
  { round: 2, clientId: 0, updateHash: "0xcccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc" },
  { round: 2, clientId: 3, updateHash: "0xdddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd" },
];

const DUMMY_REWARDS = [
  { round: 1, clientId: 0, score: 87 },
  { round: 1, clientId: 1, score: 92 },
  { round: 2, clientId: 0, score: 78 },
  { round: 2, clientId: 3, score: 95 },
];

function formatMs(ms) {
  return `${ms.toFixed(1)} ms`;
}

async function sendAndReport(label, txPromise) {
  const start = performance.now();
  const tx = await txPromise;
  const receipt = await tx.wait();
  const elapsed = performance.now() - start;

  console.log(`  ${label}`);
  console.log(`    tx hash : ${receipt.hash}`);
  console.log(`    gas used: ${receipt.gasUsed.toString()}`);
  console.log(`    time    : ${formatMs(elapsed)}`);
  console.log("");

  return receipt;
}

async function main() {
  const networkName = hre.network.name;
  console.log("=== ContributionLedger dummy test ===");
  console.log(`Network: ${networkName}\n`);

  const deployStart = performance.now();
  const ContributionLedger = await hre.ethers.getContractFactory("ContributionLedger");
  const ledger = await ContributionLedger.deploy();
  await ledger.waitForDeployment();
  const contractAddress = await ledger.getAddress();
  const deployElapsed = performance.now() - deployStart;

  console.log("Deployed ContributionLedger to:", contractAddress);
  console.log(`Deployment time: ${formatMs(deployElapsed)}\n`);

  console.log("--- Logging dummy FL update hashes ---");
  for (const row of DUMMY_UPDATES) {
    await sendAndReport(
      `logUpdate(round=${row.round}, clientId=${row.clientId})`,
      ledger.logUpdate(row.round, row.clientId, row.updateHash)
    );
  }

  console.log("--- Recording dummy reward scores ---");
  for (const row of DUMMY_REWARDS) {
    await sendAndReport(
      `recordReward(round=${row.round}, clientId=${row.clientId}, score=${row.score})`,
      ledger.recordReward(row.round, row.clientId, row.score)
    );
  }

  console.log("--- getClientHistory() results ---");
  const clientsToInspect = [0, 1, 3];
  for (const clientId of clientsToInspect) {
    const [updateRounds, updateHashes, rewardRounds, rewardScores] =
      await ledger.getClientHistory(clientId);

    console.log(`Client ${clientId}:`);
    console.log("  Updates:");
    if (updateRounds.length === 0) {
      console.log("    (none)");
    } else {
      for (let i = 0; i < updateRounds.length; i++) {
        console.log(
          `    round ${updateRounds[i]}  hash ${updateHashes[i]}`
        );
      }
    }

    console.log("  Rewards:");
    if (rewardRounds.length === 0) {
      console.log("    (none)");
    } else {
      for (let i = 0; i < rewardRounds.length; i++) {
        console.log(
          `    round ${rewardRounds[i]}  score ${rewardScores[i]}`
        );
      }
    }
    console.log("");
  }

  console.log("=== Dummy test complete ===");
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
