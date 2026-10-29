const hre = require("hardhat");
const fs = require("fs");
const path = require("path");

async function main() {
  const networkName = hre.network.name;
  console.log(`Deploying ContributionLedger to network: ${networkName}`);

  const ContributionLedger = await hre.ethers.getContractFactory("ContributionLedger");
  const ledger = await ContributionLedger.deploy();

  await ledger.waitForDeployment();
  const address = await ledger.getAddress();

  console.log("ContributionLedger deployed to:", address);

  const deploymentData = {
    address: address,
    network: networkName,
    deployedAt: new Date().toISOString()
  };

  const outputPath = path.join(__dirname, "..", "deployed_address.json");
  fs.writeFileSync(outputPath, JSON.stringify(deploymentData, null, 2));
  console.log(`Saved deployment address to ${outputPath}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});

