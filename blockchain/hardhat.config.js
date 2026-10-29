require("@nomicfoundation/hardhat-ethers");

/** @type import('hardhat/config').HardhatConfig */
module.exports = {
  solidity: "0.8.24",
  networks: {
    // Hardhat local node (see "npm run node") — port 8545
    localhost: {
      url: "http://127.0.0.1:8545",
      chainId: 31337,
    },
    // Ganache GUI desktop app — default port 7545 (see README)
    ganacheGui: {
      url: "http://127.0.0.1:7545",
      chainId: 1337,
    },
  },
};
