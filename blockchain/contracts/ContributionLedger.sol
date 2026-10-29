// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// @title ContributionLedger
/// @notice On-chain audit log for federated-learning update hashes and per-client reward scores.
contract ContributionLedger {
    struct UpdateEntry {
        uint256 round;
        bytes32 updateHash;
    }

    struct RewardEntry {
        uint256 round;
        uint256 score;
    }

    mapping(uint256 => UpdateEntry[]) private clientUpdates;
    mapping(uint256 => RewardEntry[]) private clientRewards;

    event UpdateLogged(uint256 indexed round, uint256 indexed clientId, bytes32 updateHash);
    event RewardRecorded(uint256 indexed round, uint256 indexed clientId, uint256 score);

    function logUpdate(uint256 round, uint256 clientId, bytes32 updateHash) external {
        clientUpdates[clientId].push(UpdateEntry(round, updateHash));
        emit UpdateLogged(round, clientId, updateHash);
    }

    function recordReward(uint256 round, uint256 clientId, uint256 score) external {
        clientRewards[clientId].push(RewardEntry(round, score));
        emit RewardRecorded(round, clientId, score);
    }

    function getClientHistory(uint256 clientId)
        external
        view
        returns (
            uint256[] memory updateRounds,
            bytes32[] memory updateHashes,
            uint256[] memory rewardRounds,
            uint256[] memory rewardScores
        )
    {
        UpdateEntry[] storage updates = clientUpdates[clientId];
        RewardEntry[] storage rewards = clientRewards[clientId];

        updateRounds = new uint256[](updates.length);
        updateHashes = new bytes32[](updates.length);
        for (uint256 i = 0; i < updates.length; i++) {
            updateRounds[i] = updates[i].round;
            updateHashes[i] = updates[i].updateHash;
        }

        rewardRounds = new uint256[](rewards.length);
        rewardScores = new uint256[](rewards.length);
        for (uint256 i = 0; i < rewards.length; i++) {
            rewardRounds[i] = rewards[i].round;
            rewardScores[i] = rewards[i].score;
        }
    }
}
