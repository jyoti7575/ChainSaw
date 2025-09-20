// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "./OwnerOnly.sol";

contract Protected is OwnerOnly {
    string private secret;

    event SecretUpdated(string oldSecret, string newSecret);

    constructor(string memory initialSecret) {
        secret = initialSecret;
    }

    function setSecret(string memory newSecret) public onlyOwner {
        emit SecretUpdated(secret, newSecret);
        secret = newSecret;
    }

    function getSecret() public view onlyOwner returns (string memory) {
        return secret;
    }
}




