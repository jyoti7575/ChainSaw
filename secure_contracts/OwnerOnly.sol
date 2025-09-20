// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title OwnerOnly
 * @dev Provides a basic access control mechanism, where
 * there is an account (an owner) that can be granted exclusive access to
 * specific functions. This version is revised for safety and best practices.
 */
contract OwnerOnly {
    address private _owner;

    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    constructor() {
        _owner = msg.sender;
        emit OwnershipTransferred(address(0), _owner);
    }

    /**
     * @dev Throws if called by any account other than the owner.
     */
    modifier onlyOwner() {
        require(msg.sender == _owner, "OwnerOnly: caller is not the owner");
        _;
    }

    /**
     * @dev Returns the address of the current owner.
     */
    function owner() public view returns (address) {
        return _owner;
    }

    /**
     * @dev Transfers ownership of the contract to a new account (`newOwner`).
     * Can only be called by the current owner.
     */
    function transferOwnership(address newOwner) public onlyOwner {
        require(newOwner != address(0), "OwnerOnly: new owner is the zero address");
        
        // Best practice: Change state first (Effects), then emit the event.
        address oldOwner = _owner;
        _owner = newOwner;
        emit OwnershipTransferred(oldOwner, newOwner);
    }

    // The renounceOwnership function has been removed. It is a dangerous function
    // that allows for the permanent and irreversible locking of the contract's
    // administrative functions by setting the owner to address(0).
}