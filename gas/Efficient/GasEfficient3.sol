function loopUpdatesEfficient() public {
    uint256 temp = total;
    for (uint i = 0; i < 10; i++) {
        temp++;
    }
    total = temp;
}