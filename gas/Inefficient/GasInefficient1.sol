function sumLoopInefficient() public view returns (uint256) {
    uint256 total = 0;
    for (uint i = 0; i < myArray.length; i++) {
        total += myArray[i];
    }
    return total;
}