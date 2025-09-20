function sumLoopEfficient() public view returns (uint256) {
    uint256 total = 0;
    uint256 len = myArray.length;
    for (uint i = 0; i < len; i++) {
        total += myArray[i];
    }
    return total;
}