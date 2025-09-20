uint256[10] private values;
function addValue(uint256 _value, uint _index) public {
    values[_index] = _value;
}