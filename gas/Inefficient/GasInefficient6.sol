function checkAdminInefficient(address _addr) public view returns(bool) {
    for (uint i = 0; i < admins.length; i++) {
        if (admins[i] == _addr) return true;
    }
    return false;
}