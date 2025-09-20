mapping(address => bool) private isAdmin;
function checkAdminEfficient(address _addr) public view returns(bool) {
    return isAdmin[_addr];
}