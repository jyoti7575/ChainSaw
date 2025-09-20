import os
import pickle
import json
import base64
import traceback
from flask import Flask, request, jsonify, render_template
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad

# --- Configuration ---
# The single source of truth for the key
SECRET_KEY = b'my32charactersecretkeyneedstobe1'
BLOCK_SIZE = 16

# --- All other functions (load_models, load_solution_data) remain the same ---
# ... (omitted for brevity, no changes needed to them) ...
def load_models():
    models = {}
    try:
        with open('security_model.pkl', 'rb') as f: models['security'] = pickle.load(f)
        with open('security_vectorizer.pkl', 'rb') as f: models['security_vectorizer'] = pickle.load(f)
        with open('gas_model.pkl', 'rb') as f: models['gas'] = pickle.load(f)
        with open('gas_vectorizer.pkl', 'rb') as f: models['gas_vectorizer'] = pickle.load(f)
        print("✅ All models loaded successfully!")
        return models
    except FileNotFoundError as e:
        print(f"Model load error: {e}")
        return None
models = load_models()
if not models: raise RuntimeError("Models could not be loaded.")
def load_solution_data():
    security_solutions = {}
    gas_solutions = {}
    security_file = 'smartbugs-curated-main/vulnerabilities.json'
    gas_file = 'gas/solution.json'
    if os.path.exists(security_file):
        with open(security_file, 'r') as f:
            security_solutions_list = json.load(f)
            if isinstance(security_solutions_list, list): security_solutions = {item['name']: item for item in security_solutions_list}
    if os.path.exists(gas_file):
        with open(gas_file, 'r') as f: gas_solutions = json.load(f)
    return security_solutions, gas_solutions
security_solutions, gas_solutions = load_solution_data()
# --- End of unchanged functions ---

app = Flask(__name__)

# --- THIS IS THE MODIFIED FUNCTION ---
@app.route("/")
def index():
    # We decode the key to a regular string to pass it to the HTML template
    key_for_template = SECRET_KEY.decode('utf-8')
    return render_template("index.html", secret_key=key_for_template)

# --- The decrypt_data and scan_contract functions remain the same ---
# ... (omitted for brevity, no changes needed to them) ...
def decrypt_data(encrypted_data: str) -> str:
    decoded_data = base64.b64decode(encrypted_data)
    cipher = AES.new(SECRET_KEY, AES.MODE_ECB)
    decrypted_padded_data = cipher.decrypt(decoded_data)
    decrypted_data = unpad(decrypted_padded_data, BLOCK_SIZE)
    return decrypted_data.decode('utf-8')

@app.route("/scan", methods=["POST"])
def scan_contract():
    try:
        encrypted_code = request.json['code']
        solidity_code = decrypt_data(encrypted_code)
        sec_vec = models['security_vectorizer'].transform([solidity_code])
        security_prediction = models['security'].predict(sec_vec)[0]
        gas_vec = models['gas_vectorizer'].transform([solidity_code])
        gas_prediction = models['gas'].predict(gas_vec)[0]
        report = { "security": { "status": "Secure" if security_prediction == 0 else "Vulnerable", "vulnerabilities": [] }, "gas_optimization": { "status": "Efficient" if gas_prediction == 0 else "Inefficient", "suggestions": [] } }
        # if security_prediction == 1 and 'FibonacciBalance.sol' in security_solutions:
        #     vuln = security_solutions['FibonacciBalance.sol']
        #     report["security"]["vulnerabilities"].append({ "type": vuln['vulnerabilities'][0]['category'], "description": "Detected vulnerability allows unauthorized access.", "vulnerable_code": "function doSomething() public { ... }", "fixed_code": "function doSomething() public onlyOwner { ... }" })
        if security_prediction == 1:
            report["security"]["vulnerabilities"].append({
                "type": "General Vulnerability Warning ⚠️",
                "description": "The AI model has flagged this contract as potentially vulnerable based on patterns learned from a large dataset of known exploits. This is not a definitive diagnosis but an indicator that the code requires a thorough manual security review.",
                "vulnerable_code": "N/A - The model does not identify specific lines of code.",
                "fixed_code": "Recommendation: Manually audit the contract for common issues like reentrancy, access control flaws, and integer overflows. Always follow security best practices."
            })
        
        if gas_prediction == 1 and "Repeated Storage Reads in Loops" in gas_solutions:
            opt = gas_solutions["Repeated Storage Reads in Loops"]
            report["gas_optimization"]["suggestions"].append({ "type": "Repeated Storage Reads in Loops", "description": opt['description'], "inefficient_code": opt['inefficient_code'], "efficient_code": opt['efficient_code'] })
        return jsonify(report)
    except Exception as e:
        print("\n--- DECRYPTION ERROR TRACEBACK ---")
        traceback.print_exc()
        print("----------------------------------\n")
        error_details = { "error": "Backend decryption failed. See server terminal for details.", "exception_type": type(e).__name__, "exception_message": str(e) }
        return jsonify(error_details), 500
# --- End of unchanged functions ---

if __name__ == "__main__":
    app.run(debug=True)