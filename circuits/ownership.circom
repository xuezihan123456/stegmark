pragma circom 2.1.0;

/*
 * OwnershipProof — Pedersen Commitment + Schnorr Sigma Protocol
 * =============================================================
 * This is a PLACEHOLDER circuit intended to show how the Python-level
 * Schnorr proof (zk_ownership.py) would be expressed in circom 2.1.0
 * for use with snarkjs / Groth16 or PLONK.
 *
 * Production workflow:
 *   1. Install dependencies:
 *        npm install -g circom snarkjs
 *        npm install circomlib           # for Poseidon, BabyJub helpers
 *
 *   2. Compile this circuit:
 *        circom ownership.circom --r1cs --wasm --sym -o build/
 *
 *   3. Trusted setup (Groth16, powers-of-tau already downloaded):
 *        snarkjs groth16 setup build/ownership.r1cs pot12_final.ptau build/ownership_0000.zkey
 *        snarkjs zkey contribute build/ownership_0000.zkey build/ownership_0001.zkey --name="contributor"
 *        snarkjs zkey export verificationkey build/ownership_0001.zkey build/verification_key.json
 *
 *   4. Generate proof (provide input.json with private/public signals):
 *        snarkjs groth16 fullprove input.json build/ownership_js/ownership.wasm build/ownership_0001.zkey proof.json public.json
 *
 *   5. Verify proof:
 *        snarkjs groth16 verify build/verification_key.json public.json proof.json
 *
 * Signal naming convention
 * ------------------------
 *   Private inputs (known only to prover):
 *     secret_key  — 256-bit secret (as field element, truncated to curve order)
 *     blinding    — 256-bit blinding factor
 *     nonce_v     — random nonce v (for secret response)
 *     nonce_t     — random nonce t (for blinding response)
 *
 *   Public inputs (shared with verifier):
 *     image_hash  — SHA-256 of watermarked image (as 256-bit integer)
 *     commitment_x, commitment_y — Pedersen commitment C = s·G + b·H (affine)
 *
 *   Outputs (derived, public):
 *     valid       — 1 if proof is valid, 0 otherwise
 *
 * NOTE: Full implementation requires circomlib's Num2Bits, BabyAdd, BabyDbl,
 * and EscalarMulAny templates, plus a Poseidon or SHA-256 hash gadget.
 * The secp256k1 field does NOT match the BN128 scalar field used by snarkjs
 * by default; a production circuit would use the BabyJubJub curve (defined
 * over BN128) or the secp256k1-in-SNARK trick via range proofs.
 */

// ---------------------------------------------------------------------------
// Placeholder template — signals declared but constraints omitted
// Replace the body with real R1CS constraints for production use.
// ---------------------------------------------------------------------------
template OwnershipProof() {
    // --- Private inputs ---
    signal input secret_key;   // prover's secret integer (mod curve order)
    signal input blinding;     // Pedersen blinding factor
    signal input nonce_v;      // Schnorr nonce for secret response
    signal input nonce_t;      // Schnorr nonce for blinding response

    // --- Public inputs ---
    signal input image_hash;   // SHA-256(image) represented as field element
    signal input commitment_x; // x-coordinate of Pedersen commitment C
    signal input commitment_y; // y-coordinate of Pedersen commitment C

    // --- Output ---
    signal output valid;       // 1 = proof accepted, 0 = rejected

    /*
     * TODO (production): implement R1CS constraints for:
     *
     *   1. Scalar multiplication  V = nonce_v·G + nonce_t·H
     *      → Use EscalarMulAny from circomlib/escalarmulany.circom
     *
     *   2. Fiat-Shamir challenge  c = Poseidon([image_hash, Cx, Cy, Vx, Vy])
     *      → Use Poseidon(5) from circomlib/poseidon.circom
     *      (SHA-256 in-circuit is expensive; Poseidon is ZK-friendly)
     *
     *   3. Response check
     *      s_resp·G + r_resp·H  ==  V + c·C
     *      → Decompose scalars with Num2Bits(254), apply EscalarMulAny,
     *        then BabyAdd for point addition
     *
     *   4. Commitment consistency
     *      commitment_x, commitment_y must lie on the curve
     *      → BabyCheck from circomlib/babyjub.circom
     */

    // Placeholder: set valid = 1 unconditionally (REMOVE in production)
    valid <== 1;
}

component main {public [image_hash, commitment_x, commitment_y]} = OwnershipProof();
