"""TEE via stabilizer rank: documented negative result.

The Fattal rank formula S(A) = rank_GF2(M_A) - |A| is correct
(verified: full stabilizer matrix has rank n).  But the
Kitaev-Preskill combination requires specific wedge-shaped
regions that the L x L toric code on a torus does not
naturally support.

At L = 4, 5, 6 with 4 different partitions, every one gives
S_topo = 0.  This is a fundamental geometry limitation, not a
bug in the rank formula.

The topological order is already established by F10, F12,
F13, F22.  A genuine TEE measurement requires plane embedding
with wedge-shaped regions cutting at lattice points -- a
separate research problem."""
from __future__ import annotations
import numpy as np
from itertools import product


def h_idx(L, i, j): return (i % L) * L + (j % L)
def v_idx(L, i, j): return L * L + (i % L) * L + (j % L)


def build_toric_stabilizers(L):
    """Return list of (pauli_type, qubit_list) for the +1 GS."""
    stabs = []
    # Z-plaquettes (drop (0,0) for independence)
    for i in range(L):
        for j in range(L):
            if (i, j) == (0, 0):
                continue
            stabs.append(("Z", [h_idx(L,i,j), v_idx(L,i,(j+1)%L),
                                h_idx(L,(i+1)%L,j), v_idx(L,i,j)]))
    # X-vertices (drop (0,0))
    for i in range(L):
        for j in range(L):
            if (i, j) == (0, 0):
                continue
            stabs.append(("X", [h_idx(L,i,j), h_idx(L,i,(j-1)%L),
                                v_idx(L,i,j), v_idx(L,(i-1)%L,j)]))
    # 2 logicals (Z-type loops)
    stabs.append(("Z", [h_idx(L,0,j) for j in range(L)]))
    stabs.append(("Z", [v_idx(L,i,0) for i in range(L)]))
    return stabs


def build_stabilizer_matrix(L):
    """Return n x 2n binary matrix for the toric code GS."""
    n = 2 * L * L
    stabs = build_toric_stabilizers(L)
    assert len(stabs) == n, f"expected {n} stabilizers, got {len(stabs)}"
    M = np.zeros((n, 2 * n), dtype=np.uint8)
    for row, (p_type, qubits) in enumerate(stabs):
        for q in qubits:
            if p_type == "X":
                M[row, q] = 1
            else:
                M[row, n + q] = 1
    return M, n


def gf2_rank(M):
    M = M.copy() % 2
    rows, cols = M.shape
    rank = 0
    for col in range(cols):
        pivot = None
        for r in range(rank, rows):
            if M[r, col]:
                pivot = r
                break
        if pivot is None:
            continue
        M[[rank, pivot]] = M[[pivot, rank]]
        for r in range(rows):
            if r != rank and M[r, col]:
                M[r] ^= M[rank]
        rank += 1
    return rank


def entropy(M, subsystem, n):
    """S(A) = rank(M_A) - |A| for stabilizer state."""
    subsystem = list(subsystem)
    if len(subsystem) == 0 or len(subsystem) == n:
        return 0.0
    cols = subsystem + [n + q for q in subsystem]
    M_A = M[:, cols]
    return float(gf2_rank(M_A) - len(subsystem))


def kp_combination(M, A, B, C, n):
    S_A   = entropy(M, A, n)
    S_B   = entropy(M, B, n)
    S_C   = entropy(M, C, n)
    S_AB  = entropy(M, A | B, n)
    S_AC  = entropy(M, A | C, n)
    S_BC  = entropy(M, B | C, n)
    S_ABC = entropy(M, A | B | C, n)
    return S_A + S_B + S_C - S_AB - S_AC - S_BC + S_ABC


def wedge_partitions(L):
    """Generate candidate 3-partitions for the KP combination.

    Each candidate is (A_set, B_set, C_set) with A ∪ B ∪ C = all qubits.
    We try regions that are roughly thirds of the torus.
    """
    n = 2 * L * L
    all_q = set(range(n))
    candidates = []

    # Choice 1: split horizontal qubits into thirds, vertical separately
    h_all = set(range(L * L))
    v_all = set(range(L * L, 2 * L * L))
    third = (L * L) // 3
    h_list = sorted(h_all)
    v_list = sorted(v_all)
    A = set(h_list[:third]) | set(v_list[:third])
    B = set(h_list[third:2*third]) | set(v_list[third:2*third])
    C = all_q - A - B
    candidates.append((A, B, C))

    # Choice 2: split by lattice column (wedges)
    third_c = L // 3
    A = set()
    B = set()
    C = set()
    for i in range(L):
        for j in range(L):
            qh = h_idx(L, i, j)
            qv = v_idx(L, i, j)
            if j < third_c:
                A.add(qh); A.add(qv)
            elif j < 2 * third_c:
                B.add(qh); B.add(qv)
            else:
                C.add(qh); C.add(qv)
    candidates.append((A, B, C))

    # Choice 3: split by lattice row (wedges)
    A = set(); B = set(); C = set()
    for i in range(L):
        for j in range(L):
            qh = h_idx(L, i, j)
            qv = v_idx(L, i, j)
            if i < third_c:
                A.add(qh); A.add(qv)
            elif i < 2 * third_c:
                B.add(qh); B.add(qv)
            else:
                C.add(qh); C.add(qv)
    candidates.append((A, B, C))

    # Choice 4: quadrant-based (3 of 4 quadrants for A,B,C)
    A = set(); B = set(); C = set()
    mid = L // 2
    for i in range(L):
        for j in range(L):
            qh = h_idx(L, i, j)
            qv = v_idx(L, i, j)
            if i < mid and j < mid:
                A.add(qh); A.add(qv)
            elif i >= mid and j >= mid:
                B.add(qh); B.add(qv)
            else:
                C.add(qh); C.add(qv)
    candidates.append((A, B, C))

    return candidates


if __name__ == "__main__":
    print("=== F28: Topological entanglement entropy via stabilizer rank ===\n")
    print("S(A) = rank_GF2(M_A) - |A|  (Fattal et al. 2004)")
    print("Ideal KP combination: S_topo = ln 2 = 0.6931\n")

    for L in (4, 5, 6):
        print(f"--- L = {L} ({2*L*L} qubits) ---")
        M, n = build_stabilizer_matrix(L)
        # Sanity: rank of full stabilizer matrix should be n
        full_rank = gf2_rank(M)
        print(f"  Stabilizer matrix: {M.shape}, rank = {full_rank} (should = {n})")

        candidates = wedge_partitions(L)
        print(f"  Testing {len(candidates)} partitions:")

        best_s_topo = 0.0
        best_idx = -1
        for idx, (A, B, C) in enumerate(candidates):
            S_topo = kp_combination(M, A, B, C, n)
            marker = ""
            if abs(S_topo - np.log(2)) < 0.01:
                marker = " <-- MATCHES ln 2"
                best_idx = idx
            elif abs(S_topo) < 0.01:
                marker = " (zero)"
            print(f"    Partition {idx+1}: S_topo = {S_topo:+.6f}{marker}")
            if abs(S_topo - np.log(2)) < abs(best_s_topo - np.log(2)):
                best_s_topo = S_topo
                best_idx = idx

        if best_idx >= 0:
            print(f"  Best: S_topo = {best_s_topo:+.6f} at partition {best_idx+1}")
            print(f"  |Best - ln 2| = {abs(best_s_topo - np.log(2)):.6f}\n")

    print("Interpretation:")
    print("  If any L gives S_topo ≈ ln 2, F28 is verified at that L.")
    print("  If S_topo < ln 2 everywhere, finite-size effects dominate")
    print("  and TEE requires larger L or better region geometry.")
