from quimb.tensor import MatrixProductOperator, Circuit, CircuitMPS

from qiskit_quimb import quimb_circuit
from qiskit import QuantumCircuit

from circuit_mpo import apply_circuit, apply_swaps, mpo_from_circuit

from utils import iter_layers, merge_layers, elem_counts, merge_gates, get_tn_info, to_backend_cuda

import numpy as np
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%dT%H:%M:%S%z',
    level=logging.INFO
)


def _detect_backend():
    """Return the best available GPU backend callable/string, or None for CPU.

    Priority: torch (CUDA) > cupy > None.
    Returns to_backend_cuda (a callable) for torch, 'cupy' (a string quimb
    understands natively) for cupy, or None for CPU.
    """
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            logging.info(f"CUDA GPU detected via PyTorch ({name}) — defaulting to torch backend")
            return to_backend_cuda
    except Exception:
        pass
    try:
        import cupy as cp
        cp.zeros(1)
        logging.info("CUDA GPU detected via CuPy — defaulting to cupy backend")
        return 'cupy'
    except Exception:
        pass
    logging.info("No CUDA GPU detected — defaulting to CPU (numpy) backend")
    return None

_GPU_BACKEND = _detect_backend()


# ------------------------------------------------------------------
#  Segment utilities for U1U1d P1 U2U2d P2 U3U3d structure
# ------------------------------------------------------------------

def segment_circuit(circuit: QuantumCircuit, cuts: tuple[float, ...] = (1/3, 2/3)) -> list[QuantumCircuit]:
    """Split circuit.data at fractional gate-index positions.

    For the canonical 3-block structure the natural cuts are at 1/3 and 2/3,
    giving segments whose internal centers sit at 1/6, 1/2, 5/6 of the total.
    P layers are thin relative to UU† blocks so approximate cuts work well.
    """
    n_gates = len(circuit.data)
    indices = [0] + [int(n_gates * c) for c in cuts] + [n_gates]
    segments = []
    for lo, hi in zip(indices, indices[1:]):
        seg = QuantumCircuit(circuit.num_qubits)
        for inst in circuit.data[lo:hi]:
            seg.append(inst.operation, inst.qubits, inst.clbits)
        segments.append(seg)
    return segments


def _worker(args):
    """Worker for parallel_segment_unswap — one segment per thread."""
    seg, idx, kwargs = args
    logging.info(f"[segment {idx}] start ({len(seg.data)} gates)")
    t0 = time.perf_counter()
    result = mpo_compress_unswap(seg, center_ratio=0.5, **kwargs)
    logging.info(f"[segment {idx}] done in {time.perf_counter() - t0:.1f}s")
    return idx, result


def parallel_segment_unswap(
    circuit: QuantumCircuit,
    cuts: tuple[float, ...] = (1/3, 2/3),
    n_workers: int = 3,
    **kwargs,
) -> list[tuple]:
    """Run mpo_compress_unswap in parallel over the 3 UU† segments.

    Splits at `cuts` (default 1/3, 2/3), dispatches one worker per segment with
    center_ratio=0.5 — equivalent to global centers at 1/6, 1/2, 5/6.

    Uses ThreadPoolExecutor so all workers share a single CUDA context when
    running on GPU (ProcessPoolExecutor would require a separate context per
    process, which is both expensive and fragile with cupy).

    Args:
        circuit:   Full obfuscated circuit (U1U1d P1 U2U2d P2 U3U3d).
        cuts:      Fractional gate-index positions for segmentation.
        n_workers: Max parallel threads (set to 1 to debug serially).
        **kwargs:  Forwarded to mpo_compress_unswap (max_bond, cutoff, etc.).

    Returns:
        List of (mpo_core, layers_left, layers_right, stats_data) in segment order.
    """
    segments = segment_circuit(circuit, cuts)
    logging.info(
        f"[parallel_segment_unswap] {len(segments)} segments: "
        + ", ".join(f"{len(s.data)} gates" for s in segments)
    )

    work = [(seg, idx, kwargs) for idx, seg in enumerate(segments)]
    results = [None] * len(segments)

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_worker, item): item[1] for item in work}
        for fut in as_completed(futures):
            idx, result = fut.result()
            results[idx] = result

    return results

# ------------------------------------------------------------------
#  Rewiring
# ------------------------------------------------------------------
from qiskit.transpiler.passes import ElidePermutations, SabreSwap
from qiskit.transpiler import CouplingMap

def rewire_layers(ls, perm, seed=None):
    nq = len(perm)
    qc = merge_layers(ls)
    qc = QuantumCircuit(nq).compose(qc, qubits=np.argsort(perm))

    qc = ElidePermutations()(qc)
    ss = SabreSwap(coupling_map=CouplingMap.from_line(ls[0].num_qubits), heuristic='decay', trials=10000, seed=seed)
    qc = ss(qc)

    return list(iter_layers(qc))



# ------------------------------------------------------------------
#  Unswapping
# ------------------------------------------------------------------

def get_bond_sizes(mpo: MatrixProductOperator):
    return np.array([mpo.bond_size(ii,ii+1) for ii in range(len(mpo.sites) - 1)])


def swap_perm(perm, swaps):
    for q0, q1 in swaps:
        (perm[q0], perm[q1]) = (perm[q1], perm[q0])
    return perm


def get_good_swaps(mpo, qubit_pairs, how, max_bond, cutoff, to_backend=_GPU_BACKEND, equal=False):
    current_bonds = get_bond_sizes(mpo)
    #log_print("    [debug](select)(bond sizes before) -> ", current_bonds.tolist())

    swaps_l = qubit_pairs if how in ("left", "both") else []
    swaps_r = qubit_pairs if how in ("right", "both") else []

    mpo_tmp = apply_swaps(mpo, swaps_l=swaps_l, swaps_r=swaps_r, max_bond=max_bond, cutoff=cutoff, to_backend=to_backend)
    new_bonds = get_bond_sizes(mpo_tmp)
    if equal is None:
        new_bonds = new_bonds + (np.random.rand(*new_bonds.shape)-0.5)
        improved = np.nonzero(new_bonds < current_bonds)[0]
    elif equal:
        improved = np.nonzero(new_bonds <= current_bonds)[0]
    else:
        improved = np.nonzero(new_bonds < current_bonds)[0]

    return improved


def unswap(mpo: MatrixProductOperator, hows=("left", "right", "both"), max_bond=2048, cutoff=0.0001, max_its=25, equal=False, to_backend=_GPU_BACKEND, t0=0):
    num_qubits = len(mpo.sites)
    all_pairs = [(i, i+1) for i in range(num_qubits-1)]

    perm_left = list(range(len(mpo.sites)))
    perm_right = list(range(len(mpo.sites)))

    logging.info("    [start unswap] -> " + str(get_tn_info(mpo)))
    num_improvements = 1
    start_counts = 1
    end_counts = 0
    ii = 0

    stats_data = []
    while num_improvements > 0 and ii < max_its and start_counts != end_counts:
        num_improvements = 0
        start_counts = elem_counts(mpo)

        for how in hows:
            for parity in [0, 1]:
                # Estimate which qubit pairs to swap
                new_swap_ids = get_good_swaps(mpo, qubit_pairs=all_pairs[parity::2], how=how, max_bond=max_bond, cutoff=cutoff, to_backend=to_backend, equal=equal)
                new_swaps = [all_pairs[i] for i in new_swap_ids if i % 2 == parity]

                # Apply the selected swaps
                swaps_l = new_swaps if how in ("left", "both") else []
                swaps_r = new_swaps if how in ("right", "both") else []
                mpo = apply_swaps(mpo, swaps_l=swaps_l, swaps_r=swaps_r, max_bond=max_bond, cutoff=cutoff, to_backend=to_backend)

                # Update the permutations
                if how in ("left", "both"):
                    perm_left = swap_perm(perm_left, new_swaps)
                if how in ("right", "both"):
                    perm_right = swap_perm(perm_right, new_swaps)
    
                # Track how many new swaps were applied
                num_improvements += len(new_swap_ids)
                stats_data.append({"time": time.perf_counter()-t0, "stage": "unswapping", "iteration": ii, "side": how, "parity": parity, "new_swaps": len(new_swap_ids), "total_swaps": num_improvements, **get_tn_info(mpo)})
                logging.info(f"    [{ii} | {how} | {parity}](new_swaps: {len(new_swap_ids)} | total: {num_improvements}) -> " + str(get_tn_info(mpo)))

        end_counts = elem_counts(mpo)
        ii += 1
    logging.info(f"    [end unswap] -> " + str(get_tn_info(mpo)))

    return mpo, (perm_left, perm_right), stats_data


# ------------------------------------------------------------------
#  MPO Cancellation + Unswapping
# ------------------------------------------------------------------

def mpo_compress_unswap(circuit: QuantumCircuit, max_bond=8192, cutoff=0.001, unswap_threshold=1e6, early_stopping_gates=100, center_ratio=0.5, equal=False, flip_freq=None, max_its=20, to_backend=_GPU_BACKEND, seed=None, hows=("both", "left", "right"), mpo_core=None, progress_callback=None):
    q2c = lambda qc: quimb_circuit(qc.decompose("unitary"), Circuit, to_backend=to_backend)
    t0 = time.perf_counter()

    # Split circuit into left and right
    if type(center_ratio) is float:
        C = int(len(circuit) * center_ratio)
    elif type(center_ratio) is int:
        C = center_ratio
    circuit_left = merge_gates(circuit[:C], circuit.num_qubits).inverse()
    circuit_right = merge_gates(circuit[C:], circuit.num_qubits)
    if "measure" not in circuit_right.count_ops():
        circuit_right.measure_all()
    if "measure" not in circuit_left.count_ops():
        circuit_left.measure_all()

    layers_left = list(iter_layers(circuit_left))
    layers_right = list(iter_layers(circuit_right))


    T_U = circuit.count_ops().get("unitary", 0)
    T_UL = circuit_left.count_ops().get("unitary", 0)
    T_UR = circuit_right.count_ops().get("unitary", 0)

    logging.info(f"Total unitaries: {T_U} = {T_UL} (left) + {T_UR} (right)")
    if progress_callback is not None:
        progress_callback(0, T_U)

    # Rewire layers
    layers_left = rewire_layers(layers_left, np.arange(circuit.num_qubits, dtype=int), seed=seed)
    init_meas = layers_left[-2:]
    layers_left = layers_left[:-2]

    layers_right = rewire_layers(layers_right, np.arange(circuit.num_qubits, dtype=int), seed=seed)
    final_meas = layers_right[-2:]
    layers_right = layers_right[:-2]

    # Start the MPO and counters
    ii_left = 0
    ii_right = 0
    do_left = False
    if mpo_core is None:
        mpo_core = mpo_from_circuit(q2c(QuantumCircuit(circuit.num_qubits)))
    logging.info("[start compressing] -> " + str(get_tn_info(mpo_core)))


    total_u_consumed = 0
    current_u_consumed = 0
    total_u_consumed_left = 0
    total_u_consumed_right = 0

    stats_data = []

    # Pre-warm autoray's torch.linalg.qr lazy dispatch wrapper.
    # The first call to qr initialises a global lazy wrapper; if two threads
    # race to do this simultaneously the second raises "lazy wrapper should be
    # called at most once". Trigger it once here, before the thread pool starts.
    _warm = mpo_core.arrays[0]
    if hasattr(_warm, 'device'):
        import torch as _torch
        _d = _warm.reshape(-1)[:1].expand(2).reshape(1, 2)
        _torch.linalg.qr(_d)
        del _warm, _d, _torch

    # Persistent thread pool for concurrent left/right probes.
    # Both probes read mpo_core without mutating it, so they are safe to run
    # in parallel. On GPU the two CUDA kernel streams overlap on the device.
    _probe_pool = ThreadPoolExecutor(max_workers=2)

    def _probe_left(core, layer, mb, co):
        result = apply_circuit(core, q2c(layer.inverse()), side="right", max_bond=mb, cutoff=co)
        return result, elem_counts(result)

    def _probe_right(core, layer, mb, co):
        result = apply_circuit(core, q2c(layer), side="left", max_bond=mb, cutoff=co)
        return result, elem_counts(result)

    # Start loop
    while ii_left < len(layers_left) or ii_right < len(layers_right):
        # Dispatch both probes concurrently; they read mpo_core without mutating it.
        fut_l = (
            _probe_pool.submit(_probe_left, mpo_core, layers_left[ii_left], max_bond, cutoff)
            if ii_left < len(layers_left) else None
        )
        fut_r = (
            _probe_pool.submit(_probe_right, mpo_core, layers_right[ii_right], max_bond, cutoff)
            if ii_right < len(layers_right) else None
        )

        try:
            mpo_left, counts_left   = fut_l.result() if fut_l else (None, 1e20)
            mpo_right, counts_right = fut_r.result() if fut_r else (None, 1e20)
        except KeyboardInterrupt:
            _probe_pool.shutdown(wait=False, cancel_futures=True)
            break
        
        if flip_freq is None:
            do_left = counts_left < counts_right
        else:
            if mpo_left is None:
                do_left = False
            elif mpo_right is None:
                do_left = True
            elif (ii_right + ii_left) % flip_freq == 0:
                do_left = not do_left

        # Select the smallest one
        if [counts_right, counts_left][int(do_left)] < unswap_threshold:                
            if do_left:
                mpo_core = mpo_left
                # Update counts
                new_ops = dict(layers_left[ii_left].count_ops())
                new_us = new_ops.get('unitary', 0)
                new_swaps = new_ops.get('swap', 0)
                total_u_consumed += new_us
                current_u_consumed += new_us
                total_u_consumed_left += new_us

                # Log
                side_chosen = "L"
                ii_left += 1
            else:
                mpo_core = mpo_right
                # Update counts
                new_ops = dict(layers_right[ii_right].count_ops())
                new_us = new_ops.get('unitary', 0)  
                new_swaps = new_ops.get('swap', 0)
                total_u_consumed += new_us
                current_u_consumed += new_us
                total_u_consumed_right += new_us
            
                # Log
                side_chosen = "R"
                ii_right += 1            
            
            logging.info((f"[{ii_right}R/{len(layers_right)}]" if side_chosen == "R" else f"[{ii_left}L/{len(layers_left)}]") + 
                         f"(swap: {new_swaps}, u: {new_us} | c_u: {current_u_consumed} | t_u_l: {total_u_consumed_left}/{T_UL} | t_u_r: {total_u_consumed_right}/{T_UR} | t_u: {total_u_consumed}/{T_U}) -> " +
                         str(get_tn_info(mpo_core)))
            stats_data.append({"time": time.perf_counter() - t0, "stage": "absorbing", "absorb_side": "left",
                                "it_left": ii_left, "it_right": ii_right, "layers_left": len(layers_left), "layers_right": len(layers_right),
                                "u_consumed_total_left": total_u_consumed_left, "u_consumed_total_right": total_u_consumed_right, "u_consumed_total": total_u_consumed,
                                "swap_consumed": new_swaps, "u_consumed": new_us, "u_consumed_after_unswap": current_u_consumed,
                                **get_tn_info(mpo_core)})
            if progress_callback is not None:
                progress_callback(total_u_consumed, T_U)
        
        # Unswap if both sides go over the size budget
        else: 
            # Apply unswapping
            try:
                mpo_core, (new_perm_left, new_perm_right), new_unswap_stats = unswap(mpo_core, hows=hows, max_bond=max_bond, cutoff=cutoff, max_its=max_its, equal=equal, to_backend=to_backend, t0=t0)
                stats_data += new_unswap_stats
            except KeyboardInterrupt:
                break        
            # Rewire left circuit
            if ii_left < len(layers_left):
                layers_left = rewire_layers(layers_left[(ii_left):] + init_meas, new_perm_left, seed=seed)
                init_meas = layers_left[-2:]
                layers_left = layers_left[:-2]
            else:
                layers_left = []
            
            # Rewire right circuit
            if ii_right < len(layers_right):
                layers_right = rewire_layers(layers_right[(ii_right):] + final_meas, new_perm_right, seed=seed)
                final_meas = layers_right[-2:]
                layers_right = layers_right[:-2]
            else:
                layers_right = []
            
            ii_left = 0
            ii_right = 0
            current_u_consumed = 0

            # Stop early if there are few gates left
            if (T_U - total_u_consumed) <= early_stopping_gates:
                break
    
    _probe_pool.shutdown(wait=False)

    # Remove any leftover layers
    layers_left = layers_left[(ii_left):] if ii_left < len(layers_left) else []
    layers_left += init_meas
    layers_right = layers_right[(ii_right):] if ii_right < len(layers_right) else []
    layers_right += final_meas

    logging.info(f"[end compressing](left: {len(layers_left)}, right: {len(layers_right)}) -> " + str(get_tn_info(mpo_core)))

    return mpo_core, layers_left, layers_right, stats_data


def mpo_to_mps(mpo_core, layers_left, layers_right, max_bond=4096, cutoff=0.001, to_backend=_GPU_BACKEND):
    q2c = lambda qc: quimb_circuit(qc.decompose("unitary"), Circuit, to_backend=to_backend)
    # Use the compressed MPO to get the MPS by applying it to |0> state
    final_mps = quimb_circuit(
        QuantumCircuit(len(mpo_core.sites)),
        quimb_circuit_class=CircuitMPS,
        to_backend=to_backend,
    ).psi

    # First take the leftover front layers
    layers_left = list(iter_layers(merge_layers(layers_left).inverse())) if len(layers_left) > 0 else []
    
    for ii_left in range(len(layers_left)):
        l_left = layers_left[ii_left]
        new_ops = dict(l_left.count_ops())
        layer_mpo = mpo_from_circuit(q2c(l_left))
        final_mps = layer_mpo.apply(final_mps, compress=True, max_bond=max_bond, cutoff=cutoff)
        logging.info(f"[Left {ii_left} / {len(layers_left)}] -> " + str(get_tn_info(final_mps)))

    logging.info("[Left MPS] -> " + str(get_tn_info(final_mps)))

    # Then apply the compressed MPO to the layers
    final_mps = mpo_core.apply(final_mps, compress=True, max_bond=max_bond, cutoff=cutoff)
    logging.info("[Left MPS + Core MPO] -> " + str(get_tn_info(final_mps)))

    # Then iterate through final layers if there are any
    final_meas = []
    for ii_right in range(len(layers_right)):
        l_right = layers_right[ii_right]
        new_ops = dict(l_right.count_ops())
        if "barrier" in new_ops or "measure" in new_ops:
            final_meas.append(l_right)
        else:
            layer_mpo = mpo_from_circuit(q2c(l_right))
            final_mps = layer_mpo.apply(final_mps, compress=True, max_bond=max_bond, cutoff=cutoff)
            logging.info(f"[Front MPS + Core MPO + Right {ii_right} / {len(layers_right)}] -> " + str(get_tn_info(final_mps)))
    
    logging.info(f"[Front MPS + Core MPO + Right MPS] -> " + str(get_tn_info(final_mps)))

    # Extract final permutation from measurements
    final_perm = [g.qubits[0]._index for g in final_meas[-1]]

    # Return MPS and final perm
    return final_mps, final_perm


def simulate_qasm(
    qasm_path: str,
    n_samples: int = 10000,
    top_k: int = 10,
    compress_kwargs: dict = None,
    mps_kwargs: dict = None,
    progress_callback=None,
) -> list[tuple[str, int]]:
    """Load a QASM file and return the top-k bitstrings by sample count.

    Args:
        qasm_path:        Path to the .qasm file.
        n_samples:        Number of MPS samples to draw.
        top_k:            How many (bitstring, count) pairs to return.
        compress_kwargs:  Overrides for mpo_compress_unswap (max_bond, cutoff, …).
        mps_kwargs:       Overrides for mpo_to_mps (max_bond, cutoff, …).

    Returns:
        List of (bitstring, count) tuples, highest count first.
    """
    from collections import Counter
    from qiskit.transpiler.passes import Collect2qBlocks, ConsolidateBlocks
    from qiskit.transpiler import PassManager

    compress_kwargs = compress_kwargs or {}
    mps_kwargs = mps_kwargs or {}

    circuit = QuantumCircuit.from_qasm_file(qasm_path)
    circuit = PassManager([
        Collect2qBlocks(),
        ConsolidateBlocks(force_consolidate=True),
    ]).run(circuit)
    logging.info(f"[simulate_qasm] loaded {qasm_path} — {circuit.count_ops()}")

    mpo, layers_left, layers_right, _ = mpo_compress_unswap(circuit, progress_callback=progress_callback, **compress_kwargs)
    mps, perm = mpo_to_mps(mpo, layers_left[:-2], layers_right, **mps_kwargs)

    raw = ["".join(str(b) for b in bs) for bs, _ in mps.sample(n_samples)]
    # apply the output permutation
    permuted = ["".join(bs[i] for i in perm) for bs in raw]

    return Counter(permuted).most_common(top_k)

