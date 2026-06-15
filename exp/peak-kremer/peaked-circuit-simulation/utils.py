import numpy as np
from qiskit import QuantumCircuit

import quimb.tensor as qtn
from quimb.tensor.tensor_arbgeom_compress import tensor_network_ag_compress

from qiskit_quimb import quimb_circuit
import torch

# cuSOLVER's default driver for complex64 SVD (gesvda/gesvd) fails to converge
# on ill-conditioned unitary tensors that arise in MPO compression.
# gesvdj (Jacobi) is slower but handles these cases correctly.
# Falls back to FP64 as a last resort so the run never crashes.
_real_linalg_svd = torch.linalg.svd

def _robust_svd(A, full_matrices=True, *, driver=None, out=None):
    if not A.is_cuda or driver is not None:
        return _real_linalg_svd(A, full_matrices=full_matrices, driver=driver, out=out)

    # Normalize to unit Frobenius norm before every attempt.  FP32 MPO tensors
    # can span many orders of magnitude; values rounded to 0 create exactly
    # degenerate columns that defeat iterative SVD algorithms.  Normalizing
    # puts all values in [-1, 1] where convergence is reliable.
    scale = torch.linalg.matrix_norm(A, ord='fro').clamp(min=1e-30)
    An = A / scale

    # 1. gesvdj on normalized FP32
    try:
        U, S, Vh = _real_linalg_svd(An, full_matrices=full_matrices, driver='gesvdj', out=out)
        return U, (S * scale).to(torch.float32), Vh
    except (torch._C._LinAlgError, RuntimeError):
        pass

    # 2. gesvdj on normalized FP64
    try:
        An64 = An.to(torch.complex128)
        U, S, Vh = _real_linalg_svd(An64, full_matrices=full_matrices, driver='gesvdj', out=None)
        return U.to(A.dtype), (S * scale).to(torch.float32), Vh.to(A.dtype)
    except (torch._C._LinAlgError, RuntimeError):
        pass

    # 3. CPU LAPACK zgesdd — divide-and-conquer, not iterative; handles exact
    #    degeneracy that defeats all CUDA Jacobi-based drivers.
    import numpy as np
    A_np = An.cpu().numpy()
    U_np, S_np, Vh_np = np.linalg.svd(A_np, full_matrices=False)
    s = float(scale.item())
    return (
        torch.from_numpy(U_np).to(device=A.device, dtype=A.dtype),
        torch.from_numpy(S_np * s).to(device=A.device, dtype=torch.float32),
        torch.from_numpy(Vh_np).to(device=A.device, dtype=A.dtype),
    )

torch.linalg.svd = _robust_svd

from qiskit.converters import circuit_to_dag, dag_to_circuit


# ------------------------------------------------------------------
#  Circuit layering
# ------------------------------------------------------------------

def iter_layers(qc):
    for layer in circuit_to_dag(qc).layers():
        yield dag_to_circuit(layer['graph'])

def merge_layers(ls, barrier=False):
    ls = iter(ls)
    qc = next(ls)
    for l in ls:
        if barrier:
            qc.barrier()
        qc = qc.compose(l)
    return qc

def merge_gates(gates, num_qubits=None):
    if num_qubits is None:
        num_qubits = gates[0].qubits[0]._register.size
    qc = QuantumCircuit(num_qubits, num_qubits)
    for g in gates:
        qc.append(g.operation, qargs=[q._index for q in g.qubits], cargs=[q._index for q in g.clbits])
    return qc

# ------------------------------------------------------------------
#  Backends
# ------------------------------------------------------------------

DEVICE = "cuda:0"
def to_backend_cuda(x, device=DEVICE):
    return torch.tensor(x, dtype=torch.complex64, device=device)


# ------------------------------------------------------------------
#  TN info
# ------------------------------------------------------------------

def elem_counts(tno):
    return sum(np.prod(t.shape).item() for t in tno)

def get_tn_info(tno):
    shapes_flat = [s for t in tno for s in t.shape]
    shapes_count = [len(t.shape) for t in tno]
    elem_counts = [np.prod(t.shape).item() for t in tno]

    return {
        "max_bond": tno.max_bond(),
        "max_links": max(shapes_count),
        "total_elems": sum(elem_counts),
        "total_shapes": np.sum(shapes_flat).item(),
        "total_links": sum(shapes_count),
        "num_tensors": tno.num_tensors,
    }

def get_bond_sizes(tno):
    N = len(tno.sites)
    elem_counts = [np.prod(t.shape).item() for t in tno]
    pair_bond_dims = dict()
    for ii in range(N):
        for jj in range(ii+1, N):
            try:
                pair_bond_dims[(ii,jj)] = tno.bond_size(ii,jj)
            except:
                continue

    pair_bond_weight = {(ii, jj):elem_counts[ii] + elem_counts[jj] for ii, jj in pair_bond_dims.keys()}
    return sorted(pair_bond_weight.items(), key=lambda rw: -rw[-1])


# ------------------------------------------------------------------
#  TN Methods
# ------------------------------------------------------------------

# method: 'l2bp', 'local-early', 'local-late', 'projector', 'superorthogonal'
def contract_core(layered_circuit, chunk_size=4, method='local-late', max_bond=32, cutoff=0.1, equalize_norms=True, to_backend=to_backend_cuda):
    N = layered_circuit[0].num_qubits
    L = len(layered_circuit)
    M = L // 2

    # Layers before and after the center
    layers_first_qc = layered_circuit[:M][::-1]
    layers_last_qc = layered_circuit[-M:]

    # This is a placeholder identity circuit to have all the sites
    qc_id = QuantumCircuit(N)
    qc_id.x(list(range(N)))
    qc_id.x(list(range(N)))

    # Start the TNO at the center of the circuit
    tno_core = quimb_circuit(layered_circuit[M].compose(qc_id), to_backend=to_backend).get_uni()
    stats = []

    # Merge and contract chunks of layers to the left and right of the circuit
    for i in range(0, M+1, chunk_size):

        # Make the left and right chunks
        qc_left = qc_id.copy()
        for j in reversed(range(chunk_size)):
            if (i+j) >= len(layers_first_qc):
                break
            qc_left = qc_left.compose(layers_first_qc[i+j])

        qc_right = qc_id.copy()
        for j in range(chunk_size):
            if (i+j) >= len(layers_last_qc):
                break
            qc_right = qc_right.compose(layers_last_qc[i+j])

        # Merge them with the current core
        tn_left = quimb_circuit(qc_left, to_backend=to_backend).get_uni()
        tn_right = quimb_circuit(qc_right, to_backend=to_backend).get_uni()
        tno_core = tn_left.gate_upper_with_op_lazy(tno_core).gate_upper_with_op_lazy(tn_right)

        # Compress the TNO
        tno_core = tensor_network_ag_compress(
            tno_core,
            method=method, 
            cutoff=cutoff,
            max_bond=max_bond,
            site_tags=tno_core.site_tags,
            canonize=True,
            equalize_norms=equalize_norms,
        )

        tno_core = tno_core.squeeze()
        shapes_flat = [s for t in tno_core for s in t.shape]
        shapes_count = [len(t.shape) for t in tno_core]
        elem_counts = [np.prod(t.shape).item() for t in tno_core]
        stats.append({
            "layer (distance from center)": i,
            "max_bond": tno_core.max_bond(),
            "total_elements": np.sum(shapes_flat).item()
        })
        print(f"    (|{M-1-(i+j)}-{M-1-i}|-{M}-|{M+1+i}-{M+1+i+j}|)/({len(layered_circuit)}) -> max_bond = {tno_core.max_bond()}, max_links = {max(shapes_count)}, total_elems = {sum(elem_counts)}, total_shapes = {np.sum(shapes_flat).item()}")

    return tno_core, stats


def tno_to_tne(tno, max_bond=8, cutoff=0.01, to_backend=to_backend_cuda):
    nq = len(tno.sites)
    tne = quimb_circuit(QuantumCircuit(nq), to_backend=to_backend).psi

    # Evolve state by op
    tne = qtn.tensor_arbgeom.tensor_network_apply_op_vec(tno, tne)

    # Compress
    tne = tensor_network_ag_compress(
        tne,
        method='local-late', 
        cutoff=cutoff,
        max_bond=max_bond,
        site_tags=tne.site_tags,
        canonize=True,
        #lazy=lazy,
        equalize_norms=True,
        #progbar=True,
        #to_backend=to_backend,
    )

    tne = tne.squeeze()
    shapes_count = [len(t.shape) for t in tne]
    print(f"max_bond = {tne.max_bond()}, max_links = {max(shapes_count)}")

    return tne


def extract_bitstring(tne):
    # Predict bitstring with marginals
    nq = len(tne.sites)
    pred_bs = ''
    p0s = []
    for ii in range(nq):
        if tne[0].backend == 'numpy':
            Pi0 = np.array([[1., 0.],[0., 0.]])
        else:
            Pi0 = torch.tensor(np.array([[1., 0.],[0., 0.]]), device=DEVICE, dtype=torch.cfloat)
        p0 = tne.local_expectation(Pi0, where=[ii], max_bond=2, optimize="auto", normalized=True).real.item()
        p0s.append(p0)
        pred_bs += '1' if p0 < 0.5 else '0'
        #print(f"({ii}) -> {p0:.3f} | {pred_bs}")
    return pred_bs, p0s


def bitstring_probability(psi, bitstring, optimize="auto-hq"):
    tn = psi.isel(
        {psi.site_ind(i): int(b) for i, b in zip(psi.sites, bitstring)}
    )
    amp = tn.contract(all, optimize=optimize)
    return abs(amp)**2


def sample_tns(psi, num_samples, max_distance=0, optimize="auto-hq"):
    rng = np.random.default_rng()
    samples = []
    for _ in range(num_samples):
        config, _ = psi.sample_configuration_cluster(
            gauges={}, 
            max_distance=max_distance, 
            seed=rng,
            optimize=optimize,
        )
        try:
            sites = psi.sites
        except AttributeError:
            sites = sorted(config)
        samples.append("".join(str(config[s]) for s in sites))
    return samples