"""MPI-distributed SYK wormhole calculation for NERSC CPU nodes.

Dynamite performs its normal PETSc/SLEPc initialization once in every process,
but uses ``MPI.COMM_SELF`` so each process owns an independent serial Dynamite
instance.  MPI ranks are divided adaptively: independent (Hamiltonian, state)
samples are distributed first; ranks left over collaborate on disjoint pieces
of the Majorana sum.
"""

#Python packages
from argparse import ArgumentParser
import csv
from datetime import date, datetime, timezone
from itertools import combinations
import json
import os
from pathlib import Path
import platform
import resource
from sys import stderr
from time import perf_counter
import numpy as np
today = date.today()

#multiprocessing packages 
#taken from NERSC recommended mpi4py integration - https://docs.nersc.gov/development/languages/python/parallel-python/#mpi4py
#NOTE 8/24/26 - need to adapt for GPU integration via https://docs.nersc.gov/development/languages/python/using-python-perlmutter/; currently this should just work well for CPUs
from mpi4py import MPI
import petsc4py
import slepc4py

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()
print('MPI RANK', 'MPI SIZE')
print(RANK, SIZE)
print()

_native_slepc_init = slepc4py.init


def _init_slepc_on_comm_self(args=None, arch=None):
    """Initialize PETSc on COMM_SELF before native SLEPc initialization."""
    petsc4py.init(args=args, arch=arch, comm=MPI.COMM_SELF)
    _native_slepc_init(args=args, arch=arch)


# Let Dynamite run its native initialization while intercepting its call to
# slepc4py.init so PETSc uses one independent communicator per global MPI rank.
slepc4py.init = _init_slepc_on_comm_self

from dynamite import config
from inspect import getfile
print(f'MPI rank {RANK}: imported dynamite config from {getfile(config.__class__)}')

config.initialize(gpu=False, version_check=False)

from petsc4py import PETSc
assert PETSc.COMM_WORLD.size == 1

from dynamite.extras import majorana
from dynamite.operators import op_product, op_sum
from dynamite.states import State
from dynamite.subspaces import Parity
from dynamite.tools import get_version_str, track_memory, get_memory_usage



def root_print(*args, **kwargs):
    """Print only from global MPI rank zero."""
    if RANK == 0:
        print(*args, **kwargs)


def main():
    args = parse_args()
    COMM.Barrier()
    start_time = perf_counter()
    start_utc = datetime.now(timezone.utc)
    track_memory()

    root_print('== Run parameters: ==', file=stderr)
    for key, value in vars(args).items():
        if key != 'seed':
            root_print(f'  {key}, {value}', file=stderr)
    root_print(f'  mpi_ranks, {SIZE}', file=stderr)

    sample_count = args.H_iters * args.state_iters
    if sample_count < 1:
        raise ValueError('H-iters and state-iters must both be positive')

    # Use outer-loop parallelism first.  Only when there are more ranks than
    # independent samples do multiple ranks cooperate on each Majorana sum.
    group_count = min(SIZE, sample_count)
    group_id = RANK % group_count
    GROUP_COMM = COMM.Split(color=group_id, key=RANK)
    group_rank = GROUP_COMM.Get_rank()
    group_size = GROUP_COMM.Get_size()
    root_print(f'  sample_groups, {group_count}', file=stderr)
    root_print(f'  ranks_per_group, {group_size} (or one fewer)', file=stderr)

    seed = get_shared_seed(args.seed)
    root_print(f'  seed, {seed}', file=stderr)
    np.random.seed(seed)
    root_print(file=stderr)

    config.shell = not args.no_shell
    config.L = (args.N + 1) // 2

    even_space = Parity('even')
    odd_space = Parity('odd')

    # Every rank constructs identical operators, but computes only the indices
    # j = rank, rank + size, rank + 2*size, ... in compute_two_point().
    maj_ops = []
    for j in range(args.N):
        maj = majorana(j)
        maj.add_subspace(even_space, odd_space)
        maj.add_subspace(odd_space, even_space)
        maj_ops.append(maj)

    sorted_beta = sorted(args.b)

    root_print('beta,t,G_LL', flush=True)

    bt_pairs = [(b, t) for b in sorted_beta for t in args.t]
    local_sums = np.zeros(len(bt_pairs), dtype=np.complex128)
    local_counts = np.zeros(len(bt_pairs), dtype=np.int64)
    local_records = []

    seed_sequence = np.random.SeedSequence(seed)
    h_sequences = seed_sequence.spawn(args.H_iters)

    for h_idx, h_sequence in enumerate(h_sequences):
        state_sequences = h_sequence.spawn(args.state_iters + 1)
        h_seed = state_sequences[0].generate_state(1, dtype=np.uint64)[0]

        # Build this realization only in groups that own at least one of its
        # state samples. It may be replicated across groups to expose state-level
        # parallelism, while ranks within a group construct identical copies.
        owned_states = [
            state_idx for state_idx in range(args.state_iters)
            if (h_idx * args.state_iters + state_idx) % group_count == group_id
        ]
        if not owned_states:
            continue

        H = build_hamiltonian(args.N, np.random.default_rng(h_seed))
        H.add_subspace(even_space)
        H.add_subspace(odd_space)

        for state_idx in owned_states:
            state_seed = int(state_sequences[state_idx + 1].generate_state(1)[0])
            psi_r = State(state='random', subspace=even_space, seed=state_seed)
            beta_state = psi_r.copy()
            evolved = psi_r.copy()

            pair_idx = 0
            for i, b in enumerate(sorted_beta):
                delta_b = b if i == 0 else b - sorted_beta[i - 1]
                H.evolve(beta_state, t=-1j * delta_b / 2, result=evolved)
                evolved.normalize()
                evolved.copy(result=beta_state)

                for t in args.t:
                    result = compute_two_point(
                        beta_state, t, b, H, maj_ops, args.N,
                        even_space, odd_space, GROUP_COMM, group_rank, group_size
                    )
                    if group_rank == 0:
                        local_sums[pair_idx] += result
                        local_counts[pair_idx] += 1
                        local_records.append((h_idx, state_idx, b, t, result))
                        # Emit each sample immediately rather than waiting for
                        # all Hamiltonian/state averages and MPI gathering.
                        print(f'{b},{t},{result}', flush=True)
                    pair_idx += 1

    global_sums = np.zeros_like(local_sums) if RANK == 0 else None
    global_counts = np.zeros_like(local_counts) if RANK == 0 else None
    COMM.Reduce(local_sums, global_sums, op=MPI.SUM, root=0)
    COMM.Reduce(local_counts, global_counts, op=MPI.SUM, root=0)
    gathered_records = COMM.gather(local_records, root=0)

    if RANK == 0:
        # These receive buffers exist only on rank 0 by construction.
        assert global_sums is not None and global_counts is not None
        records = sorted(
            (record for rank_records in gathered_records for record in rank_records),
            key=lambda record: (record[0], record[1], record[2], record[3])
        )

        sorted_t = sorted(set(args.t))
        average_by_bt = {
            pair: global_sums[idx] / global_counts[idx]
            for idx, pair in enumerate(bt_pairs)
            if global_counts[idx]
        }
        for b in sorted_beta:
            t_list = [t for t in sorted_t if (b, t) in average_by_bt]
            result_list = [average_by_bt[(b, t)] for t in t_list]
            root_print('times', t_list)
            root_print('values', result_list)

    # ru_maxrss captures the whole Python process, including non-PETSc memory.
    # Linux (including NERSC) reports KiB; macOS reports bytes.
    local_peak_rss_gb = peak_rss_gb()
    peak_rss_sum_gb = COMM.reduce(local_peak_rss_gb, op=MPI.SUM, root=0)
    peak_rss_max_rank_gb = COMM.reduce(local_peak_rss_gb, op=MPI.MAX, root=0)
    local_petsc_peak_gb = get_memory_usage(group_by='rank', max_usage=True)
    petsc_peak_sum_gb = COMM.reduce(local_petsc_peak_gb, op=MPI.SUM, root=0)
    petsc_peak_max_rank_gb = COMM.reduce(local_petsc_peak_gb, op=MPI.MAX, root=0)
    hostnames = COMM.gather(MPI.Get_processor_name(), root=0)

    COMM.Barrier()
    local_elapsed_seconds = perf_counter() - start_time
    elapsed_seconds = COMM.reduce(local_elapsed_seconds, op=MPI.MAX, root=0)

    if RANK == 0:
        save_results(
            args=args,
            records=records,
            global_sums=global_sums,
            global_counts=global_counts,
            bt_pairs=bt_pairs,
            sorted_beta=sorted_beta,
            sorted_t=sorted_t,
            group_count=group_count,
            hostnames=hostnames,
            seed_used=seed,
            start_utc=start_utc,
            elapsed_seconds=elapsed_seconds,
            peak_rss_sum_gb=peak_rss_sum_gb,
            peak_rss_max_rank_gb=peak_rss_max_rank_gb,
            petsc_peak_sum_gb=petsc_peak_sum_gb,
            petsc_peak_max_rank_gb=petsc_peak_max_rank_gb,
        )


def peak_rss_gb():
    """Return this process's maximum resident set size in decimal GB."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    bytes_per_unit = 1 if platform.system() == 'Darwin' else 1024
    return rss * bytes_per_unit / 1E9


def compact_number(value):
    """Format a numeric parameter for a filesystem-safe run name."""
    return f'{value:g}'.replace('-', 'm').replace('.', 'p').replace('+', '')


def run_stem(args, start_utc):
    """Build an informative but bounded filename from the run parameters."""
    b_min, b_max = min(args.b), max(args.b)
    t_min, t_max = min(args.t), max(args.t)
    return (
        f'syk_GLL_N{args.N}_H{args.H_iters}_S{args.state_iters}'
        f'_b{compact_number(b_min)}-{compact_number(b_max)}x{len(args.b)}'
        f'_t{compact_number(t_min)}-{compact_number(t_max)}x{len(args.t)}'
        f'_ranks{SIZE}_{start_utc.strftime("%Y-%m-%d_%H%M%SUTC")}'
    )


def save_results(args, records, global_sums, global_counts, bt_pairs,
                 sorted_beta, sorted_t, group_count, hostnames, seed_used,
                 start_utc, elapsed_seconds, peak_rss_sum_gb, peak_rss_max_rank_gb,
                 petsc_peak_sum_gb, petsc_peak_max_rank_gb):
    """Save dense numerical arrays to NPZ and one-row run metadata to CSV."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = run_stem(args, start_utc)
    npz_path = output_dir / f'{stem}.npz'
    csv_path = output_dir / f'{stem}_metadata.csv'

    beta_values = np.asarray(sorted_beta, dtype=np.float64)
    time_values = np.asarray(sorted_t, dtype=np.float64)
    beta_index = {value: idx for idx, value in enumerate(sorted_beta)}
    time_index = {value: idx for idx, value in enumerate(sorted_t)}

    sample_gll = np.full(
        (args.H_iters, args.state_iters, len(sorted_beta), len(sorted_t)),
        np.nan + 1j * np.nan,
        dtype=np.complex128,
    )
    for h_idx, state_idx, beta, time, result in records:
        sample_gll[h_idx, state_idx, beta_index[beta], time_index[time]] = result

    average_gll = np.full(
        (len(sorted_beta), len(sorted_t)),
        np.nan + 1j * np.nan,
        dtype=np.complex128,
    )
    sample_counts = np.zeros((len(sorted_beta), len(sorted_t)), dtype=np.int64)
    for idx, (beta, time) in enumerate(bt_pairs):
        b_idx, t_idx = beta_index[beta], time_index[time]
        sample_counts[b_idx, t_idx] = global_counts[idx]
        if global_counts[idx]:
            average_gll[b_idx, t_idx] = global_sums[idx] / global_counts[idx]

    # This is useful for separating disorder variance from state-typicality
    # variance without rereading or regrouping the complete sample tensor.
    h_average_gll = np.nanmean(sample_gll, axis=1)

    np.savez(
        npz_path,
        average_gll=average_gll,
        h_average_gll=h_average_gll,
        sample_gll=sample_gll,
        sample_counts=sample_counts,
        beta_values=beta_values,
        time_values=time_values,
        h_indices=np.arange(args.H_iters, dtype=np.int64),
        state_indices=np.arange(args.state_iters, dtype=np.int64),
    )

    unique_hosts = sorted(set(hostnames))
    end_utc = datetime.now(timezone.utc)
    metadata = {
        'data_file': npz_path.name,
        'start_utc': start_utc.isoformat(),
        'end_utc': end_utc.isoformat(),
        'elapsed_seconds': elapsed_seconds,
        'N': args.N,
        'H_iters': args.H_iters,
        'state_iters': args.state_iters,
        'beta_values': json.dumps(args.b),
        'time_values': json.dumps(args.t),
        'seed_requested': args.seed if args.seed is not None else '',
        'seed_used': seed_used,
        'shell_matrices': not args.no_shell,
        'mpi_world_size': SIZE,
        'mpi_rank_range': f'0-{SIZE - 1}',
        'sample_groups': group_count,
        'node_count': len(unique_hosts),
        'node_names': json.dumps(unique_hosts),
        'slurm_job_id': os.environ.get('SLURM_JOB_ID', ''),
        'slurm_job_name': os.environ.get('SLURM_JOB_NAME', ''),
        'slurm_nodes_requested': os.environ.get('SLURM_JOB_NUM_NODES', ''),
        'slurm_tasks_per_node': os.environ.get('SLURM_TASKS_PER_NODE', ''),
        'slurm_cpus_per_task': os.environ.get('SLURM_CPUS_PER_TASK', ''),
        'peak_rss_sum_over_rank_peaks_gb': peak_rss_sum_gb,
        'peak_rss_max_single_rank_gb': peak_rss_max_rank_gb,
        'petsc_peak_sum_over_rank_peaks_gb': petsc_peak_sum_gb,
        'petsc_peak_max_single_rank_gb': petsc_peak_max_rank_gb,
        'python_version': platform.python_version(),
        'numpy_version': np.__version__,
        'dynamite_petsc_slepc_versions': get_version_str(),
        'hostname_of_rank_0': MPI.Get_processor_name(),
        'working_directory': os.getcwd(),
        'command_line_output_directory': args.output_dir,
    }
    with csv_path.open('w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=metadata.keys())
        writer.writeheader()
        writer.writerow(metadata)

    root_print(f'Saved numerical results to {npz_path}', file=stderr)
    root_print(f'Saved run metadata to {csv_path}', file=stderr)


def compute_two_point(beta_state, t, beta, H, chi_ops, N,
                      even_space, odd_space, group_comm, group_rank, group_size):
    """Compute G(t), distributing the Majorana sum within one rank group."""
    tau_ket = -1j * beta / 2 + 2 * t
    tau_bra = 1j * beta / 2 + 2 * t

    tmp_even = State(subspace=even_space)
    tmp_odd = State(subspace=odd_space)
    ket = State(subspace=odd_space)
    bra = State(subspace=odd_space)

    local_G_t = 0.0 + 0j

    # Independent of j; computed once on each rank's local Dynamite instance.
    H.evolve(beta_state, t=tau_bra, result=tmp_even)

    # Static cyclic scheduling balances Majorana terms within this sample group.
    for j in range(group_rank, len(chi_ops), group_size):
        chi_j = chi_ops[j]

        chi_j.dot(beta_state, result=tmp_odd)
        H.evolve(tmp_odd, t=tau_ket, result=ket)

        chi_j.dot(tmp_even, result=bra)
        local_G_t += bra.dot(ket)

    G_t = group_comm.allreduce(local_G_t, op=MPI.SUM)
    return G_t / N


def build_hamiltonian(N, rng):
    majoranas = [majorana(i) for i in range(N)]

    def gen_products():
        for idxs in combinations(range(N), 4):
            p = op_product(majoranas[idx] for idx in idxs)
            p.scale(rng.normal())
            yield p

    H = op_sum(gen_products())
    H.scale(np.sqrt(6 / N**3))
    return H


def get_shared_seed(requested_seed):
    """Choose one seed on rank zero and broadcast it to every MPI rank."""
    if RANK == 0:
        if requested_seed is None:
            from random import SystemRandom
            seed = SystemRandom().randrange(2**32)
        else:
            seed = requested_seed
    else:
        seed = None
    return COMM.bcast(seed, root=0)


def parse_args():
    parser = ArgumentParser(
        description='Compute the SYK two-point function with MPI over chi_j.'
    )
    parser.add_argument('-N', default=30, type=int,
                        help='number of Majorana fermions')
    parser.add_argument('-b', default=[0.0],
                        type=lambda s: [float(x) for x in s.split(',')],
                        help='comma-separated list of beta values')
    parser.add_argument('-t', default=[0.0],
                        type=lambda s: [float(x) for x in s.split(',')],
                        help='comma-separated list of real times t')
    parser.add_argument('--H-iters', default=1, type=int,
                        help='number of Hamiltonian disorder realizations')
    parser.add_argument('--state-iters', default=1, type=int,
                        help='number of random states per Hamiltonian')
    parser.add_argument('-s', '--seed', type=lambda x: int(x, 0),
                        help='RNG seed (random if omitted)')
    parser.add_argument('--no-shell', action='store_true',
                        help='disable shell matrices')
    parser.add_argument('--output-dir', default='results',
                        help='directory for NPZ data and CSV run metadata')
    return parser.parse_args()


if __name__ == '__main__':
    main()
