"""Prepare, submit, and collect spectator-qubit tomography runs."""

import argparse
import csv
import json
import math
import os
from datetime import UTC, datetime
from importlib.metadata import version

import numpy as np
from design import BASES, INPUTS, ROOT, SEED, SWEEPS, read_json, run_path
from dotenv import load_dotenv
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, qpy
from qiskit.quantum_info import DensityMatrix
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from qiskit_ibm_runtime.api.exceptions import RequestsApiError
from qiskit_ibm_runtime.exceptions import IBMError
from qiskit_ibm_runtime.models.exceptions import BackendPropertyError

READOUT = "central"


def utc_now():
    """Return the current time as an ISO 8601 string with a UTC offset."""
    return datetime.now(UTC).isoformat()


def write_json(path, value):
    """Serialize a value to a temporary sibling file before replacing the target.

    The target path is a Path. The value must be JSON-serializable and may
    contain only finite numeric values.
    """
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def service():
    """Create the IBM Runtime service from environment variables or the root .env.

    IBM_QUANTUM_TOKEN is required. IBM_QUANTUM_INSTANCE optionally selects an
    instance. Existing environment variables take precedence over the file.
    """
    load_dotenv(ROOT.parent / ".env")
    token = os.getenv("IBM_QUANTUM_TOKEN")
    if not token:
        raise ValueError("Set IBM_QUANTUM_TOKEN in the local .env file.")
    return QiskitRuntimeService(
        channel="ibm_quantum_platform",
        token=token,
        instance=os.getenv("IBM_QUANTUM_INSTANCE") or None,
    )


def design():
    """Yield one logical circuit specification per sweep, block, input, and basis.

    Each row has a unique circuit name in ``id``. The ``theta`` angle is in
    radians and ``n`` is the number of exchange blocks. ``input`` selects
    the central qubit's preparation and ``basis`` selects its readout axis.
    ``sweep`` is ``small``, ``large``, or ``reset``.
    """
    for sweep, (theta, block_counts) in SWEEPS.items():
        for n in block_counts:
            for preparation in INPUTS:
                for basis in BASES:
                    yield {
                        "id": f"{sweep}_{n:02d}_{preparation}_{basis}",
                        "sweep": sweep, "theta": theta, "n": n, "input": preparation, "basis": basis,
                    }


def circuit_for(row, *, measure=True):
    """Build the logical circuit for a single preparation and Pauli basis.

    Parameters
    ----------
    row : dict
        One specification from ``design``. The angle ``theta`` is the
        argument of both RXX and RYY in each of the ``n`` blocks.
    measure : bool
        Add a one-bit measurement of the central qubit when true. Setting
        this to false keeps the basis rotation for density-matrix simulation.

    Returns
    -------
    QuantumCircuit
        Logical q0 is the central qubit and q1 is the spectator. The reset
        sweep resets q1 after each exchange, before the block-end barrier.
    """
    circuit = QuantumCircuit(QuantumRegister(2, "q"), ClassicalRegister(1, READOUT),
                             name=row["id"])
    if row["input"] == "excited":
        circuit.x(0)
    elif row["input"] == "plus":
        circuit.h(0)
    circuit.barrier(0, 1, label="exchange_0")
    for block in range(1, row["n"] + 1):
        circuit.rxx(row["theta"], 0, 1)
        circuit.ryy(row["theta"], 0, 1)
        if row["sweep"] == "reset":
            circuit.reset(1)
        circuit.barrier(0, 1, label=f"exchange_{block}")
    if row["basis"] == "Y":
        circuit.sdg(0)
    if row["basis"] in ("X", "Y"):
        circuit.h(0)
    if measure:
        circuit.measure(0, 0)
    return circuit


def calibration(properties, pair):
    """Extract the calibration fields used by the two-qubit noise model.

    Parameters
    ----------
    properties : BackendProperties or None
        IBM backend calibration at a particular snapshot time.
    pair : tuple of int or list of int
        Two physical qubit indices in central, spectator order.

    Returns
    -------
    dict or None
        A timestamp and a ``qubits`` mapping keyed by ``central`` and
        ``spectator``. Each entry contains a physical index, T1 and T2 in
        seconds, and an optional readout error probability. Missing or
        invalid lifetimes on either qubit give None.
    """
    if properties is None:
        return None
    qubits = {}
    for role, qubit in zip(("central", "spectator"), pair):
        try:
            t1, t2 = properties.t1(qubit), properties.t2(qubit)
        except BackendPropertyError:
            return None
        if not all(math.isfinite(value) and value > 0 for value in (t1, t2)):
            return None
        readout_error = properties.qubit_property(qubit).get("readout_error", (None,))[0]
        qubits[role] = {"physical_index": qubit, "t1_s": t1, "t2_s": t2, "readout_error": readout_error}
    return {
        "calibration_at": properties.last_update_date.isoformat(),
        "qubits": qubits,
    }


def choose_pair(backend, override):
    """Select a directly connected physical pair with a calibrated entangling gate.

    Parameters
    ----------
    backend : IBMBackend or FakeFez
        Backend that supplies the gate target and qubit calibration.
    override : str or None
        An ordered pair such as ``"12,13"`` with the central qubit first.
        Without an override, candidates are ranked by reported two-qubit
        gate error.

    Returns
    -------
    tuple
        The selected pair and its calibration snapshot. The spectator
        must support reset and both qubits must have valid T1 and T2 values.
    """
    candidates = []
    for name in ("cz", "ecr", "cx"):
        if name in backend.target:
            for qubits, gate in backend.target[name].items():
                if (qubits is not None and gate is not None and gate.error is not None
                        and math.isfinite(gate.error)):
                    candidates.append((float(gate.error), qubits, name))
    if override:
        pair = tuple(int(value.strip()) for value in override.split(","))
        if len(pair) != 2 or pair[0] == pair[1]:
            raise ValueError("IBM_QUANTUM_QUBITS must contain two distinct physical qubits.")
        candidates = [item for item in candidates if frozenset(item[1]) == frozenset(pair)]
    else:
        pair = None
    properties = backend.properties()
    for _, qubits, _ in sorted(candidates):
        selected = pair or qubits
        if not backend.target.instruction_supported("reset", (selected[1],)):
            continue
        snapshot = calibration(properties, selected)
        if snapshot is not None:
            return selected, snapshot
    raise ValueError("No connected pair with a calibrated two-qubit gate, reset, and T1/T2.")


def compiled_metadata(row, circuit, backend, pair):
    """Attach scheduled durations and operation counts to a logical circuit row.

    Parameters
    ----------
    row : dict
        The matching logical circuit specification from ``design``.
    circuit : QuantumCircuit
        Scheduled physical circuit with ``exchange_0`` through
        ``exchange_n`` boundary barriers. The schedule includes spectator
        resets in the reset sweep.
    backend : IBMBackend or FakeFez
        Backend that supplied the target used to estimate the schedule.
    pair : tuple of int
        Expected physical indices in central, spectator order.

    Returns
    -------
    dict
        The logical fields plus each block duration in backend clock ticks,
        the complete scheduled circuit duration in seconds, and operation
        counts with delays and barriers omitted.

    Raises
    ------
    ValueError
        The compiled layout, block durations, or reset count disagree with
        the design.
    """
    if tuple(circuit.layout.final_index_layout(filter_ancillas=True)) != tuple(pair):
        raise ValueError(f"{row['id']} changed the physical qubit mapping.")
    markers = {
        instruction.operation.label: start
        for instruction, start in zip(circuit.data, circuit.op_start_times, strict=True)
        if instruction.operation.name == "barrier"
    }
    times = np.array([markers[f"exchange_{n}"] for n in range(row["n"] + 1)])
    durations = np.diff(times).astype(int)
    if np.any(durations <= 0):
        raise ValueError(f"{row['id']} has a missing or zero exchange-block duration.")
    gate_counts = {
        gate: count for gate, count in sorted(circuit.count_ops().items())
        if gate not in ("delay", "barrier")
    }
    if gate_counts.get("reset", 0) != (row["n"] if row["sweep"] == "reset" else 0):
        raise ValueError(f"{row['id']} has an unexpected number of resets.")
    return {**row, "block_durations_dt": durations.tolist(),
            "scheduled_duration_s": circuit.estimate_duration(backend.target, unit="s"),
            "compiled_operation_counts": gate_counts}


def remove_unused_delays(circuit, pair):
    """Copy the scheduled circuit with padding removed from inactive wires.

    ``circuit`` uses physical wire indices and ``pair`` lists the two active
    indices. Delays on those two qubits and all other operations are retained.
    """
    cleaned = circuit.copy_empty_like()
    for instruction in circuit.data:
        if (instruction.operation.name != "delay"
                or circuit.find_bit(instruction.qubits[0]).index in pair):
            cleaned.append(instruction)
    return cleaned


def prepare(args):
    """Compile a new run and write its plan and QPY file without submitting jobs.

    Parameters
    ----------
    args : argparse.Namespace
        ``name`` names a new run directory. The ``fake`` flag chooses
        FakeFez. ``qubits`` optionally supplies a comma-separated physical
        pair in central, spectator order. ``shots`` is the per-circuit shot
        count and ``repeats`` is the number of planned jobs.
    """
    folder = run_path(args.name)
    if folder.exists():
        raise ValueError(f"Run {args.name} already exists. Choose a new name.")
    if args.fake:
        from qiskit_ibm_runtime.fake_provider import FakeFez

        backend = FakeFez()
        pair, snapshot = choose_pair(backend, args.qubits)
    else:
        runtime = service()
        backend_name = os.getenv("IBM_QUANTUM_BACKEND")
        backend = runtime.backend(backend_name) if backend_name else runtime.least_busy(
            simulator=False, operational=True, min_num_qubits=2)
        backend.properties(refresh=True)
        pair, snapshot = choose_pair(backend, args.qubits or os.getenv("IBM_QUANTUM_QUBITS"))
    if backend.dt is None or backend.dt <= 0:
        raise ValueError("The backend does not report a valid scheduling clock.")
    rows = list(design())
    manager = generate_preset_pass_manager(
        backend=backend, initial_layout=list(pair), optimization_level=1, seed_transpiler=SEED,
        routing_method="none", scheduling_method="asap",
    )
    compiled = manager.run([circuit_for(row) for row in rows], num_processes=1)
    records = [compiled_metadata(row, circ, backend, pair)
               for row, circ in zip(rows, compiled, strict=True)]
    submitted_circuits = [remove_unused_delays(circuit, pair) for circuit in compiled]
    for circuit in submitted_circuits:
        backend.check_faulty(circuit)
    block_signatures = {}
    for row in records:
        key = row["sweep"], row["n"]
        signature = row["block_durations_dt"]
        if block_signatures.setdefault(key, signature) != signature:
            raise ValueError(f"Exchange-window timing differs across inputs or bases for {key}.")
    plan = {
        "schema_version": 1, "source": "fake" if args.fake else "hardware",
        "prepared_at_utc": utc_now(), "backend": backend.name, "physical_pair": list(pair),
        "shots": args.shots, "repeats": args.repeats, "seed": SEED, "dt_s": backend.dt,
        "calibration": snapshot, "scheduling": "asap",
        "qiskit": version("qiskit"), "qiskit_ibm_runtime": version("qiskit-ibm-runtime"),
        "circuits": records,
    }
    folder.mkdir(parents=True)
    with (folder / "circuits.qpy").open("wb") as stream:
        qpy.dump(submitted_circuits, stream)
    write_json(folder / "plan.json", plan)
    print(f"{backend.name}, central {pair[0]}, spectator {pair[1]}.")
    print(f"{len(records)} circuits per repeat x {args.shots} shots x {args.repeats} repeats.")
    print(f"Review {folder / 'plan.json'}. No jobs were submitted.")


def order_for(plan, repeat):
    """Return circuit indices in the deterministic submission order for a job.

    ``plan`` supplies the random seed and the ordered circuit list. The
    one-based ``repeat`` is added to the seed so collection can reconstruct
    the same permutation for that job.
    """
    return np.random.default_rng(plan["seed"] + repeat).permutation(len(plan["circuits"]))


def job_record_path(folder, repeat):
    """Return the local job ledger path in folder for a one-based repeat index."""
    return folder / f"job-{repeat}.json"


def submit(args):
    """Submit one repeat of a hardware plan and record the IBM job ID locally.

    ``args.name`` identifies the prepared run. ``args.repeat`` is a one-based
    job index within that run. A pending ledger entry is created before the
    Sampler call so repeated invocations for that index are rejected.
    """
    folder = run_path(args.name)
    plan = read_json(folder / "plan.json")
    if plan["source"] != "hardware" or not 1 <= args.repeat <= plan["repeats"]:
        raise ValueError("Select a valid repeat from a hardware plan.")
    with (folder / "circuits.qpy").open("rb") as stream:
        circuits = qpy.load(stream)
    order = order_for(plan, args.repeat)
    runtime = service()
    backend = runtime.backend(plan["backend"])
    sampler = SamplerV2(mode=backend)
    sampler.options.execution.init_qubits = True
    sampler.options.dynamical_decoupling.enable = False
    sampler.options.twirling.enable_gates = False
    sampler.options.twirling.enable_measure = False
    record_path = job_record_path(folder, args.repeat)
    pending = {"repeat": args.repeat, "status": "submitting", "submitted_at_utc": utc_now()}
    try:
        with record_path.open("x", encoding="utf-8") as stream:
            json.dump(pending, stream)
    except FileExistsError as exc:
        raise ValueError(f"{record_path.name} exists. Check the IBM jobs page before retrying.") from exc
    job = sampler.run([circuits[index] for index in order], shots=plan["shots"])
    job_id = job.job_id()
    print(f"Repeat {args.repeat} job ID {job_id}", flush=True)
    write_json(record_path, {**pending, "status": "submitted", "job_id": job_id})


def write_counts(path, plan, counts):
    """Save one-bit counts in repeat and circuit-plan order.

    Parameters
    ----------
    path : Path
        Destination CSV file.
    plan : dict
        Run plan supplying the expected shot count and the ordered circuit
        list used to resolve each index to a circuit ID.
    counts : iterable of tuple
        Each record contains a one-based repeat index, zero-based circuit
        index, and outcome counts keyed by the strings ``"0"`` and ``"1"``.
        A missing outcome contributes zero counts.
    """
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        fields = ("repeat", "circuit_id", "count_0", "count_1", "shots")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for repeat, index, counter in sorted(counts, key=lambda row: (row[0], row[1])):
            shots = sum(counter.values())
            if set(counter) - {"0", "1"} or shots != plan["shots"]:
                raise ValueError("Sampler result does not match the planned one-bit measurement.")
            writer.writerow({"repeat": repeat, "circuit_id": plan["circuits"][index]["id"],
                                 "count_0": counter.get("0", 0), "count_1": counter.get("1", 0), "shots": shots})
    temporary.replace(path)


def collect(args):
    """Retrieve all saved job results for the hardware run named by args.name.

    The saved permutation maps Sampler results back to circuit IDs. The run
    directory receives counts.csv and execution_calibration.json, with
    missing calibration recorded as None for the corresponding repeat.
    """
    folder = run_path(args.name)
    plan = read_json(folder / "plan.json")
    if plan["source"] != "hardware":
        raise ValueError("Use simulate with a fake plan.")
    jobs = [read_json(job_record_path(folder, repeat)) for repeat in range(1, plan["repeats"] + 1)]
    if any(record["status"] != "submitted" for record in jobs):
        raise ValueError("Each repeat needs a recorded IBM job ID before collection.")
    runtime = service()
    counts, snapshots = [], {}
    for record in jobs:
        repeat = record["repeat"]
        job = runtime.job(record["job_id"])
        results = job.result()
        for index, pub in zip(order_for(plan, repeat), results, strict=True):
            counts.append((repeat, index, getattr(pub.data, READOUT).get_counts()))
        try:
            snapshots[str(repeat)] = calibration(job.properties(), plan["physical_pair"])
        except (IBMError, RequestsApiError):
            snapshots[str(repeat)] = None
            print(f"Repeat {repeat}: IBM did not return execution-time calibration. Counts were kept.")
    write_counts(folder / "counts.csv", plan, counts)
    write_json(folder / "execution_calibration.json", snapshots)
    print(f"Collected {len(counts)} circuit results in {folder / 'counts.csv'}.")


def simulate(args):
    """Populate the fake run named by args.name with reproducible ideal counts.

    The logical circuit supplies each outcome probability. Counts are
    binomial draws using the run's shot count and seed, and every repeat's
    execution calibration is recorded as None.
    """
    folder = run_path(args.name)
    plan = read_json(folder / "plan.json")
    if plan["source"] != "fake":
        raise ValueError("Simulated counts can only be written to a fake run.")
    probabilities = [
        DensityMatrix.from_instruction(circuit_for(row, measure=False)).probabilities([0])[1]
        for row in plan["circuits"]
    ]
    generator = np.random.default_rng(plan["seed"])
    counts = []
    for repeat in range(1, plan["repeats"] + 1):
        draws = generator.binomial(plan["shots"], np.clip(probabilities, 0, 1))
        counts += [(repeat, index, {"0": plan["shots"] - int(n1), "1": int(n1)})
                   for index, n1 in enumerate(draws)]
    write_counts(folder / "counts.csv", plan, counts)
    write_json(folder / "execution_calibration.json",
               {str(repeat): None for repeat in range(1, plan["repeats"] + 1)})
    print(f"Simulated {len(counts)} circuit results. No hardware data were generated.")


def main():
    """Parse and dispatch the prepare, submit, collect, or simulate command."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(required=True)
    for action in (prepare, submit, collect, simulate):
        command = commands.add_parser(action.__name__, help=action.__doc__.splitlines()[0])
        command.add_argument("name", help="Local run name")
        command.set_defaults(action=action)
        if action is prepare:
            command.add_argument("--fake", action="store_true", help="Compile using FakeFez offline")
            command.add_argument("--qubits", help="Comma-separated physical indices, central first")
            command.add_argument("--shots", type=int, default=512,
                                 help="Shots per circuit (default %(default)s)")
            command.add_argument("--repeats", type=int, default=3,
                                 help="Number of jobs in the plan (default %(default)s)")
        if action is submit:
            command.add_argument("--repeat", type=int, required=True,
                                 help="One-based index of the job to submit")
    args = parser.parse_args()
    if args.action is prepare and min(args.shots, args.repeats) < 1:
        parser.error("shots and repeats must be positive")
    try:
        args.action(args)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
