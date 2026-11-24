"""Score fixed spectator models and compute the two-input witness from a collected run."""

import argparse
import csv
import math

import numpy as np
from design import BASES, INPUTS, ROOT, SWEEPS, read_json, run_path

INITIAL_BLOCH = {
    "ground": np.array([0.0, 0.0, 1.0]),
    "excited": np.array([0.0, 0.0, -1.0]),
    "plus": np.array([1.0, 0.0, 0.0]),
}
PAULIS = (
    np.array([[0, 1], [1, 0]]),
    np.array([[0, -1j], [1j, 0]]),
    np.diag([1, -1]),
)


def amplitude_channel(initial, amplitude):
    """Apply the ideal reduced channel for a real survival amplitude.

    Parameters
    ----------
    initial : array-like of shape (3,)
        Central-qubit Bloch vector in X, Y, Z order.
    amplitude : float
        Real excitation survival amplitude in [-1, 1]. Its square scales
        the excited-state population. Its sign is retained in the X and Y
        coordinates, so a negative value reverses the coherence.

    Returns
    -------
    numpy.ndarray of shape (3,)
        Output Bloch vector in X, Y, Z order.
    """
    x, y, z = initial
    return np.array([amplitude * x, amplitude * y, 1 - amplitude**2 * (1 - z)])


def physical_bloch(vectors):
    """Project linear-inversion Bloch vectors onto the physical unit ball.

    ``vectors`` has shape (..., 3) with X, Y, Z along the last axis. The
    output has the same shape. Vectors with norm above one are rescaled to
    unit length. Vectors already inside the ball pass through unchanged.
    """
    radius = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(1, radius)


def relaxation_and_dephasing(duration, t1, t2):
    """Construct Kraus operators for local relaxation and pure dephasing.

    Parameters
    ----------
    duration : float
        Noise interval in seconds.
    t1 : float
        Energy relaxation time in seconds.
    t2 : float
        Total transverse coherence time in seconds. Since T2 already includes
        relaxation, the pure-dephasing rate is 1/T2 minus 1/(2 T1).

    Returns
    -------
    list of numpy.ndarray
        Four 2-by-2 matrices for the composed zero-temperature channel.

    Raises
    ------
    ValueError
        T2 exceeds 2 T1 and would give a negative pure-dephasing rate.
    """
    pure_dephasing = 1 / t2 - 1 / (2 * t1)
    if pure_dephasing < 0:
        raise ValueError("Inconsistent calibration for this model: T2 exceeds 2 T1.")
    survival = math.exp(-duration / t1)
    phase = math.exp(-duration * pure_dephasing)
    damping = (np.diag([1, np.sqrt(survival)]),
               np.array([[0, np.sqrt(-math.expm1(-duration / t1))], [0, 0]]))
    dephasing = (np.sqrt((1 + phase) / 2) * np.eye(2),
                 np.sqrt((1 - phase) / 2) * PAULIS[2])
    return [p @ a for p in dephasing for a in damping]


def calibrated_exchange(initial, theta, block_times, snapshot):
    """Apply no-reset exchange with local noise before and after each ideal block.

    Parameters
    ----------
    initial : array-like of shape (3,)
        Initial central Bloch vector in X, Y, Z order. The spectator begins
        in its ground state.
    theta : float
        Exchange angle per block in radians.
    block_times : iterable of float
        Scheduled block durations in seconds. For each block, half of its
        duration is used for noise before the ideal unitary and half after.
    snapshot : dict
        Execution-time calibration with ``snapshot["qubits"]`` entries
        for ``central`` and ``spectator``. Each entry supplies ``t1_s``
        and ``t2_s`` in seconds.

    Returns
    -------
    numpy.ndarray of shape (3,)
        Reduced central-qubit Bloch vector after the last block.
    """
    central = (np.eye(2) + sum(r * matrix for r, matrix in zip(initial, PAULIS))) / 2
    state = np.kron(central, np.diag([1, 0]))
    unitary = np.eye(4, dtype=complex)
    unitary[1:3, 1:3] = [[np.cos(theta), -1j * np.sin(theta)],
                         [-1j * np.sin(theta), np.cos(theta)]]
    qubits = snapshot["qubits"]
    for duration in block_times:
        channels = [
            relaxation_and_dephasing(duration / 2, qubits[role]["t1_s"], qubits[role]["t2_s"])
            for role in ("central", "spectator")
        ]
        kraus = [np.kron(a, b) for a in channels[0] for b in channels[1]]
        state = sum(operator @ state @ operator.conj().T for operator in kraus)
        state = unitary @ state @ unitary.conj().T
        state = sum(operator @ state @ operator.conj().T for operator in kraus)
    return np.array([np.trace(state @ np.kron(matrix, np.eye(2))).real for matrix in PAULIS])


def predictions(rows, plan, snapshot):
    """Yield fixed model predictions grouped by their scoring window.

    Parameters
    ----------
    rows : list of dict
        Nonempty list of circuit metadata for one sweep, with one row per
        block count and input preparation. Each row gives the block angle,
        block count, input label, sweep label, and scheduled durations.
    plan : dict
        Run plan with ``dt_s``, the number of seconds per backend clock
        tick. It converts each row's ``block_durations_dt`` to seconds.
    snapshot : dict or None
        Execution calibration for the current repeat. A missing snapshot
        skips the calibrated-noise predictions.

    Yields
    ------
    tuple
        Model name, boolean mask of shape (len(rows),), and Bloch vectors
        of shape (len(rows), 3) in input row order. The mask selects the
        full or early window. Both TCL models use u = n * theta <= pi/4.
        The reset sweep yields the ideal fresh-spectator model.
    """
    theta = np.array([row["theta"] for row in rows])
    blocks = np.array([row["n"] for row in rows])
    angles = theta * blocks
    inputs = [INITIAL_BLOCH[row["input"]] for row in rows]
    if rows[0]["sweep"] == "reset":
        reset = [amplitude_channel(initial, amplitude)
                 for initial, amplitude in zip(inputs, np.cos(theta)**blocks)]
        yield "fresh-spectator", np.ones(len(rows), dtype=bool), np.array(reset)
        return

    early = angles <= math.pi / 4
    exact = np.array([amplitude_channel(initial, amplitude)
                      for initial, amplitude in zip(inputs, np.cos(angles))])
    yield "exact-exchange", np.ones(len(rows), dtype=bool), exact
    yield "exact-exchange", early, exact
    for label, amplitude in (
        ("spectator-TCL2", np.exp(-angles**2 / 2)),
        ("spectator-TCL4", np.exp(-angles**2 / 2 - angles**4 / 12)),
    ):
        yield label, early, np.array([amplitude_channel(initial, value)
                                     for initial, value in zip(inputs, amplitude)])
    if snapshot is not None:
        calibrated = np.array([
            calibrated_exchange(initial, row["theta"],
                                np.array(row["block_durations_dt"]) * plan["dt_s"], snapshot)
            for row, initial in zip(rows, inputs)
        ])
        yield "calibrated-block-noise", np.ones(len(rows), dtype=bool), calibrated
        yield "calibrated-block-noise", early, calibrated


def load_counts(folder, plan):
    """Read one-bit counts and check that each planned circuit has a result.

    ``folder`` contains counts.csv. ``plan`` supplies the expected circuit
    IDs, repeat count, and shots per circuit. The returned dictionary maps
    (one-based repeat, circuit ID) to the number of outcome-1 shots.
    """
    with (folder / "counts.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    counts = {}
    for row in rows:
        key = int(row["repeat"]), row["circuit_id"]
        n0, n1, shots = (int(row[column]) for column in ("count_0", "count_1", "shots"))
        if key in counts or min(n0, n1) < 0 or n0 + n1 != shots or shots != plan["shots"]:
            raise ValueError(f"Invalid counts for {key}.")
        counts[key] = n1
    expected = {(repeat, row["id"]) for repeat in range(1, plan["repeats"] + 1)
                for row in plan["circuits"]}
    if counts.keys() != expected:
        raise ValueError("Counts do not match the prepared run.")
    return counts


def interval(draws):
    """Return percentile bounds lo95 and hi95 for a vector of bootstrap draws."""
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return {"lo95": lo, "hi95": hi}


def write_table(path, rows):
    """Write a nonempty sequence of row dictionaries to the specified CSV path.

    The first row's keys define the column names and their order.
    """
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(folder, bootstrap=2000, *, output=None):
    """Reconstruct the central qubit and score fixed models separately by job.

    Parameters
    ----------
    folder : Path
        Directory containing counts.csv, plan.json, and
        execution_calibration.json. Calibration entries are keyed by the
        string form of each one-based repeat index.
    bootstrap : int
        Number of binomial resamples per job. Every model for a given
        sweep is evaluated on the same resampled counts.
    output : Path or None
        Optional directory for the CSV tables. Pass None to return the
        tables in memory only.

    Returns
    -------
    dict of list of dict
        ``states`` holds raw linear-inversion Bloch vectors and Z-basis
        populations. ``trace_distances`` holds the ground-versus-excited
        distance at each block count. ``witness`` holds its change from
        u = pi/2 to u = pi. ``matched_angles`` holds small-angle minus
        large-angle population contrasts at common angles. ``model_scores``
        holds mean trace distances and population mean absolute errors.
        The contrast and score tables include per-job 95% shot intervals.
    """
    plan = read_json(folder / "plan.json")
    hardware_calibration = read_json(folder / "execution_calibration.json")
    counts = load_counts(folder, plan)
    rng = np.random.default_rng(plan["seed"])
    shots = plan["shots"]
    metadata = {(row["sweep"], row["n"], row["input"], row["basis"]): row for row in plan["circuits"]}
    states, distances, witnesses, contrasts, scores = [], [], [], [], []
    for repeat in range(1, plan["repeats"] + 1):
        measured_states, shot_draws = {}, {}
        snapshot = hardware_calibration.get(str(repeat))
        if snapshot is not None and any(
            qubit["t2_s"] > 2 * qubit["t1_s"] for qubit in snapshot["qubits"].values()
        ):
            print(f"Repeat {repeat}: skipped the calibrated-noise model because T2 exceeds 2 T1.")
            snapshot = None
        if snapshot is None and plan["source"] == "hardware":
            print(f"Repeat {repeat}: fixed ideal predictions will be scored without a calibration model.")
        for sweep, (_, block_counts) in SWEEPS.items():
            rows = [metadata[sweep, n, preparation, "Z"] for n in block_counts for preparation in INPUTS]
            ones = np.array([[counts[repeat, metadata[sweep, row["n"], row["input"], basis]["id"]]
                             for basis in BASES] for row in rows])
            observed = 1 - 2 * ones / shots
            draws = 1 - 2 * rng.binomial(shots, ones / shots, size=(bootstrap, *ones.shape)) / shots
            for index, row in enumerate(rows):
                key = sweep, row["n"], row["input"]
                x, y, z = observed[index]
                measured_states[key], shot_draws[key] = observed[index], draws[:, index]
                states.append({"repeat": repeat, "sweep": sweep, "n": row["n"], "input": row["input"],
                                   "angle": row["n"] * row["theta"], "x": x, "y": y, "z": z, "p1": (1 - z) / 2,
                                   "outside_bloch_ball": np.linalg.norm(observed[index]) > 1})
            for model, mask, expected in predictions(rows, plan, snapshot):
                point = physical_bloch(observed[mask]) - expected[mask]
                residuals = physical_bloch(draws[:, mask]) - expected[mask]
                population_error = np.abs(observed[mask, 2] - expected[mask, 2]) / 2
                sampled_population_error = np.abs(draws[:, mask, 2] - expected[mask, 2]) / 2
                for metric, value, sampled in (
                    ("mean_trace_distance", np.linalg.norm(point, axis=-1).mean() / 2,
                     np.linalg.norm(residuals, axis=-1).mean(axis=-1) / 2),
                    ("population_mae", population_error.mean(), sampled_population_error.mean(axis=-1)),
                ):
                    window = "all observed" if mask.all() else "u <= pi/4"
                    scores.append(dict(repeat=repeat, sweep=sweep, model=model, window=window,
                                       metric=metric, value=value, **interval(sampled)))

        def distance(sweep, n, trajectories):
            """Compare the two input states for one sweep and block count.

            The trajectory mapping is keyed by (sweep, n, input). Each
            value is a Bloch vector or an array of bootstrap Bloch vectors.
            """
            a = physical_bloch(trajectories[sweep, n, "ground"])
            b = physical_bloch(trajectories[sweep, n, "excited"])
            return np.linalg.norm(a - b, axis=-1) / 2

        for sweep, (_, block_counts) in SWEEPS.items():
            for n in block_counts:
                distances.append({"repeat": repeat, "sweep": sweep, "n": n,
                                      "distance": distance(sweep, n, measured_states)})

        for sweep, first, last in (("small", 8, 16), ("large", 2, 4), ("reset", 2, 4)):
            d_first, d_last = (distance(sweep, n, measured_states) for n in (first, last))
            change = distance(sweep, last, shot_draws) - distance(sweep, first, shot_draws)
            witnesses.append(dict(repeat=repeat, sweep=sweep, transfer_n=first, return_n=last,
                                  distance_at_transfer=d_first, distance_at_return=d_last,
                                  increase=d_last - d_first, **interval(change)))
        for small, large in ((4, 1), (8, 2), (12, 3), (16, 4)):
            a, b = ("small", small, "excited"), ("large", large, "excited")
            change = (shot_draws[b][..., 2] - shot_draws[a][..., 2]) / 2
            contrasts.append(dict(repeat=repeat, angle=small * math.pi / 16,
                                  small_n=small, large_n=large,
                                  p1_difference=(measured_states[b][2] - measured_states[a][2]) / 2,
                                  **interval(change)))

    tables = {"states": states, "trace_distances": distances, "witness": witnesses,
                  "matched_angles": contrasts, "model_scores": scores}
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        for name, rows in tables.items():
            write_table(output / f"{name}.csv", rows)
    return tables


def main():
    """Analyze the bundled data or a named run from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", nargs="?", help="Name of a collected run (default: bundled IBM Fez data)")
    parser.add_argument("--bootstrap", type=int, default=2000,
                        help="Shot resamples per job, at least 100 (default %(default)s)")
    args = parser.parse_args()
    if args.bootstrap < 100:
        parser.error("use at least 100 shot resamples")
    try:
        folder = run_path(args.name) if args.name else ROOT / "data"
        output = folder if args.name else ROOT / "results"
        tables = analyze(folder, args.bootstrap, output=output)
        if not args.name:
            from report import write_report

            write_report(tables, read_json(folder / "plan.json"), output)
        print(f"Analyzed {read_json(folder / 'plan.json')['source']} data by repeat "
              f"with {args.bootstrap} shot resamples. Results: {output}")
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
