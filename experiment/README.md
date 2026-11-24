# IBM Quantum spectator experiment

The [paper](paper/paper.pdf) and [notebook](analysis.ipynb) use 567 circuit count records from three jobs on IBM Fez. The [counts](data/counts.csv), [circuit plan](data/plan.json), and [calibration snapshots](data/execution_calibration.json) are the inputs to the offline analysis.

## Reproduce the results offline

From the repository root, run the following with Python 3.11 or later.

```sh
python -m pip install -r experiment/requirements.txt
python experiment/analyze.py
```

The analysis rebuilds the figures, tables, and per-job results under `experiment/results/` using 2,000 shot resamples per job. Once dependencies are installed, the analysis runs using from the bundled data.

To rebuild the paper after regenerating the figures and tables, run `latexmk -pdf paper.tex` from `experiment/paper/` with REVTeX 4.2 and BibTeX available.

The count records use local job ordinals and circuit labels. The circuit plan records logical settings, physical qubits, scheduled durations, and compiled operation counts. The calibration file records the data reported for execution. The run records with IBM job IDs and saved circuits stay in the local acquisition workspace.

## Reading the code

| Module | Role |
|---|---|
| [design.py](design.py) | Defines the sweep angles, block counts, input preparations, and Pauli basis order shared by the scripts. |
| [acquire.py](acquire.py) | Builds and schedules circuits, records the plan, and maps results back to circuit IDs. |
| [analyze.py](analyze.py) | Reconstructs central-qubit states, evaluates the fixed models, and computes per-job shot intervals. |
| [report.py](report.py) | Turns the analysis tables into the shared figure and the paper's LaTeX tables. |

Block angles are in radians. The plan stores block durations as backend clock ticks and gives the tick length in seconds as `dt_s`. The noise model uses durations and lifetimes in seconds. Bloch vectors use X, Y, Z order, and job repeats are numbered from one.

## Experimental design

The same central and spectator qubits are used throughout a run. Each circuit starts with the spectator in its ground state and the central qubit in the ground, excited, or plus state. The central qubit is measured in separate X, Y, and Z circuits after a specified number of exchange blocks. An exchange block applies RXX and then RYY with the listed argument.

| Sweep | Block argument | Block counts | Spectator |
|---|---|---|---|
| Small | π/16 | 0, 1, 2, 3, 4, 6, 8, 10, 12, 14, 16 | Carried across blocks |
| Large | π/4 | 0, 1, 2, 3, 4 | Carried across blocks |
| Reset | π/4 | 0, 1, 2, 3, 4 | Reset after every block, including the last |

Both no-reset sweeps reach the first ideal return at an accumulated angle of π. The common nonzero angles are π/4, π/2, 3π/4, and π. Ground and excited inputs give a trace-distance witness. The plus input tests the coherence prediction, including the sign change at the ideal return. The reset adds hardware operations and duration, which are recorded.

## Running the experiment

Run these commands from the project root using an activated Python 3.11 or later environment.
Tested with Python 3.14, Qiskit 2.3.0, and IBM Runtime 0.45.0.

If you haven't already, do:

```powershell
python -m pip install -r experiment/requirements.txt
```

Copy `.env.example` to `.env`. Set `IBM_QUANTUM_TOKEN` to your IBM Cloud API key. Set `IBM_QUANTUM_INSTANCE` to your instance CRN if needed. You can specify an available backend and ordered physical pair in the same file. With those options empty, preparation selects an available device and ranks directly connected pairs with T1/T2 data and reset support by reported entangling-gate error.

```powershell
python experiment/acquire.py prepare new-run
```

Preparation queries IBM for the backend and calibration, then compiles and schedules the circuits locally. Review the printed layout and `experiment/runs/new-run/plan.json`. Each of the three default repeats contains 189 circuits at 512 shots each. Submitting all three consumes 290,304 circuit executions before IBM-specific overhead. The plan reports estimated durations from the device target. Preparing it uses no QPU time.

Each submit command consumes QPU allocation. The three commands create three independent jobs on the same physical pair, with a different interleaved circuit order for each repeat.

```powershell
python experiment/acquire.py submit new-run --repeat 1
python experiment/acquire.py submit new-run --repeat 2
python experiment/acquire.py submit new-run --repeat 3
```

The job IDs are written locally. A repeated submit for the same index is rejected. If a submission times out before returning an ID, inspect the IBM job list before retrying. Keep the run directory so collection can use the saved plan and job order.

```powershell
python experiment/acquire.py collect new-run
python experiment/analyze.py new-run
```

Collection waits for the submitted jobs and saves one-bit counts and available execution-time calibration. Keep the entire run directory local. The analysis writes reconstructed states, four equal-angle population contrasts, trace-distance changes from transfer to return, and fixed-prediction scores. The state table retains raw linear inversion and flags estimates outside the Bloch ball. For trace distance, the analysis radially projects those estimates onto the Bloch ball and repeats the projection in every resample. Population scores use the Z-basis counts directly. It reports each repeat separately and uses a binomial shot bootstrap within the repeat. Repeat variation, preparation and readout bias, and uncertainty in calibration sit outside those intervals.

The witness uses the fixed endpoints $\pi$/2 and $\pi$ and computes the change in trace distance between the ground- and excited-input trajectories. An increase is evidence of a distinguishability revival under the tested implementations. Interpreting it as a channel witness assumes the two preparations experience the same subsequent channel, including any correlated or preparation-dependent errors. The plus input is a separate check on the coherence prediction. The experiment does not reconstruct a complete process.

## Local simulation

These commands compile with IBM's FakeFez snapshot and generate shot-sampled counts from the ideal logical circuits. The plan explicitly marks them as simulated.

```powershell
python experiment/acquire.py prepare rehearsal --fake
python experiment/acquire.py simulate rehearsal
python experiment/analyze.py rehearsal
```
