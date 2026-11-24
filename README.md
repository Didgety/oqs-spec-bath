# Reduced dynamics of a qubit coupled to a coherent spectator

Jai Veilleux

On IBM Fez, I followed a central qubit as it transferred an excitation to a spectator and recovered it. In all three jobs, the trace distance between the ground- and excited-input trajectories increased from transfer to return in both no-reset sweeps and decreased over the same interval in the reset control. For a plus-state input, the measured coherence reversed sign at the return.

[Paper](experiment/paper/paper.pdf) · [Analysis notebook](experiment/analysis.ipynb) · [Measured counts](experiment/data/counts.csv) · [Experiment details](experiment/README.md)

The paper derives the exact reduced channel and its memory kernel, compares time-local approximations from the same Hamiltonian, and tests predictions fixed by circuit angles and reported hardware timescales. Once the dependencies are installed, the bundled counts and calibration reproduce the results offline.

```sh
python -m pip install -r experiment/requirements.txt
python experiment/analyze.py
```
