"""Build the shared notebook figure and LaTeX tables from the experiment analysis."""

import matplotlib.pyplot as plt
import numpy as np
from design import SWEEPS

SWEEP_STYLES = {
    "small": (r"$\theta=\pi/16$, no-reset", "#0072B2", "o"),
    "large": (r"$\theta=\pi/4$, no-reset", "#D55E00", "s"),
    "reset": (r"$\theta=\pi/4$, reset", "#008358", "^"),
}


def plot_experiment(tables):
    """Plot measured populations, distinguishability, and coherence.

    Parameters
    ----------
    tables : dict
        Analysis output from ``analyze.analyze`` containing ``states``
        and ``trace_distances``. Points are means across jobs and the
        error bars span the observed job range.

    Returns
    -------
    matplotlib.figure.Figure
        Three panels with ideal reset and no-reset curves for reference.
        The caller can display, save, or close the figure.
    """
    states = tables["states"]
    datasets = (
        ([row for row in states if row["input"] == "excited"], "p1", r"Population $p_1$"),
        (tables["trace_distances"], "distance", r"Trace distance $D$"),
        ([row for row in states if row["input"] == "plus"], "x", r"Coherence $\langle X\rangle$"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.12), layout="constrained")
    angle = np.linspace(0, np.pi, 300)
    for ax, (rows, column, ylabel), panel in zip(axes, datasets, ("a", "b", "c")):
        order = 1 if column == "x" else 2
        ax.plot(angle / np.pi, np.cos(angle)**order, color="0.2", lw=1.2,
                label="Ideal no-reset")
        reset_n = np.arange(5)
        ax.plot(reset_n / 4, np.cos(np.pi / 4)**(order * reset_n), color="0.45",
                ls="--", lw=1.1, label="Ideal reset")
        for sweep, (label, color, marker) in SWEEP_STYLES.items():
            theta, blocks = SWEEPS[sweep]
            values = np.array([[row[column] for row in rows
                                if row["sweep"] == sweep and row["n"] == n]
                               for n in blocks])
            average = values.mean(axis=1)
            ranges = np.array([average - values.min(axis=1), values.max(axis=1) - average])
            ax.errorbar(np.array(blocks) * theta / np.pi, average, yerr=ranges,
                        color=color, marker=marker, ms=3.8, lw=0.8, capsize=2,
                        mfc="white", ls="none", label=label)
        ax.axvline(0.5, color="0.7", lw=0.8, ls=":")
        ax.set(xlim=(-0.035, 1.035), ylim=((-1.07, 1.07) if column == "x" else (-0.05, 1.05)),
               xlabel=r"Programmed angle $u/\pi$", ylabel=ylabel,
               xticks=[0, 0.25, 0.5, 0.75, 1])
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(f"({panel})", loc="left", fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=5, frameon=False,
               fontsize=8, handletextpad=0.4, columnspacing=1.15)
    return fig


def write_table(path, columns, header, data):
    """Write a booktabs tabular fragment for inclusion in the paper.

    Parameters
    ----------
    path : Path
        Destination .tex file.
    columns : str
        LaTeX column specifier, for example ``"rccc"``.
    header : str
        Header cells already joined by LaTeX's ampersand separators.
    data : iterable of iterable of str
        Cell strings grouped by row. Cells are written as supplied, so
        LaTeX markup and special-character escaping belong to the caller.
    """
    lines = [rf"\begin{{tabular}}{{@{{}}{columns}@{{}}}}", r"\toprule",
             header + r" \\", r"\midrule"]
    lines.extend(" & ".join(row) + r" \\" for row in data)
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(tables, plan, output):
    """Generate the figure and LaTeX table inputs for the bundled experiment.

    Parameters
    ----------
    tables : dict
        Analysis tables from ``analyze.analyze``, including witness and
        model scores for each job. The paper's model table uses means of
        those job-level scores.
    plan : dict
        Run plan whose ``repeats`` field determines the job rows in the
        witness table.
    output : Path
        Existing directory for exchange.pdf, witness.tex, and scores.tex.
    """
    figure = plot_experiment(tables)
    figure.savefig(output / "exchange.pdf", metadata={"CreationDate": None})
    plt.close(figure)

    witness = {(row["repeat"], row["sweep"]): row for row in tables["witness"]}
    witness_rows = []
    for repeat in range(1, plan["repeats"] + 1):
        cells = [str(repeat)]
        for sweep in SWEEPS:
            row = witness[repeat, sweep]
            cells.append("$" + f"{row['increase']:.3f}\\;[{row['lo95']:.3f},{row['hi95']:.3f}]$")
        witness_rows.append(cells)
    write_table(output / "witness.tex", "rccc",
                r"Job & No-reset $\pi/16$ & No-reset $\pi/4$ & Reset $\pi/4$",
                witness_rows)

    scores = tables["model_scores"]
    score_rows = []
    for sweep, label, window, model in (
        ("small", r"$\pi/16$", "all observed", "exact-exchange"),
        ("small", r"$\pi/16$", "all observed", "calibrated-block-noise"),
        ("large", r"$\pi/4$", "all observed", "exact-exchange"),
        ("large", r"$\pi/4$", "all observed", "calibrated-block-noise"),
        ("small", r"$\pi/16$", "u <= pi/4", "exact-exchange"),
        ("small", r"$\pi/16$", "u <= pi/4", "spectator-TCL2"),
        ("small", r"$\pi/16$", "u <= pi/4", "spectator-TCL4"),
        ("large", r"$\pi/4$", "u <= pi/4", "exact-exchange"),
        ("large", r"$\pi/4$", "u <= pi/4", "spectator-TCL2"),
        ("large", r"$\pi/4$", "u <= pi/4", "spectator-TCL4"),
        ("reset", r"$\pi/4$ reset", "all observed", "fresh-spectator"),
    ):
        pretty_name = {
            "exact-exchange": "Ideal exchange", "calibrated-block-noise": "Scheduled local noise",
            "spectator-TCL2": "Spectator TCL2", "spectator-TCL4": "Spectator TCL4",
            "fresh-spectator": "Ideal reset",
        }[model]
        selection = [row for row in scores if row["sweep"] == sweep
                     and row["window"] == window and row["model"] == model]
        values = [f"{np.mean([row['value'] for row in selection if row['metric'] == metric]):.4f}"
                  for metric in ("mean_trace_distance", "population_mae")]
        score_rows.append([label, "Full" if window == "all observed" else r"$u\le\pi/4$",
                           pretty_name, *values])
    write_table(output / "scores.tex", "lllrr",
                r"Block angle & Window & Model & Mean $D$ & Population MAE", score_rows)
