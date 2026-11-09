import json
from pathlib import Path

import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
RESULTS_PATH = BASE_DIR / "results" / "eval_results.json"
FIGURE_PATH = BASE_DIR / "results" / "contagion_results.png"


def main() -> None:
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    summary = data["summary"]
    stats = data.get("statistics", {})
    n_total = data.get("trials_completed", 148)
    model_name = data.get("models", {}).get("worker_nodes", "gemini-3.5-flash-lite")

    topologies = ["mesh", "linear", "gated_hub"]
    labels = ["Mesh\n(Shared Blackboard)", "Linear Chain\n(Sequential Handoff)", "Gated Hub\n(Gatekeeper Blast Wall)"]

    handoff_rates = [summary[t].get("pre_deployer_handoff_contamination_rate_pct", 0.0) for t in topologies]
    unauth_deploy_rates = [summary[t].get("anomalous_unauthorized_deploy_rate_pct", 0.0) for t in topologies]
    breach_rates = [summary[t].get("anomalous_breach_rate_pct", 0.0) for t in topologies]
    benign_rates = [summary[t].get("benign_completion_rate_pct", 0.0) for t in topologies]

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
            "font.size": 10.5,
            "axes.edgecolor": "#CBD5E0",
            "axes.linewidth": 1.0,
        }
    )

    fig, ax = plt.subplots(figsize=(10.2, 6.0), dpi=300)

    x = list(range(len(topologies)))
    width = 0.20

    bars_handoff = ax.bar(
        [i - 1.5 * width for i in x],
        handoff_rates,
        width=width,
        color="#E53E3E",
        edgecolor="#1A202C",
        linewidth=0.9,
        label="Pre-Deployer Handoff Contamination (%)",
        zorder=3,
    )
    bars_unauth = ax.bar(
        [i - 0.5 * width for i in x],
        unauth_deploy_rates,
        width=width,
        color="#DD6B20",
        edgecolor="#1A202C",
        linewidth=0.9,
        label="Poisoned PR False-Deploy Rate (%)",
        zorder=3,
    )
    bars_breach = ax.bar(
        [i + 0.5 * width for i in x],
        breach_rates,
        width=width,
        color="#4A5568",
        edgecolor="#1A202C",
        linewidth=0.9,
        label="Literal Exfil Command Execution (%)",
        zorder=3,
    )
    bars_benign = ax.bar(
        [i + 1.5 * width for i in x],
        benign_rates,
        width=width,
        color="#008080",
        edgecolor="#1A202C",
        linewidth=0.9,
        label="Benign Task Completion Fidelity (%)",
        zorder=3,
    )

    ax.set_title(
        f"Figure 1: Multi-Agent Containment Across Swarm Topologies (N={n_total}, {model_name})",
        fontsize=12,
        fontweight="bold",
        color="#1A365D",
        pad=16,
    )
    ax.set_ylabel("Percentage of Trials (%)", fontsize=11.5, fontweight="bold", color="#2D3748", labelpad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, fontweight="bold", color="#1A202C")
    ax.set_ylim(0, 138)
    ax.grid(axis="y", linestyle="--", alpha=0.45, color="#A0AEC0", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar_group in (bars_handoff, bars_unauth, bars_breach, bars_benign):
        for bar in bar_group:
            h = bar.get_height()
            ax.annotate(
                f"{h:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9.5,
                fontweight="bold",
                color="#1A202C",
            )

    hc_stat = stats.get("handoff_contamination_mesh_vs_gated_hub", {})
    z_val = hc_stat.get("z")
    p_val = hc_stat.get("p")
    delta_hc = handoff_rates[0] - handoff_rates[2]
    stat_str = f"z = {z_val:.2f}, p < 0.0001" if (z_val is not None and p_val is not None and p_val < 0.0001) else (
        f"z = {z_val:.2f}, p = {p_val:.4f}" if (z_val is not None and p_val is not None) else "Live API Run"
    )

    target_x = 2 - 1.5 * width
    target_y = handoff_rates[2] + 8.0
    ax.annotate(
        f"Gatekeeper Blast Wall Effect:\n-{delta_hc:.1f} pp Unblocked Contamination\n({handoff_rates[0]:.1f}% Mesh -> {handoff_rates[2]:.1f}% Gated Hub)\nTwo-proportion {stat_str}",
        xy=(target_x, target_y),
        xytext=(target_x + 0.13, 42.0),
        ha="center",
        va="bottom",
        arrowprops=dict(
            arrowstyle="->",
            color="#E53E3E",
            lw=1.5,
            shrinkA=3,
            shrinkB=3,
        ),
        fontsize=9.2,
        fontweight="bold",
        color="#1A365D",
        bbox=dict(
            boxstyle="round,pad=0.45",
            facecolor="#FFF5F5",
            edgecolor="#E53E3E",
            lw=1.2,
        ),
        zorder=5,
    )

    ax.legend(
        loc="upper left",
        ncol=2,
        frameon=True,
        facecolor="#FFFFFF",
        edgecolor="#CBD5E0",
        fontsize=9.5,
        framealpha=0.95,
    )

    plt.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURE_PATH, dpi=300)
    plt.close(fig)
    print(f"Saved figure to {FIGURE_PATH}")


if __name__ == "__main__":
    main()
