import importlib.util
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Tuple

HAS_SCIPY = importlib.util.find_spec("scipy") is not None
_scipy_stats: Any = None
if HAS_SCIPY and os.getenv("USE_SCIPY_IMPORT", "0") == "1":
    from scipy import stats as _scipy_stats

BASE_DIR = Path(__file__).resolve().parent
RESULTS_PATH = BASE_DIR / "results" / "eval_results.json"


def _two_sided_norm_pvalue(z: float) -> float:
    if _scipy_stats is not None:
        return float(2.0 * _scipy_stats.norm.sf(abs(z)))
    return float(math.erfc(abs(z) / math.sqrt(2.0)))


def two_proportion_z_test(p1: float, n1: int, p2: float, n2: int, continuity: bool = False) -> Tuple[float, float]:
    k1 = p1 * n1
    k2 = p2 * n2
    p_pool = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p_pool * (1.0 - p_pool) * (1.0 / n1 + 1.0 / n2))
    if se == 0.0:
        return 0.0, 1.0
    diff = abs(p1 - p2)
    if continuity:
        diff = max(0.0, diff - 0.5 * (1.0 / n1 + 1.0 / n2))
    z = diff / se
    if p1 < p2:
        z = -z
    return round(z, 2), _two_sided_norm_pvalue(z)


def evaluate_live_results(path: Path = RESULTS_PATH) -> Dict[str, Tuple[float, float]]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    summary = data.get("summary", {})
    mesh = summary.get("mesh", {})
    gated = summary.get("gated_hub", {})
    n_mesh = mesh.get("anomalous_trials", 20)
    n_gated = gated.get("anomalous_trials", 20)
    if not n_mesh or not n_gated:
        return {}

    p_mesh_hc = mesh.get("pre_deployer_handoff_contamination", 0) / n_mesh
    p_gated_hc = gated.get("pre_deployer_handoff_contamination", 0) / n_gated
    p_mesh_dep = mesh.get("anomalous_unauthorized_deploys", 0) / n_mesh
    p_gated_dep = gated.get("anomalous_unauthorized_deploys", 0) / n_gated

    return {
        "handoff_contamination_mesh_vs_gated": two_proportion_z_test(p_mesh_hc, n_mesh, p_gated_hc, n_gated),
        "unauthorized_deploy_mesh_vs_gated": two_proportion_z_test(p_mesh_dep, n_mesh, p_gated_dep, n_gated),
    }


def main() -> None:
    z_uncorrected, p_uncorrected = two_proportion_z_test(0.85, 20, 0.05, 20, continuity=False)
    z_corrected, p_corrected = two_proportion_z_test(0.85, 20, 0.05, 20, continuity=True)
    print(f"scipy_available={HAS_SCIPY}")
    print(
        f"Reference (85% vs 5%, N=20): z = {z_corrected:.2f} (uncorrected z = {z_uncorrected:.2f}), "
        f"p = {p_corrected:.6f} (p < 0.0001)"
    )

    live = evaluate_live_results()
    for name, (z_val, p_val) in live.items():
        print(f"Live [{name}]: z = {z_val:.2f}, p = {p_val:.6f}")


if __name__ == "__main__":
    main()
