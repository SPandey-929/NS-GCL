"""Paired subject-level statistical comparisons for LOSO results."""
import numpy as np
from scipy.stats import ttest_rel, wilcoxon


def paired_subject_tests(method_a, method_b, alternative="two-sided"):
    """Paired t-test and Wilcoxon signed-rank test across matched subjects.

    Inputs must contain one comparable score per identical held-out subject,
    in the same subject order. These tests do not correct for multiple
    comparisons and do not make correlated CV folds independent.
    """
    a = np.asarray(method_a, dtype=np.float64)
    b = np.asarray(method_b, dtype=np.float64)
    if a.ndim != 1 or b.shape != a.shape or a.size < 2:
        raise ValueError("Provide at least two paired subject-level scores.")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Scores must be finite.")
    if alternative not in ("two-sided", "less", "greater"):
        raise ValueError("Unsupported alternative hypothesis.")
    differences = a - b
    t_result = ttest_rel(a, b, alternative=alternative)
    if np.allclose(differences, 0):
        w_stat, w_p = 0.0, 1.0
    else:
        w_result = wilcoxon(a, b, alternative=alternative, zero_method="wilcox")
        w_stat, w_p = float(w_result.statistic), float(w_result.pvalue)
    sd = float(differences.std(ddof=1))
    effect = float(differences.mean() / sd) if sd > 0 else 0.0
    return {
        "n_subjects": int(a.size),
        "mean_difference_a_minus_b": float(differences.mean()),
        "std_difference": sd,
        "paired_t_statistic": float(t_result.statistic),
        "paired_t_p_value": float(t_result.pvalue),
        "wilcoxon_statistic": w_stat,
        "wilcoxon_p_value": w_p,
        "cohens_dz": effect,
        "alternative": alternative,
    }
