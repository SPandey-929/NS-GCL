import pytest
np = pytest.importorskip("numpy")
from training.statistics import paired_subject_tests


def test_paired_subject_statistics_return_both_tests():
    result = paired_subject_tests([0.8, 0.7, 0.9, 0.75], [0.7, 0.65, 0.8, 0.7])
    assert result["n_subjects"] == 4
    assert result["mean_difference_a_minus_b"] > 0
    assert 0 <= result["paired_t_p_value"] <= 1
    assert 0 <= result["wilcoxon_p_value"] <= 1


def test_paired_statistics_require_matched_subject_vectors():
    with pytest.raises(ValueError):
        paired_subject_tests([0.8], [0.7])
    with pytest.raises(ValueError):
        paired_subject_tests([0.8, 0.7], [0.7])
