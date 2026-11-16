import math

import pytest

from ragwrench.evalgate.formats import InputError
from ragwrench.evalgate.statistics import paired_test, t_two_sided


@pytest.mark.parametrize("method", ["bootstrap", "randomization", "ttest"])
def test_identical(method):
    result = paired_test([0.0] * 20, method, samples=1000)
    assert result["p_value"] == 1
    assert result["ci"] == [0, 0]


@pytest.mark.parametrize("method", ["bootstrap", "randomization", "ttest"])
def test_clearly_better(method):
    result = paired_test([0.2, 0.3, 0.4, 0.5] * 10, method, samples=1000)
    assert result["p_value"] < 0.01
    assert 0 < result["ci"][0] < result["ci"][1]
    if method != "ttest":
        assert result["p_value"] >= 1 / 1001


@pytest.mark.parametrize("method", ["bootstrap", "randomization"])
def test_seeded_reproducible_and_different_seed(method):
    data = [0.2, -0.1, 0.4, -0.3, 0.9, -0.05, 0.1]
    a = paired_test(data, method, seed=42, samples=500)
    b = paired_test(data, method, seed=42, samples=500)
    c = paired_test(data, method, seed=99, samples=500)
    assert a == b
    assert (a["p_value"], a["ci"]) != (c["p_value"], c["ci"])


@pytest.mark.parametrize("degrees,t,p", [(1, 1.0, 0.5), (2, 2.0, 1 - 2 / math.sqrt(6)), (10, 2.2281388519649385, 0.05), (30, 2.0422724563012373, 0.05), (100, 1.9839715184496334, 0.05)])
def test_t_distribution_reference_values(degrees, t, p):
    assert t_two_sided(t, degrees) == pytest.approx(p, abs=1e-10)
    assert t_two_sided(-t, degrees) == pytest.approx(p, abs=1e-10)
    assert t_two_sided(0, degrees) == 1


def test_t_test_known_mean_and_interval():
    result = paired_test([1, 2, 3, 4, 5], "ttest")
    assert result["p_value"] == pytest.approx(0.013235599563682695)
    assert result["ci"] == pytest.approx([1.036756838522439, 4.9632431614775605])


def test_t_test_constant_effect():
    result = paired_test([0.5] * 5, "ttest")
    assert result["p_value"] == 0
    assert result["ci"] == [0.5, 0.5]
    assert "zero paired variance" in result["warning"]


@pytest.mark.parametrize("kwargs", [{"samples": 0}, {"samples": 1000001}, {"confidence": 1}, {"confidence": float("nan")}, {"method": "nope"}])
def test_invalid_arguments(kwargs):
    with pytest.raises(InputError):
        paired_test([0.1, 0.2], **kwargs)


def test_small_n_nan_and_budget():
    for data in ([0.1], [0.1, float("nan")]):
        with pytest.raises(InputError):
            paired_test(data)
    with pytest.raises(InputError, match="budget"):
        paired_test([0.1] * 51, samples=1000000)


@pytest.mark.parametrize("correction,expected", [
    ("none", [.04, 1, .01, .03, .03]),
    ("bonferroni", [.2, 1, .05, .15, .15]),
    # Holm: sorted .01*5=.05, .03*4=.12, .03*3=.09, .04*2=.08, 1*1=1.
    # Cumulative maxima preserve monotonicity, including tied p-values.
    ("holm", [.12, 1, .05, .12, .12]),
])
def test_multiple_testing_adjustment(correction, expected):
    from ragwrench.evalgate.statistics import adjust_p_values
    values = [.04, 1, .01, .03, .03]
    assert adjust_p_values(values, correction) == pytest.approx(expected)
    assert adjust_p_values(list(reversed(values)), correction) == pytest.approx(list(reversed(expected)))
    assert adjust_p_values([0, 1], correction) == [0, 1]
    assert adjust_p_values([], correction) == []
    assert adjust_p_values([.03], correction) == [.03]


@pytest.mark.parametrize("values,correction", [([.1], "invalid"), ([float("nan")], "holm"),
                                                  ([float("inf")], "none"), ([-.1], "bonferroni"), ([1.1], "holm")])
def test_invalid_adjustment(values, correction):
    from ragwrench.evalgate.statistics import adjust_p_values
    with pytest.raises(InputError):
        adjust_p_values(values, correction)
