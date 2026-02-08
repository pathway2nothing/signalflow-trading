"""Unit tests for mutual information computation functions."""

import numpy as np
import pytest

from signalflow.feature.mutual_information import (
    entropy_continuous,
    entropy_discrete,
    mutual_information_continuous,
    mutual_information_continuous_discrete,
    mutual_information_discrete,
    normalized_mutual_information,
)


class TestEntropyDiscrete:
    def test_uniform_distribution(self):
        """Uniform distribution over N values should have entropy log2(N)."""
        x = np.array([0, 1, 2, 3] * 250)
        h = entropy_discrete(x)
        assert h == pytest.approx(2.0, abs=0.01)  # log2(4) = 2.0

    def test_single_value_zero_entropy(self):
        """Constant distribution should have entropy 0."""
        x = np.array([5] * 100)
        h = entropy_discrete(x)
        assert h == pytest.approx(0.0, abs=1e-10)

    def test_insufficient_data(self):
        x = np.array([1])
        assert np.isnan(entropy_discrete(x))

    def test_empty_array(self):
        x = np.array([])
        assert np.isnan(entropy_discrete(x))


class TestEntropyContinuous:
    def test_returns_finite(self):
        np.random.seed(42)
        x = np.random.randn(1000)
        h = entropy_continuous(x, bins=20)
        assert np.isfinite(h)
        assert h > 0

    def test_insufficient_data(self):
        x = np.array([1.0])
        assert np.isnan(entropy_continuous(x))


class TestMIDiscrete:
    def test_identical_equals_entropy(self):
        """MI(X, X) should equal H(X)."""
        x = np.array([0, 0, 1, 1, 2, 2] * 100)
        mi = mutual_information_discrete(x, x)
        h = entropy_discrete(x)
        assert mi == pytest.approx(h, rel=0.05)

    def test_independent_near_zero(self):
        """MI of independent variables should be near zero."""
        np.random.seed(42)
        x = np.random.choice([0, 1, 2], size=10000)
        y = np.random.choice([0, 1], size=10000)
        mi = mutual_information_discrete(x, y)
        assert mi < 0.02

    def test_deterministic_function(self):
        """MI(X, f(X)) should equal H(X) when f is injective."""
        x = np.array([0, 1, 2] * 200)
        y = x * 2 + 1
        mi = mutual_information_discrete(x, y)
        h = entropy_discrete(x)
        assert mi == pytest.approx(h, rel=0.05)

    def test_non_negative(self):
        np.random.seed(42)
        x = np.random.choice([0, 1, 2], size=500)
        y = np.random.choice(["a", "b"], size=500)
        mi = mutual_information_discrete(x, y)
        assert mi >= 0.0

    def test_insufficient_data(self):
        x = np.array([1])
        y = np.array([0])
        assert np.isnan(mutual_information_discrete(x, y))


class TestMIContinuousDiscrete:
    def test_informative_feature_high_mi(self):
        """Feature that separates classes should have high MI."""
        np.random.seed(42)
        y = np.array([0] * 500 + [1] * 500)
        x = np.where(y == 0, np.random.normal(0, 1, 1000), np.random.normal(3, 1, 1000))
        mi = mutual_information_continuous_discrete(x, y, bins=20)
        assert mi > 0.3

    def test_uninformative_feature_low_mi(self):
        """Feature independent of target should have near-zero MI."""
        np.random.seed(42)
        x = np.random.randn(5000)
        y = np.random.choice([0, 1, 2], size=5000)
        mi = mutual_information_continuous_discrete(x, y, bins=20)
        assert mi < 0.03

    def test_nan_handling(self):
        """Should handle NaN values in feature."""
        np.random.seed(42)
        x = np.random.randn(1000)
        x[::10] = np.nan
        y = np.random.choice([0, 1], size=1000)
        mi = mutual_information_continuous_discrete(x, y, bins=20)
        assert np.isfinite(mi)


class TestMIContinuous:
    def test_identical_high_mi(self):
        """MI(X, X) should be high."""
        np.random.seed(42)
        x = np.random.randn(1000)
        mi = mutual_information_continuous(x, x, bins=20)
        assert mi > 1.0

    def test_independent_near_zero(self):
        """MI of independent continuous variables should be near zero."""
        np.random.seed(42)
        x = np.random.randn(10000)
        y = np.random.randn(10000)
        mi = mutual_information_continuous(x, y, bins=20)
        assert mi < 0.05

    def test_non_negative(self):
        np.random.seed(42)
        x = np.random.randn(500)
        y = np.random.randn(500)
        mi = mutual_information_continuous(x, y, bins=20)
        assert mi >= 0.0


class TestNMI:
    def test_range_zero_one(self):
        nmi = normalized_mutual_information(0.5, 1.0, 1.0)
        assert 0.0 <= nmi <= 1.0

    def test_perfect_nmi(self):
        """NMI should be 1.0 when MI = H(X) = H(Y)."""
        nmi = normalized_mutual_information(1.5, 1.5, 1.5)
        assert nmi == pytest.approx(1.0, abs=0.01)

    def test_zero_entropy_returns_nan(self):
        nmi = normalized_mutual_information(0.0, 0.0, 1.0)
        assert np.isnan(nmi)

    def test_nan_input_returns_nan(self):
        assert np.isnan(normalized_mutual_information(np.nan, 1.0, 1.0))
        assert np.isnan(normalized_mutual_information(0.5, np.nan, 1.0))
