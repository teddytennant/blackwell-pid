import numpy as np
from numpy.testing import assert_allclose
from scipy import linalg

from blackwell_pid import (
    channel_posterior,
    mutual_info,
    normalized_posterior,
    redundancy_fastpath,
    union_closed_form,
)


def test_copy_channels():
    C1 = np.diag([0.5, 1.0])
    C2 = np.diag([1.0, 0.5])
    value, _W = union_closed_form(C1, C2)
    assert_allclose(value, np.log(2.0), atol=1e-10)
    red, _R = redundancy_fastpath(C1, C2)
    assert red is not None
    assert_allclose(red, 0.0, atol=1e-10)
    assert_allclose(channel_posterior(np.array([[1.0, 0.0]])), C1, atol=1e-12)
    assert_allclose(channel_posterior(np.array([[0.0, 1.0]])), C2, atol=1e-12)


def test_scalar():
    C1 = np.array([[0.8]])
    C2 = np.array([[0.2]])
    value, _W = union_closed_form(C1, C2)
    assert_allclose(value, -0.5 * np.log(0.2), atol=1e-12)
    red, _R = redundancy_fastpath(C1, C2)
    assert red is not None
    assert_allclose(red, -0.5 * np.log(0.8), atol=1e-12)


def _random_spd(rng, d=3):
    q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    ev = rng.uniform(0.05, 0.95, size=d)
    return (q * ev) @ q.T


def test_random_spd_pairs():
    rng = np.random.default_rng(0)
    for _ in range(20):
        C1 = _random_spd(rng)
        C2 = _random_spd(rng)
        value, W = union_closed_form(C1, C2)
        assert np.min(linalg.eigvalsh(C1 - W)) >= -1e-7
        assert np.min(linalg.eigvalsh(C2 - W)) >= -1e-7
        i1 = mutual_info(C1)
        i2 = mutual_info(C2)
        assert value >= max(i1, i2) - 1e-8
        lam, _V = linalg.eigh(C2, C1)
        sign, logdet = np.linalg.slogdet(C1)
        assert sign > 0
        other = -0.5 * (logdet + np.sum(np.log(np.minimum(1.0, lam))))
        assert_allclose(value, other, atol=1e-8)


def test_binding_fixture():
    C1 = np.array(
        [[0.83322104, 0.00833471], [0.00833471, 0.2147173]]
    )
    C2 = np.array(
        [[0.63562389, 0.2836853], [0.2836853, 0.42628473]]
    )
    value, _W = union_closed_form(C1, C2)
    assert_allclose(value, 1.234410271979598, atol=1e-6)
    red, _R = redundancy_fastpath(C1, C2)
    assert red is None


def test_normalized_matches_channel():
    rng = np.random.default_rng(2)
    H = rng.normal(size=(2, 3))
    # Y ~ N(0, I_3), X = H Y + N, N ~ N(0, I_2)
    sigma_y = np.eye(3)
    sigma_yx = H.T
    sigma_x = H @ H.T + np.eye(2)
    C = normalized_posterior(sigma_y, sigma_yx, sigma_x)
    assert_allclose(C, channel_posterior(H), atol=1e-7)
