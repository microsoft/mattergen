# Copyright (c) 2022 The Google Research Authors
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# from https://github.com/google-research/google-research/blob/master/d3pm/text/diffusion_test.py
# Keeping the original copyright notice
# Changes
# * adapt code style
# * Jax -> PyTorch
# * Remove Diffusion types that are not used by MatterGen
# # coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for d3pm.py."""
import functools

import numpy as np
import pytest
import torch

from mattergen.diffusion.d3pm import d3pm as diffusion


@pytest.mark.parametrize("schedule_kind", ["linear", "standard", "cosine"])
def test_prior_kl(schedule_kind: str):
    """Test the prior KL computation."""

    schedule = diffusion.create_discrete_diffusion_schedule(
        kind=schedule_kind,
        beta_min=1e-3,
        beta_max=1e-1,
        num_steps=1000,
    )

    dim = 100
    num_samples = 71
    x_in = torch.randint(0, dim, size=(num_samples,))
    diff = diffusion.MaskDiffusion(dim=dim + 1, schedule=schedule)
    prior_kl = diffusion.compute_prior_kl(x_in, diff)
    assert torch.isclose(prior_kl, torch.tensor(0.0), atol=1e-5)


def test_product_the_hard_way():
    """Tests that the discrete transition matrices computed via q(x_t | x_0) and q(x_t|x_{t-1}) are equivalent
    for t in {0, 1}. Uses the slow iterative method of computing the transition matrix q(x_t | x_0).
    """
    schedule = diffusion.create_discrete_diffusion_schedule(
        kind="linear",
        beta_min=1e-3,
        beta_max=1e-3,
        num_steps=100,
    )

    diff = diffusion.MaskDiffusion(dim=100, schedule=schedule, use_fast_inference=False)

    assert not diff.supports_efficient_inference()

    product = diff.get_qt_matrix(torch.tensor(0))
    np.testing.assert_array_almost_equal(product, torch.eye(100))

    product = diff.get_qt_matrix(torch.tensor(1)[None])
    np.testing.assert_array_almost_equal(product, diff.get(torch.tensor(0)))


def test_product_fast():
    """Tests that the discrete transition matrices computed via q(x_t | x_0) and q(x_t|x_{t-1}) are equivalent
    for t in {0, 1}. Uses the fast closed-form method of computing the transition matrix q(x_t | x_0).
    """
    schedule = diffusion.create_discrete_diffusion_schedule(
        kind="linear",
        beta_min=1e-3,
        beta_max=1e-3,
        num_steps=100,
    )

    diff = diffusion.MaskDiffusion(dim=100, schedule=schedule, use_fast_inference=True)

    assert diff.supports_efficient_inference()

    product = diff.get_qt_matrix(torch.tensor(0))
    np.testing.assert_array_almost_equal(product, torch.eye(100))

    product = diff.get_qt_matrix(torch.tensor(1))
    np.testing.assert_array_almost_equal(product, diff.get(torch.tensor(0)))


def test_product_constant():
    """Tests, when we have a constant beta schedule (transition probabilities don't change over time),
    whether the transition matrices computed via q(x_t | x_0) and q(x_t|x_{t-1}), and via explicit matrix
    multiplication are equivalent."""
    schedule = diffusion.create_discrete_diffusion_schedule(
        kind="linear",
        beta_min=1e-3,
        beta_max=1e-3,
        num_steps=100,
    )

    diff = diffusion.MaskDiffusion(dim=100, schedule=schedule)

    assert diff.supports_efficient_inference()

    product = diff.get_qt_matrix(0)
    np.testing.assert_array_almost_equal(product, torch.eye(100))

    product = diff.get_qt_matrix(1)
    np.testing.assert_array_almost_equal(product, diff.get(torch.tensor(0)))

    product = diff.get_qt_matrix(10)
    expected = np.linalg.matrix_power(diff.get(torch.tensor(0)), 10)
    np.testing.assert_array_almost_equal(product, expected)


def test_sample_and_posterior():
    """Tests whether the samples and posterior are as expected when providing timestep 0 for the sampling."""

    schedule = diffusion.create_discrete_diffusion_schedule(
        kind="linear",
        beta_min=1e-3,
        beta_max=1e-3,
        num_steps=100,
    )

    diff = diffusion.MaskDiffusion(dim=100, schedule=schedule)

    inputs = torch.ones((1,), dtype=torch.long)

    probs, sample = diff.sample_and_compute_posterior_q(
        inputs, torch.tensor([0]), return_logits=False
    )

    assert probs.shape == (1, 100)
    assert torch.allclose(probs[0, 1], torch.tensor(1.0), atol=1e-5)

    assert sample.shape == (1,)
    np.testing.assert_array_equal(sample, np.array([1]))


def test_compute_posterior():
    """Tests that the forward diffusion probabilities are correct for t=0."""

    schedule = diffusion.create_discrete_diffusion_schedule(
        kind="linear",
        beta_min=1e-3,
        beta_max=1e-3,
        num_steps=100,
    )

    diff = diffusion.MaskDiffusion(dim=100, schedule=schedule)

    inputs = torch.ones((2,), dtype=torch.long)
    q_t = diff.get_qt_given_q0(inputs, torch.tensor([0, 0]), make_one_hot=True)

    assert q_t.shape == (2, 100)
    assert torch.allclose((q_t[0][1]), torch.tensor(1.0))
    assert torch.allclose((q_t[0][0]), torch.tensor(0.0))


def test_model():
    """Test the Diffusion noise diffusion."""
    schedule = diffusion.create_discrete_diffusion_schedule(
        kind="standard",
        beta_min=1e-3,
        beta_max=1e-3,
        num_steps=100,
    )
    dim = 100
    length = 100
    x0 = torch.randint(0, dim, (length,))
    diff = diffusion.MaskDiffusion(dim=100, schedule=schedule)
    if hasattr(diffusion, "get"):
        np.testing.assert_allclose(diff.get(0).sum(0), 1.0, rtol=1e-6)
        np.testing.assert_allclose(diff.get(10).sum(0), 1.0, rtol=1e-6)
        np.testing.assert_allclose(diff.get(99).sum(0), 1.0, rtol=1e-6)
        np.testing.assert_allclose(diff.get_qt_matrix(0), torch.eye(100), rtol=1e-6)
    expected = torch.eye(dim)[x0]
    result = diff.get_qt_given_q0(q0=x0, t=torch.tensor([0]), make_one_hot=True)
    np.testing.assert_allclose(result, expected)
    expected = torch.randn((length, dim)).softmax(-1)
    result = diff.get_qt_given_q0(q0=expected, t=torch.tensor([0]), make_one_hot=False)
    np.testing.assert_allclose(result, expected)
    q0 = torch.randn((length, dim)).softmax(-1)
    result = diff.get_qt_given_q0(q0=q0, t=torch.tensor([0]), make_one_hot=False)
    np.testing.assert_allclose(result.sum(axis=-1), 1.0, rtol=1e-6)
    expected = diff.stationary_probs(x0.shape)
    result = diff.get_qt_given_q0(q0=x0, t=torch.tensor([100]), make_one_hot=True)
    np.testing.assert_allclose(result, expected)


def test_mask_diffusion():
    """Test the Diffusion noise diffusion."""
    schedule = diffusion.create_discrete_diffusion_schedule(
        kind="linear",
        beta_min=1e-3,
        beta_max=1e-1,
        num_steps=100,
    )
    diff = diffusion.MaskDiffusion(dim=100, schedule=schedule)
    np.testing.assert_allclose(diff.get(torch.tensor(0)).sum(0), 1.0, rtol=1e-6)
    np.testing.assert_allclose(diff.get(torch.tensor(10)).sum(0), 1.0, rtol=1e-6)
    np.testing.assert_allclose(diff.get(torch.tensor(0))[0, 0], 1.0 - schedule(0), rtol=1e-6)
    np.testing.assert_allclose(diff.get(torch.tensor(1))[0, 0], 1.0 - schedule(1), rtol=1e-6)
    np.testing.assert_allclose(diff.get_qt_matrix(0), torch.eye(100), rtol=1e-6)


def test_mask_diffusion_slow_and_fast():
    """Compares fast and slow inference for mask diffusion."""
    schedule = diffusion.create_discrete_diffusion_schedule(
        kind="standard",
        beta_min=5e-4,
        beta_max=5e-2,
        num_steps=100,
    )
    dim = 16
    length = 16
    fast_diff = diffusion.MaskDiffusion(dim=dim, schedule=schedule, use_fast_inference=True)
    slow_diff = diffusion.MaskDiffusion(dim=dim, schedule=schedule, use_fast_inference=False)
    x0 = torch.randint(0, dim, (length,))
    for _t in range(100):
        t = torch.tensor([_t]).expand_as(x0)
        _t_item = torch.tensor(_t)
        qt_slow = slow_diff.get_qt_matrix(_t_item)
        qt_fast = fast_diff.get_qt_matrix(t)
        np.testing.assert_array_almost_equal(qt_slow, qt_fast, decimal=3)
        qt_slow = slow_diff.get_qt_given_q0(q0=x0, t=t, make_one_hot=True)
        qt_fast = fast_diff.get_qt_given_q0(q0=x0, t=t, make_one_hot=True)
        np.testing.assert_array_almost_equal(qt_slow, qt_fast, decimal=3)
        np.testing.assert_array_almost_equal(qt_slow.sum(axis=-1), 1.0, decimal=3)
        np.testing.assert_array_almost_equal(qt_fast.sum(axis=-1), 1.0, decimal=3)
        torch.manual_seed(234)
        posterior_slow, samples_slow = slow_diff.sample_and_compute_posterior_q(
            x_0=x0, t=t, make_one_hot=True
        )
        torch.manual_seed(234)
        posterior_fast, samples_fast = fast_diff.sample_and_compute_posterior_q(
            x_0=x0, t=t, make_one_hot=True
        )
        np.testing.assert_array_almost_equal(posterior_slow, posterior_fast, decimal=3)
        np.testing.assert_array_equal(samples_slow, samples_fast)
    t_100 = torch.tensor([100]).expand_as(x0)
    qt = fast_diff.get_qt_given_q0(q0=x0, t=t_100, make_one_hot=True)
    np.testing.assert_allclose(
        qt, torch.eye(dim)[torch.full(x0.shape, fill_value=dim - 1)], rtol=1e-6
    )
    qt = slow_diff.get_qt_given_q0(q0=x0, t=t_100, make_one_hot=True)
    np.testing.assert_allclose(
        qt, torch.eye(dim)[torch.full(x0.shape, fill_value=dim - 1)], rtol=1e-6
    )


def test_large_matrices():
    """Tests precision for large matrices."""
    dim = 1000
    length = 64
    x0 = torch.randint(0, dim, (length,))
    schedule = diffusion.create_discrete_diffusion_schedule(
        kind="linear",
        beta_min=5e-4,
        beta_max=5e-2,
        num_steps=100,
    )
    diff = diffusion.MaskDiffusion(dim, schedule, use_fast_inference=True)
    fn = functools.partial(diff.get_qt_given_q0, make_one_hot=True)
    result = fn(x0, torch.tensor([100]))
    np.testing.assert_array_almost_equal(result.sum(axis=-1), 1.0)


def test_loss_computation():
    """Tests whether the loss computation uses the right terms (KL / cross-entropy) and broadcasts correctly."""
    torch.manual_seed(234)
    num_steps = 100
    num_classes = 7
    hybrid_lambda = 0.0
    schedule = diffusion.create_discrete_diffusion_schedule(
        kind="linear",
        beta_min=1e-3,
        beta_max=1e-3,
        num_steps=num_steps,
    )
    t = torch.arange(0, 100)
    diff = diffusion.MaskDiffusion(dim=num_classes, schedule=schedule)
    inputs = torch.ones((num_steps,), dtype=torch.long)
    q_t_minus_one, x_t_samples = diff.sample_and_compute_posterior_q(
        inputs, t, make_one_hot=True, return_logits=True
    )

    # Ground-truth denoising function
    def denoise_fn(targets, timestep):
        return q_t_minus_one

    loss_dict = diffusion.compute_kl_reverse_process(
        x_start=inputs,
        t=t,
        x_t_plus_1=x_t_samples,
        diffusion=diff,
        denoise_fn=denoise_fn,
        predict_x0=False,
        hybrid_lambda=hybrid_lambda,
    )
    loss = loss_dict.pop("loss")
    kl_loss = loss_dict.pop("kl/kl_loss")
    cross_entropy_loss = loss_dict.pop("kl/cross_entropy_loss")
    assert loss.shape == t.shape
    # KL loss should be the same as the loss for all timesteps except the first one, where cross-entropy is used.
    assert torch.allclose(kl_loss[1:], loss[1:])
    assert torch.allclose(cross_entropy_loss[:1], loss[:1])
    assert torch.allclose(kl_loss, torch.zeros_like(kl_loss), atol=1e-6)
