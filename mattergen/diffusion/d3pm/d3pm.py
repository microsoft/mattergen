# Copyright (c) 2022 The Google Research Authors.
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Adapted from https://github.com/google-research/google-research/blob/master/d3pm/text/diffusion.py
# Siginificant changes:
# * adapt code style/ formatting
# * Jax -> PyTorch
# * Remove Diffusion types that are not used by MatterGen
# ORIGINAL LICENSE NOTICE:
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Diffusions for training and noise scheduling."""

import abc
import dataclasses
from typing import Any, Callable, Dict, Optional, Union

import torch
import torch.nn.functional as F
from torch.distributions import Categorical


class DiffusionSchedule:
    """A wrapper around a simple schedule function."""

    def __init__(self, schedule_fn, num_steps, is_constant=False):
        self._schedule_fn = schedule_fn
        self.num_steps = num_steps
        self.is_constant = is_constant

    def __call__(self, step):
        return self._schedule_fn(step)

    def __repr__(self):
        return f"DiffusionSchedule(steps: {self.num_steps}, is_constant: {self.is_constant})"


class DiscreteDiffusionBase(abc.ABC):
    """Base class for all matrix-noise schedules."""

    num_steps: int
    dim: int
    precision: Any = torch.float32

    @abc.abstractmethod
    def stationary_probs(self, shape):
        """Returns probs for the stationary distribution."""

    @abc.abstractmethod
    def sample_stationary(self, shape):
        """Draws a sample from the stationary distribution (q(x_T))."""

    @property
    def has_state(self):
        """Indicates if the diffusion has state which needs to be set/updated."""
        return False

    def set_state(self, state):
        pass

    def reset_state(self):
        pass

    def update_state(self, state):
        pass

    def sample_t(self, shape=(1,)):
        """Samples batches of time steps to use."""

        num_steps = self.num_steps
        t = torch.randint(shape, minval=0, maxval=num_steps)
        return t

    @abc.abstractmethod
    def get_qt_given_q0(self, q0, t, return_logits=False, make_one_hot=False, epsilon=1e-20):
        """Get q(x_t), the n-step posterior.

        For example, for t = 0, it returns q0 unchanged.

        Args:
          q0: an array of floats specifying a distribution over p(x_0).
          t: t in q(x_t | x_0).
          return_logits: if True, return the output logits
          make_one_hot: if True, will convert q0 to floats if needed.
          epsilon: a small number to normalize logits conversion with, if needed.

        Returns:
          q(x_t | x_0).
        """

    @abc.abstractmethod
    def sample_and_compute_posterior_q(
        self,
        x_0,
        t,
        samples=None,
        transition_probs=None,
        return_logits=True,
        return_transition_probs=False,
        transition_probs_in_logits=True,
        make_one_hot=True,
        epsilon=1e-20,
        step_size=1,
    ):
        """Samples from q(x_{t+1} | x_0), then computes q(x_t | x_{t+1}, x_0).

        Args:
          x_0: an array containing x_0 samples. These are expected to be integral
            unless make_one_hot is False (in which case probabilities can be
            provided).
          t: the timestep to compute (as an int or integer array with shape that
            matches x_0.
          samples: if not None, use these samples to compute the posterior.
          transition_probs: precomputed transition probabilities.
          return_logits: if True, returns the (noisy) log of the probabilities.
          return_transition_probs: if true, returns the transition probs as well.
          transition_probs_in_logits: include transition probs in logits.
          make_one_hot: if True, will convert the input to a one_hot vector.
          epsilon: a small amount of noise to add to logits if needed.
          step_size: if provided, computes q(x_{t + step_size} | x_0), etc. This is
            used to sample fewer steps for ELBO evaluation on a longer trained
            model.

        Returns:
          a list of samples with the same shape as x_0 and the associated posterior
          probabilities (or logits).
        """


class DiscreteDiffusionMatrixBase(DiscreteDiffusionBase):
    """Base class for all matrix-noise schedulers."""

    num_steps: int
    dim: int
    precision: Any = torch.float32

    def get(self, t):
        """Returns the transition matrix q(x_{t+1} | x_t)."""
        raise NotImplementedError

    def custom_product_fn(self, t):
        """Returns q(x_t | x_0), the product of the first t matrices."""
        raise NotImplementedError

    def supports_efficient_get(self):
        """Returns true if get() is implemented/efficient."""
        return False

    def supports_efficient_inference(self):
        """Returns true if custom_product_fn is implemented.

        The ontology of efficient_get and efficient_inference is this:
          * if efficient_inference is enabled, it is used to return q(x_t | x_0)
            without computing expensive products.
          * if efficient_get is enabled, get(...) is used to get the posterior of
            q(x_{t-1} | x_t, x_0). If not, get_q_given_q0 is called to get
            q(x_{t+1} | x_0), and qt_reverse is called to get the q(x_{t+1} | x_t).
        """
        return False

    def qt_reverse(self, qt_plus_1, t, return_logits=False, make_one_hot=False, epsilon=1e-20):
        """Get q(x_{t+1} | x_t), for each possible value of x_t. Thus, the rows of the output do not sum to 1.

        Args:
          qt_plus_1: an array of floats specifying a distribution over q(x_{t+1} | x_0).
          t: t in q(x_{t+1} | x_t).
          return_logits: if True, return the output logits
          make_one_hot: if True, will convert q(x_{t+1}) to floats if needed.
          epsilon: a small number to normalize logits conversion with, if needed.

        Returns:
          q(x_{t+1} | x_t), shape [num_samples, num_classes].
        """
        raise NotImplementedError

    def get_qt_matrix(self, t):
        """Returns the matrix Q = q(x_t | x_0) materialized over all x_0."""
        if self.supports_efficient_inference():
            return self.custom_product_fn(t)

        # otherwise, multiply by the ith matrix in a for-loop.
        def product_fn(i, state):
            return torch.matmul(self.get(torch.tensor(i)), state)

        val = torch.eye(self.dim, device=t.device)
        for i in range(0, t):
            val = product_fn(i, val)
        return val

    def get_qt_given_q0(self, q0, t, return_logits=False, make_one_hot=False, epsilon=1e-20):
        """Get q(x_t), the n-step posterior.

        For example, for t = 0, it returns q0 unchanged.

        Args:
          q0: an array of floats specifying a distribution over p(x_0).
          t: t in q(x_t | x_0).
          return_logits: if True, return the output logits
          make_one_hot: if True, will convert q0 to floats if needed.
          epsilon: a small number to normalize logits conversion with, if needed.

        Returns:
          q(x_t | x_0).
        """

        if make_one_hot:
            assert q0.dtype == torch.long or q0.dtype == torch.int32
            q0 = torch.eye(self.dim, device=q0.device)[q0]

        assert q0.dtype == torch.float32

        # if efficient inference is supported, just return those matrices.
        if self.supports_efficient_inference():
            prob_at_time_t = torch.einsum("bij,bj->bi", self.get_qt_matrix(t).to(q0.dtype), q0)

            if return_logits:
                return torch.log(prob_at_time_t + epsilon)
            else:
                return prob_at_time_t

        @dataclasses.dataclass
        class ScanState:
            final_time: int  # target time
            q: Any

        def product_fn(state, current_time):
            cond = current_time < state.final_time
            transition = self.get(current_time)
            q_t_plus_1 = torch.einsum("ij,sj->si", transition, state.q)

            new_q = torch.where(cond[:, None], q_t_plus_1, state.q)
            return ScanState(final_time=state.final_time, q=new_q), None

        init_val = ScanState(final_time=t, q=q0)
        carry = init_val
        idx = torch.arange(self.num_steps, device=q0.device)
        for i in idx:
            carry, _ = product_fn(carry, i)
        final_state = carry
        prob_at_time_t = final_state.q

        if return_logits:
            return torch.log(prob_at_time_t + epsilon)
        else:
            return prob_at_time_t

    def sample_and_compute_posterior_q(
        self,
        x_0,
        t,
        samples=None,
        transition_probs=None,
        return_logits=True,
        return_transition_probs=False,
        transition_probs_in_logits=True,
        make_one_hot=True,
        epsilon=1e-20,
        step_size=1,
    ):
        """Samples from q(x_{t+1} | x_0), then computes q(x_t | x_{t+1}, x_0).

        Args:
          x_0: an array containing x_0 samples. These are expected to be integral
            unless make_one_hot is False (in which case probabilities can be
            provided).
          t: the timestep to compute (as an int or integer array with shape that
            matches x_0.
          samples: if not None, use these samples to compute the posterior.
          transition_probs: precomputed transition probabilities.
          return_logits: if True, returns the (noisy) log of the probabilities.
          return_transition_probs: if true, returns the transition probs as well.
          transition_probs_in_logits: include transition probs in logits.
          make_one_hot: if True, will convert the input to a one_hot vector.
          epsilon: a small amount of noise to add to logits if needed.
          step_size: if provided, computes q(x_{t + step_size} | x_0), etc. This is
            used to sample fewer steps for ELBO evaluation on a longer trained
            model.

        Returns:
          a list of samples with the same shape as x_0 and the associated posterior
          probabilities (or logits).
        """

        dim = self.dim
        device = x_0.device
        # t = torch.tensor(t, device=x_0.device)
        if make_one_hot:
            assert x_0.dtype in [torch.long, torch.int32]
            x_0 = torch.eye(dim, device=device)[x_0].reshape(x_0.shape + (dim,))
        assert x_0.dtype == torch.float32
        assert t.dtype in [torch.long, torch.int32]
        prob_at_time_t = self.get_qt_given_q0(q0=x_0, t=t)
        # most methods support efficiently returning the t-th transition matrix
        # if so, we use that. Otherwise we recompute the t+1th probability.
        if self.supports_efficient_get():
            if step_size > 1:
                transition_matrix = torch.eye(self.dim, device=x_0.device)

                for i in range(step_size):
                    transition_matrix = self.get(t + i) @ transition_matrix

            else:
                transition_matrix = self.get(t)

            prob_at_time_t_plus_one = torch.einsum(
                "bij,bj->bi",
                transition_matrix,
                prob_at_time_t,
            )

        else:
            prob_at_time_t_plus_one = self.get_qt_given_q0(q0=x_0, t=t + step_size)

        if samples is None and transition_probs is not None:
            raise ValueError("samples were not provided but transition_probs were.")

        # if samples are provided, we use those. otherwise, we sample more.
        if samples is None:
            logits = torch.log(prob_at_time_t_plus_one + epsilon)
            samples = Categorical(logits=logits).sample()

        # we can optionally provide transition probs from another call to this
        # function. If not, we recompute this. For most methods, we can reuse the
        # transition matrix. If we didn't compute it, our method must support
        # qt_reverse which usually computes efficient backwards VJPs.

        if transition_probs is None:
            if self.supports_efficient_get():
                transition_probs = transition_matrix[range(samples.shape[0]), samples]
            else:
                if step_size > 1:
                    transition_probs = torch.eye(self.dim, device=samples.device)[samples]
                    for i in range(step_size):
                        transition_probs = self.qt_reverse(
                            qt_plus_1=transition_probs, make_one_hot=False, t=t + step_size - 1 - i
                        )
                else:
                    # Computes q(x_{t+1} | x_t), i.e., for each possible x_t, what is the probability of transitioning to each x_{t+1}.
                    # Thus, these probabilities do not sum to 1 per row.
                    # If we don't return logits, transition_probs will be used to compute q(x_t | x_{t+1}).
                    # Otherwise, we return the logits of q(x_t | x_{t+1}) = q(x_{t+1} | x_t) * q(x_t | x_0), i.e., omit normalization by q(x_{t+1} | x_0).
                    # Shape [batch_size, num_classes]
                    transition_probs = self.qt_reverse(qt_plus_1=samples, make_one_hot=True, t=t)

        if not transition_probs_in_logits and not return_logits:
            raise ValueError(
                "Cannot exclude transition probs from logits if return_logits is false."
            )

        if return_logits:
            # for numerical stability, we can compute log(a*b) = log(a) + log(b)
            posterior_logits = torch.log(prob_at_time_t + epsilon)

            if transition_probs_in_logits:
                posterior_logits += torch.log(transition_probs + epsilon)

            if return_transition_probs:
                return posterior_logits, samples, transition_probs
            else:
                return posterior_logits, samples
        else:
            # here we hope this never actually sums to zero. There's a chance
            # this will produce NaN gradients, but that's OK because they'll be
            # skipped.
            posterior = transition_probs * prob_at_time_t
            denominator = torch.sum(posterior, dim=-1, keepdims=True)
            posterior = posterior / denominator

            if return_transition_probs:
                return posterior, samples, transition_probs
            else:
                return posterior, samples


class MaskDiffusion(DiscreteDiffusionMatrixBase):
    """A simple schedule that diffuses away from the identity matrix."""

    def __init__(self, dim, schedule, precision=torch.float32, use_fast_inference=True):
        """A simple scheduler for masking policies.

        Args:
          dim: int, the dimensionality of the state space.
          schedule: a DiffusionSchedule object for scheduling rates.
          precision: matmul precision.
          use_fast_inference: if False, uses a slower, brute force approach.
        """

        self.num_steps = schedule.num_steps
        self.schedule = schedule
        self.use_fast_inference = use_fast_inference
        self.precision = precision
        self.dim = dim  # allow mask
        self.state = self._create_state()

    def _create_state(self):
        """Initializes values used by the get function."""
        betas = torch.cat([torch.tensor([0.0]), self.schedule(torch.arange(self.num_steps))]).to(
            torch.float64
        )
        alphas = 1 - betas
        state = torch.cumprod(alphas, dim=0)
        state[-1] = 0.0

        return state.float()

    def supports_efficient_inference(self):
        return self.use_fast_inference

    def stationary_probs(self, shape):
        """Stationary distribution is one-hot at mask token."""
        sample = torch.full(shape, self.dim - 1)
        probs = torch.eye(self.dim, device=sample.device)[sample]
        return probs

    def sample_stationary(self, shape):
        """Stationary distribution is one-hot at mask token."""
        return torch.full(shape, self.dim - 1)

    def custom_product_fn(self, t):
        """Returns product of first n matrices. Only supported for beta constant."""
        dim = self.dim

        if self.schedule.is_constant:
            beta = self.schedule(0)
            return (1 - beta) ** t * torch.eye(dim) + (1 - (1 - beta) ** t) * self._get_mask()

        else:
            p = self.state[t]
            return p * torch.eye(dim) + (1 - p) * self._get_mask()

    def _get_mask(self):
        dim = self.dim
        return torch.ones((dim, dim)) * (torch.arange(0, dim)[:, None] == (dim - 1)).to(
            torch.float32
        )

    def get(self, t):
        _t = t if len(t.shape) == 1 else t[None]
        beta = self.schedule(_t)
        dim = self.dim

        ret = (1 - beta)[:, None, None] * torch.eye(dim, device=_t.device)[None] + beta[
            :, None, None
        ] * self._get_mask().to(_t.device)[None]
        return ret if len(t.shape) == 1 else ret.squeeze(0)

    def qt_reverse(self, qt_plus_1, t, return_logits=False, make_one_hot=False, epsilon=1e-20):
        """Get q(x_{t+1} | x_t), for each possible value of x_t. Thus, the rows of the output do not sum to 1.

        Args:
          qt_plus_1: an array of floats specifying a distribution over q(x_{t+1} | x_0).
          t: t in q(x_{t+1} | x_t).
          return_logits: if True, return the output logits
          make_one_hot: if True, will convert q(x_{t+1}) to floats if needed.
          epsilon: a small number to normalize logits conversion with, if needed.

        Returns:
          q(x_{t+1} | x_t), shape [num_samples, num_classes].
        """

        if make_one_hot:
            assert qt_plus_1.dtype in [torch.long, torch.int32]
            qt_plus_1 = torch.eye(self.dim, device=qt_plus_1.device)[qt_plus_1]

        assert qt_plus_1.dtype == torch.float32

        beta = self.schedule(t)

        # q(x_{t+1} | x_t) = (1 - beta) if x_t = x_{t+1} != mask type
        #   else: beta if x_t != mask type else 1. (beta is the probability of transitioning to the absorbing state at t).
        # I.e., if x_{t+1} is in some non-masked state S, then the probability of transitioning from S in t to S in t+1 is (1 - beta).
        # Else, if x_{t+1} is in the masked state, then the probability of transitioning from a non-masked state S in t to the masked state in t+1 is beta,
        # and the probability of transitioning from the masked state to itself is 1.
        non_mask_prob = (1 - beta)[:, None] * qt_plus_1[:, :-1] + beta[:, None] * qt_plus_1[:, -1:]
        prob_at_time_t = (
            torch.eye(self.dim, device=qt_plus_1.device)[self.dim - 1][None] * qt_plus_1[:, -1:]
        )
        prob_at_time_t[:, :-1] = non_mask_prob

        if return_logits:
            return torch.log(prob_at_time_t + epsilon)
        else:
            return prob_at_time_t

    def get_qt_given_q0(self, q0, t, return_logits=False, make_one_hot=False, epsilon=1e-20):
        """Get q(x_t), the n-step posterior.

        Can do efficiently for masks.

        For example, for t = 0, it returns q0 unchanged.

        Args:
          q0: an array of floats specifying a distribution over p(x_0).
          t: t in q(x_t | x_0).
          return_logits: if True, return the output logits
          make_one_hot: if True, will convert q0 to floats if needed.
          epsilon: a small number to normalize logits conversion with, if needed.

        Returns:
          q(x_t | x_0).
        """
        if not self.supports_efficient_inference():
            return super().get_qt_given_q0(
                q0, t, return_logits=return_logits, make_one_hot=make_one_hot, epsilon=epsilon
            )

        if make_one_hot:
            assert q0.dtype in [torch.int32, torch.long]
            q0 = torch.eye(self.dim, device=q0.device)[q0]

        assert q0.dtype == torch.float32
        assert len(q0.shape) == 2

        # p is probability of staying the same. (1 - p) is prob of masking.
        p = self.state.to(q0.device)[t]

        non_mask_prob = p[:, None] * q0[:, :-1]
        mask_prob = 1 - non_mask_prob.sum(-1)

        prob_at_time_t = (
            mask_prob[:, None] * torch.eye(self.dim, device=q0.device)[self.dim - 1][None]
        )
        prob_at_time_t[:, :-1] = non_mask_prob

        prob_at_time_t = torch.where(t[:, None] == 0, q0, prob_at_time_t)

        if return_logits:
            return torch.log(prob_at_time_t + epsilon)
        else:
            return prob_at_time_t

    def supports_efficient_get(self):
        return not self.use_fast_inference


def create_discrete_diffusion_schedule(
    kind="linear",
    beta_min=1e-3,
    beta_max=1e-1,
    num_steps=100,
    scale=1.0,
):
    """Creates a callable schedule object to use for diffusion rates.

    Args:
      kind: str, one of 'standard', 'linear', 'cosine', 'mutual_information'. If
        standard, performs standard binomial diffusion taken from Sohl-Dicksteein
        et al, ignoring betas. Otherwise, linear schedule between beta_min and
        beta_max.
      beta_min: the minimum beta. Ignored if kind == standard.
      beta_max: the maximum beta.
      num_steps: int, the number of steps to take.
      scale: for standard schedule, rescales num_steps by this amount.

    Returns:
      a DiffusionSchedule object.
    """

    assert beta_min <= beta_max
    assert num_steps > 0
    assert scale >= 1

    if kind == "standard":

        def schedule_fn(step: Union[int, torch.Tensor]):
            return 1 / (scale * num_steps - step)

        return DiffusionSchedule(schedule_fn, num_steps, is_constant=False)

    elif kind == "linear":
        is_constant = beta_min == beta_max

        linspace = torch.linspace(beta_min, beta_max, num_steps)

        def schedule_fn(step: Union[int, torch.Tensor]):
            return linspace[step]

        return DiffusionSchedule(schedule_fn, num_steps, is_constant=is_constant)
    elif kind == "cosine":
        s = 0.008

        def cosine_fn(step: torch.Tensor):
            return torch.cos((step / num_steps + s) / (1 + s) * torch.pi / 2)

        def schedule_fn(step: Union[int, torch.Tensor]):
            if isinstance(step, int):
                step = torch.tensor(step)
            return torch.clamp(1 - (cosine_fn(step + 1) / cosine_fn(step)), 0, 0.999)

        return DiffusionSchedule(schedule_fn, num_steps, is_constant=False)
    else:
        raise ValueError(f"kind {kind} is not supported.")


def p_forward(
    denoise_fn,
    x_t,
    t,
    diffusion,
    predict_x0=True,
    return_x0=False,
    return_logits=False,
    special_case_x0=False,
    transition_probs=None,
    transition_probs_in_logits=True,
    maximum_likelihood=False,
    epsilon=1e-20,
    step_size=1,
):
    """Returns probabilities from the reverse process p(x_{t-1} | x_t).

    Args:
      denoise_fn: the reverse process. Must support embed, call, and attend.
      x_t: the current value of x_t to condition on.
      t: the timestep t.
      diffusion: the Diffusion object to use for noise.
      predict_x0: if True, assumes the model output corresponds to its prediction
        for p(x_0 | x_t). Otherwise assumes model predicts p(x_{t-1} | x_t).
      return_x0: if True, will return probs for x_0 as well as x_{t-1}.
      return_logits: if True, will return logits instead of probabilities.
      special_case_x0: if True, will directly predict x0 instead of using the
        forward process probabilities.
      transition_probs: if provided, q(x_{t+1} | x_t) probs to reuse.
      transition_probs_in_logits: if False, will ignore transition probs in logits
        (only allowed if return_logits is True). This is because this term is
        independent of theta.
      maximum_likelihood: if true, will draw the most likely x0 before applying
        the forward process.
      epsilon: a small number.
      step_size: step size to compute posterior from.

    Returns:
      probabilities for q(x_{t-1} | x_t) (and probabilities for x0 if predict_x0
      is True)
    """
    assert not (step_size > 1 and not predict_x0)

    logits = denoise_fn(targets=x_t, timestep=t)
    probs = logits.softmax(dim=-1)

    if not predict_x0:
        retval = logits if return_logits else probs
        if return_x0:
            return retval, None
        else:
            return retval

    if maximum_likelihood:
        probs = probs.argmax(-1)

    # we use this to compute p(x_{t-1} | x_t) = sum_x0 q(x_{t-1} | x_t, x_0)
    # p(x_0 | x_t).
    qt_probs, _ = diffusion.sample_and_compute_posterior_q(
        x_0=probs,
        t=t - step_size,
        make_one_hot=maximum_likelihood,
        return_logits=return_logits,
        transition_probs_in_logits=transition_probs_in_logits,
        transition_probs=transition_probs,
        samples=x_t,
        epsilon=epsilon,
        step_size=step_size,
    )

    retval_x0 = logits if return_logits else probs
    retval = qt_probs

    # we can special case t = 1 to just use the raw logits outputs.
    mask = (t == step_size) & special_case_x0
    retval = mask[:, None] * retval_x0 + (mask.logical_not())[:, None] * retval

    if return_x0:
        return retval, retval_x0
    else:
        return retval


def q_sample(x_start, t, diffusion, return_logits=False):
    """Draws a sample from the posterior q(x_t | x_start)."""

    assert x_start.dtype in [torch.int32, torch.long]

    dim = diffusion.dim
    x_start = torch.eye(dim, device=x_start.device)[x_start]

    logits = diffusion.get_qt_given_q0(q0=x_start, t=t, return_logits=True)
    sample = Categorical(logits=logits).sample()
    if return_logits:
        return sample, logits
    return sample


def compute_prior_kl(x_start, diffusion, target_mask=None):
    """Computes KL divergence between q(x_T) and the true distribution."""
    assert x_start.dtype in [torch.long, torch.int32]

    num_steps = diffusion.num_steps

    q_probs = diffusion.get_qt_given_q0(
        q0=x_start,
        t=torch.tensor(
            [
                num_steps,
            ],
            device=x_start.device,
        ),
        return_logits=False,
        make_one_hot=True,
    )  # get end step
    p_probs = diffusion.stationary_probs(q_probs.shape[:-1]).to(q_probs.device)

    d1 = Categorical(probs=q_probs)
    d2 = Categorical(probs=p_probs)
    loss = torch.distributions.kl_divergence(d1, d2)

    if target_mask is not None:
        loss = (loss * target_mask).sum()
    else:
        loss = loss.sum()

    return loss


def compute_kl_reverse_process(
    x_start: torch.Tensor,
    t: torch.Tensor,
    *,
    x_t_plus_1: Optional[torch.Tensor] = None,
    diffusion: DiscreteDiffusionBase,
    denoise_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    predict_x0: bool = True,
    log_space: bool = False,
    label_smoothing: float = 0.0,
    hybrid_lambda: float = 0.0,
    use_cached_transition: bool = True,
    target_mask: Optional[torch.Tensor] = None,
    step_size: int = 1,
) -> Dict[str, torch.Tensor]:
    """Returns the KL for one term in the ELBO (time t) (loss L_t).

    This assumes x_start is a sample from x_0, from which we draw samples from
    q(x_t | x_0) and then compute q(x_{t-1} | x_t, x_0) following the LaTeX. This
    is the KL divergence for terms L_1 through L_{T-1}.

    Args:
      x_start: a sample from p(data) (or q(x_0)).
      t: the loss term to compute.
      diffusion: the diffusion object to use.
      denoise_fn: a functool.partial-ed version of the model_apply function which
        takes a set of targets (x_t) and noise level and returns q(x_{t-1} | x_t,
        x_0).
      predict_x0: if True, will predict a distribution over x0 instead of x_{t-1}.
      log_space: if True, will perform the loss calculations in log space.
      label_smoothing: label smoothing for cross entropy.
      hybrid_lambda: coefficient for hybrid cross-entropy loss.
      use_cached_transition: if True, will reuse q(x_{t+1} | x_t) computation.
      target_mask: mask for target sequence.
      step_size: the step size over which the ELBO is computed.

    Returns:
      the KL divergence and denominator.
    """
    assert x_start.dtype in [torch.int32, torch.long]

    if step_size > 1 and not predict_x0:
        raise ValueError("cannot skip steps when not predicting x0.")

    # If x_t_plus_1 is None, sample from q(x_{t+1} | x_start). Otherwise use the provided samples for x_{t+1}.
    # Then compute q(x_t | x_{t+1}, x_start)
    # q_t and p_t can be logits or probs depending on log_space.
    q_t, x_t_plus_1, transition_probs = diffusion.sample_and_compute_posterior_q(
        x_0=x_start,
        t=t,
        return_logits=log_space,
        return_transition_probs=True,
        step_size=step_size,
        samples=x_t_plus_1,
    )

    transition_probs = transition_probs if use_cached_transition else None

    p_t = p_forward(
        denoise_fn=denoise_fn,
        x_t=x_t_plus_1,
        t=t + step_size,
        diffusion=diffusion,
        predict_x0=predict_x0,
        return_x0=predict_x0 and hybrid_lambda > 0.0,
        return_logits=log_space,
        transition_probs=transition_probs,
        step_size=step_size,
    )

    hybrid_loss = torch.tensor(0.0, device=x_start.device)
    if predict_x0 and hybrid_lambda > 0.0:
        # p_t, p_0 are shape [num_atoms, ].
        p_t, p_0 = p_t
        if log_space:
            # [num_atoms, ]
            cross_entropy = F.cross_entropy(
                input=p_0, target=x_start, label_smoothing=label_smoothing, reduction="none"
            )
        else:
            # [num_atoms, ]
            cross_entropy = F.cross_entropy(
                input=(p_0 + 1e-7).log(),
                target=x_start,
                label_smoothing=label_smoothing,
                reduction="none",
            )

        hybrid_loss = hybrid_lambda * cross_entropy

    assert not q_t.isnan().any() and not p_t.isnan().any()

    if log_space:
        d1 = Categorical(logits=q_t)
        d2 = Categorical(logits=p_t)
        # [num_atoms, ]
        kl = torch.distributions.kl_divergence(p=d1, q=d2)
        # [num_atoms, ]
        cross_entropy = F.cross_entropy(
            input=p_t, target=x_start, label_smoothing=label_smoothing, reduction="none"
        )
    else:
        d1 = Categorical(logits=(q_t + 1e-7).log())
        d2 = Categorical(logits=(p_t + 1e-7).log())
        # [num_atoms, ]
        kl = torch.distributions.kl_divergence(p=d1, q=d2)
        # [num_atoms, ]
        cross_entropy = F.cross_entropy(
            input=(p_t + 1e-7).log(),
            target=x_start,
            label_smoothing=label_smoothing,
            reduction="none",
        )

    if target_mask is not None:  # can be used for inpainting
        kl = kl * target_mask
        cross_entropy = cross_entropy * target_mask
        hybrid_loss = hybrid_loss * target_mask

    # [num_atoms, ]
    mask = t == 0
    base_loss = mask * cross_entropy + (mask.logical_not()) * kl
    loss = base_loss + hybrid_loss
    denominator = torch.tensor(1, device=x_start.device)

    metrics_dict = {
        "loss": loss,
        "denominator": denominator,
        "kl/hybrid_loss": hybrid_loss,
        "kl/base_loss": base_loss,
        "kl/cross_entropy_loss": cross_entropy,
        "kl/t0_loss": mask * cross_entropy,
        "kl/kl_loss": kl,
    }

    return metrics_dict
