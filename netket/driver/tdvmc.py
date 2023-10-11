# Copyright 2021 The NetKet Authors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

import jax
import jax.numpy as jnp

import flax

from textwrap import dedent
from inspect import signature


import netket.jax as nkjax
from netket.utils.types import PyTree
from netket.operator import AbstractOperator, TimeDerivativeOperator
from netket.stats import Stats
from netket.vqs import MCState
from netket.optimizer import (
    identity_preconditioner,
    PreconditionerT,
    _DeprecatedPreconditionerSignature,
)
from netket.optimizer.qgt import QGTOnTheFly, QGTJacobianDense, QGTJacobianPyTree

from .vmc_common import info
from .abstract_variational_driver import AbstractVariationalDriver



def model_output(t, x, params, unravel_func, model, component='real'):
    out =  model.apply({"params": unravel_func(params)}, x, t)
    if component == 'real':
        return out.real
    elif component == 'imag':
        return out.imag
    elif component == 'complex':
        return out

# func_dpsi_dt = jax.grad(model_output, argnums=0)
# func_dpsi_dtheta = jax.grad(model_output, argnums=2)

# Gradient with respect to t
_grad_dpsi_dt = jax.grad(model_output, argnums=0)
func_dpsi_dt = jax.vmap(_grad_dpsi_dt, in_axes=(None, 0, None, None, None, None))

# Gradient with respect to params
_grad_dpsi_dtheta = jax.grad(model_output, argnums=2)
func_dpsi_dtheta = jax.vmap(_grad_dpsi_dtheta, in_axes=(None, 0, None, None, None, None))


def compute_dtheta_dt(hamiltonian, qgt, expect_and_grad, gamma=-1.0j):
    """Compute the time derivative of the variational parameters theta given the hamiltonian."""
    energy, forces = expect_and_grad(hamiltonian)
    gamma_f = jax.tree_map(lambda x: gamma * x, forces)
    dtheta_dt, _ = qgt.solve(jax.scipy.sparse.linalg.cg, gamma_f)
    ravel_dtheta_dt, unravel = nkjax.tree_ravel(dtheta_dt)
    return ravel_dtheta_dt, unravel

def compute_dpsi_dtheta(t, x, params, model, component='real'):
    """Compute the derivative of the model output with respect to the variational parameters theta."""
    ravel_params, unravel_func = nkjax.tree_ravel(params)
    dpsi_dtheta = func_dpsi_dtheta(t, x.reshape(-1,x.shape[-1]), ravel_params, unravel_func, model, component).reshape(x.shape[:-1]+(-1,))
    ravel_dpsi_dtheta, unravel = nkjax.tree_ravel(dpsi_dtheta)
    return ravel_dpsi_dtheta, unravel

def compute_dpsi_dt_from_params(t, x, unrav_parameters, model, hamiltonian, qgt, expect_and_grad, dynamics='real'):
    if isinstance(t, (int, float)):
        t = jnp.array(t, dtype=jnp.complex128)
    dtheta_dt, _ = compute_dtheta_dt(hamiltonian, qgt, expect_and_grad, gamma=-1.0j)
    # nuermically verified that component='real' is correct
    # however, have not verified theoretically why this is the case
    if dynamics == 'real':
        dpsi_dtheta, _ = compute_dpsi_dtheta(t, x, unrav_parameters, model, component='real')
    elif dynamics == 'imag':
        dpsi_dtheta, _ = compute_dpsi_dtheta(t, x, unrav_parameters, model, component='imag')
    elif dynamics == 'lindblad':
        dpsi_dtheta, _ = compute_dpsi_dtheta(t, x, unrav_parameters, model, component='imag')
        dpsi_dtheta = -1 * dpsi_dtheta
    dpsi_dt = jnp.einsum("...j, j -> ...", dpsi_dtheta.reshape(x.shape[:-1] + (-1,)), dtheta_dt)
    return dpsi_dt

def compute_dpsi_dt_from_inputs(t, x, rav_parameters, parameters_unrav_func, model, dynamics='real'):
    if dynamics == 'real':
        return func_dpsi_dt(t, x.reshape(-1,x.shape[-1]), rav_parameters, parameters_unrav_func, model, 'real').reshape(x.shape[:-1])
    elif dynamics == 'imag':
        return func_dpsi_dt(t, x.reshape(-1,x.shape[-1]), rav_parameters, parameters_unrav_func, model, 'imag').reshape(x.shape[:-1])
    elif dynamics == 'lindblad':
        return -1 * func_dpsi_dt(t, x.reshape(-1,x.shape[-1]), rav_parameters, parameters_unrav_func, model, 'imag').reshape(x.shape[:-1])

def consistency_loss_fn(t, x, hamiltonian, parameters, parameters_unravel_fn, model, qgt, expect_and_grad):
    dpsi_dt_from_inputs = compute_dpsi_dt_from_inputs(t, x, parameters, parameters_unravel_fn, model, 'real')
    dpsi_dt_from_params = compute_dpsi_dt_from_params(t, x, parameters_unravel_fn(parameters), model, hamiltonian, qgt, expect_and_grad, 'real')
    loss = (jnp.abs(dpsi_dt_from_params - dpsi_dt_from_inputs) ** 2).mean()
    return loss
grad_consistency_loss_fn = jax.grad(consistency_loss_fn, argnums=3)

class TDVMC(AbstractVariationalDriver):
    """
    Energy minimization using Variational Monte Carlo (VMC).
    """

    def __init__(
        self,
        hamiltonian: AbstractOperator,
        optimizer,
        td_operator: TimeDerivativeOperator,
        l: float = 1e-3,
        *args,
        variational_state=None,
        preconditioner: PreconditionerT = identity_preconditioner,
        **kwargs,
    ):
        """
        Initializes the driver class.

        Args:
            hamiltonian: The Hamiltonian of the system.
            optimizer: Determines how optimization steps are performed given the
                bare energy gradient.
            preconditioner: Determines which preconditioner to use for the loss gradient.
                This must be a tuple of `(object, solver)` as documented in the section
                `preconditioners` in the documentation. The standard preconditioner
                included with NetKet is Stochastic Reconfiguration. By default, no
                preconditioner is used and the bare gradient is passed to the optimizer.
        """
        if variational_state is None:
            variational_state = MCState(*args, **kwargs)

        if variational_state.hilbert != hamiltonian.hilbert:
            raise TypeError(
                dedent(
                    f"""the variational_state has hilbert space {variational_state.hilbert}
                    (this is normally defined by the hilbert space in the sampler), but
                    the hamiltonian has hilbert space {hamiltonian.hilbert}.
                    The two should match.
                    """
                )
            )

        super().__init__(variational_state, optimizer, minimized_quantity_name="Energy")

        self._ham = hamiltonian.collect()  # type: AbstractOperator
        self._tdop = td_operator

        self.preconditioner = preconditioner

        self._dp: PyTree = None
        self._S = None
        self._sr_info = None

        self.l = l

    @property
    def preconditioner(self):
        """
        The preconditioner used to modify the gradient.

        This is a function with the following signature

        .. code-block:: python

            precondtioner(vstate: VariationalState,
                          grad: PyTree,
                          step: Optional[Scalar] = None)

        Where the first argument is a variational state, the second argument
        is the PyTree of the gradient to precondition and the last optional
        argument is the step, used to change some parameters along the
        optimisation.

        Often, this is taken to be :func:`nk.optimizer.SR`. If it is set to
        `None`, then the identity is used.
        """
        return self._preconditioner

    @preconditioner.setter
    def preconditioner(self, val: Optional[PreconditionerT]):
        if val is None:
            val = identity_preconditioner

        if len(signature(val).parameters) == 2:
            val = _DeprecatedPreconditionerSignature(val)

        self._preconditioner = val

    def _compute_consistency_loss_and_grad(self):
        # self.state.reset()
        t = jnp.array(0., dtype=jnp.complex128) # later need to sample this t
        x = self.state.samples
        qgt = self.state.quantum_geometric_tensor(QGTJacobianPyTree(holomorphic=True))
        # consistency_loss = consistency_loss_fn(t, x, self._ham, 
        #                                        *nkjax.tree_ravel(self.state.parameters), 
        #                                        self.state.model, 
        #                                        qgt, 
        #                                        self.state.expect_and_grad)
        grad_consistency_loss = grad_consistency_loss_fn(t, x, self._ham, 
                                                         *nkjax.tree_ravel(self.state.parameters), 
                                                         self.state.model, 
                                                         qgt, 
                                                         self.state.expect_and_grad)
        return grad_consistency_loss


    def sample_t(self, a, b, key=jax.random.PRNGKey(0)):
        return a + (b - a) * jax.random.uniform(key).astype(jnp.complex128)

    def _update_state_time(self, new_t_value):
        updated_variables = self.state.variables.unfreeze()
        if isinstance(new_t_value, (int, float)):
            new_t_value = jnp.array(new_t_value, dtype=jnp.complex128)
        updated_variables['batch_stats']['t'] = new_t_value
        self.state.variables = flax.core.freeze(updated_variables)
    
    def _combine_grad_by_mse(self, expt_H, grad_H, expt_dt, grad_dt):
        scale_factor = 1
        # _grad_total = jax.tree_map(
        #     lambda gH, gdt: 
        #     self.l * (scale_factor * 1j * gdt - gH).conjugate() * (scale_factor * 1j * expt_dt - expt_H) + self.l * (scale_factor * 1j * expt_dt - expt_H).conjugate() * (scale_factor * 1j * gdt - gH),
        #     grad_H, grad_dt
        #     )
        # _grad_total = jax.tree_map(
        #     lambda gH, gdt: 
        #     jnp.sign(gdt.real) * gdt,
        #     grad_H, grad_dt
        #     )
        _grad_total = jax.tree_map(
            lambda gH, gdt: 
            2 * self.l * (scale_factor * 1j * gdt - gH) * (scale_factor * 1j * expt_dt - expt_H),
            grad_H, grad_dt
            )
        return _grad_total

    def _forward_and_backward(self):
        """
        Performs a number of VMC optimization steps.

        Args:
            n_steps (int): Number of steps to perform.
        """

        self.state.reset()
        # self._update_state_time(1.0)
        t = self.sample_t(-1.0, 1.0)
        self._update_state_time(t)

        # Compute the local energy estimator and average Energy
        self._loss_stats, self._loss_grad = self.state.expect_and_grad(self._ham)

        self._tdop.t = self.state.variables['batch_stats']['t']
        self._tdop.model = self.state.model
        self._tdop.parameters = self.state.parameters

        self._td_expectation, self._grad_td_expectation = self.state.expect_and_grad(self._tdop)

        self._loss_grad_total = self._combine_grad_by_mse(self._loss_stats.mean, self._loss_grad, 
                                                     self._td_expectation.mean, self._grad_td_expectation)
        
        # _consistency_loss_grad = self._compute_consistency_loss_and_grad()
        # self._consistency_loss_grad = nkjax.tree_ravel(self.state.parameters)[1](_consistency_loss_grad)
        # self._loss_grad_total = jax.tree_map(lambda x, y: x + 1e-5 * y, self._loss_grad_total, self._consistency_loss_grad)

        # if it's the identity it does
        # self._dp = self._loss_grad
        self._dp = self.preconditioner(self.state, self._loss_grad_total, self.step_count)

        # If parameters are real, then take only real part of the gradient (if it's complex)
        self._dp = jax.tree_map(
            lambda x, target: (x if jnp.iscomplexobj(target) else x.real),
            self._dp,
            self.state.parameters,
        )

        return self._dp

    @property
    def energy(self) -> Stats:
        """
        Return MCMC statistics for the expectation value of observables in the
        current state of the driver.
        """
        return self._loss_stats

    def __repr__(self):
        return (
            "Vmc("
            + f"\n  step_count = {self.step_count},"
            + f"\n  state = {self.state})"
        )

    def info(self, depth=0):
        lines = [
            f"{name}: {info(obj, depth=depth + 1)}"
            for name, obj in [
                ("Hamiltonian    ", self._ham),
                ("Optimizer      ", self._optimizer),
                ("Preconditioner ", self.preconditioner),
                ("State          ", self.state),
            ]
        ]
        return "\n{}".format(" " * 3 * (depth + 1)).join([str(self)] + lines)
