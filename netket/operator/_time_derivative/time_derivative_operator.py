from typing import Union, Optional

import numbers

from textwrap import dedent

import numpy as np
import jax.numpy as jnp
import numba
from scipy.sparse import issparse

import jax
import netket.jax as nkjax
from netket.hilbert import AbstractHilbert, DiscreteHilbert
from netket.utils.types import DType, Array
from netket.utils.numbers import dtype as _dtype, is_scalar
from netket.errors import concrete_or_error, NumbaOperatorGetConnDuringTracingError

from .._discrete_operator import DiscreteOperator
from .._lazy import Transpose

def exp_model_output(t, x, params, unravel_func, model, component):
    out =  model.apply({"params": unravel_func(params)}, x, t)
    if component == 'real':
        return out.real
    elif component == 'imag':
        return out.imag
    elif component == 'complex':
        return out

# Gradient with respect to t
_grad_dpsi_dt = jax.grad(exp_model_output, argnums=0)
func_dpsi_dt = jax.vmap(_grad_dpsi_dt, in_axes=(None, 0, None, None, None, None))

class TimeDerivativeOperator(DiscreteOperator):

    def __init__(
        self,
        hilbert: DiscreteHilbert,
        dtype: Optional[DType] = None,
    ):
        super().__init__(hilbert)
        self.mel_cutoff = 1.0e-6
        self._initialized = None
        self._is_hermitian = True

        self._dtype = _dtype(dtype)
    
    @property
    def is_hermitian(self):
        return self._is_hermitian

    @property
    def t(self):
        if not hasattr(self, '_t'):
            self._t = jnp.array(0.0, dtype=self.dtype)
        return self._t
    
    @t.setter
    def t(self, value):
        # Ensure the incoming value is a jnp array
        if not isinstance(value, jnp.ndarray):
            value = jnp.array(value)
        # Ensure it has shape ()
        if value.shape != ():
            raise ValueError(f"Expected shape () for 't', but got {value.shape}")
        # Ensure it has the correct dtype
        if value.dtype != self.dtype:
            value = value.astype(self.dtype)
        self._t = value

    @property
    def model(self):
        return self._model
    
    @model.setter
    def model(self, _model):
        self._model = _model

    @property
    def parameters(self):
        return self._parameters
    
    @parameters.setter
    def parameters(self, _parameters):
        self._parameters = _parameters
    
    def get_conn_padded(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r"""Finds the connected elements of the Operator.

        Starting from a batch of quantum numbers :math:`x={x_1, ... x_n}` of
        size :math:`B \times M` where :math:`B` size of the batch and :math:`M`
        size of the hilbert space, finds all states :math:`y_i^1, ..., y_i^K`
        connected to every :math:`x_i`.

        Returns a matrix of size :math:`B \times K_{max} \times M` where
        :math:`K_{max}` is the maximum number of connections for every
        :math:`y_i`.

        Args:
            x : A N-tensor of shape :math:`(...,hilbert.size)` containing
                the batch/batches of quantum numbers :math:`x`.
        Returns:
            **(x_primes, mels)**: The connected states x', in a N+1-tensor and an
            N-tensor containing the matrix elements :math:`O(x,x')`
            associated to each x' for every batch.
        """
        n_visible = x.shape[-1]
        n_samples = x.size // n_visible

        sections = np.empty(n_samples, dtype=np.int32)
        x_primes, mels = self.get_conn_flattened(
            x.reshape(-1, x.shape[-1]), sections, pad=True
        )

        x_primes_r = x_primes.reshape(*x.shape[:-1], 1, n_visible)
        mels_r = mels.reshape(*x.shape[:-1], 1)

        return x_primes_r, mels_r
    
    def get_conn_flattened(
        self, x: np.ndarray, sections: np.ndarray, pad=False
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""Finds the connected elements of the Operator.

        Starting from a given quantum number :math:`x`, it finds all
        other quantum numbers  :math:`x'` such that the matrix element
        :math:`O(x,x')` is different from zero. In general there will be
        several different connected states :math:`x'` satisfying this
        condition, and they are denoted here :math:`x'(k)`, for
        :math:`k=0,1...N_{\mathrm{connected}}`.

        This is a batched version, where x is a matrix of shape
        :code:`(batch_size,hilbert.size)`.

        Args:
            x: A matrix of shape `(batch_size, hilbert.size)`
                containing the batch of quantum numbers x.
            sections: An array of sections for the flattened x'.
                See numpy.split for the meaning of sections.
        Returns:
            (matrix, array): The connected states x', flattened together in
                a single matrix.
                An array containing the matrix elements :math:`O(x,x')`
                associated to each x'.

        """
        x_prime = x.copy()
        rav_parameters, unravel_fn = nkjax.tree_ravel(self.parameters)
        logpsi = exp_model_output(self.t, x_prime, rav_parameters, unravel_fn, self.model, 'complex').reshape(x.shape[:-1])
        mels = func_dpsi_dt(self.t, x.reshape(-1,x.shape[-1]), rav_parameters, unravel_fn, self.model, 'real').reshape(x.shape[:-1])
        mels = jnp.exp(logpsi) * mels
        return x_prime, mels
    
    
    @property
    def dtype(self) -> DType:
        return self._dtype
