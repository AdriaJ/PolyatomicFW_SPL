import numpy as np
import matplotlib.pyplot as plt

import pycsou.core as pcore
from pycsou.linop import DenseLinearOperator
import scipy.fft as sfft


########################

# Sampling opertors

class RealInputDFT(pcore.linop.LinearOperator):
    def __init__(self, side_length: int):
        self.explicit = False
        self.side_length = side_length
        self.rdft_shape = (self.side_length, self.side_length // 2 + 1)
        self.mat = None
        self.computed_columns_indices = np.array([])
        super(RealInputDFT, self).__init__(shape=(side_length * (side_length // 2 + 1),
                                                  side_length ** 2),
                                           dtype=np.complex128,
                                           lipschitz_cst=1.)

    def __call__(self, x):
        if x.ndim == 1:
            xx = x.reshape((self.side_length, self.side_length))
            xx_f = sfft.rfft2(xx, s=(self.side_length, self.side_length), norm="ortho").flatten()
        elif x.ndim == 2:  # horizontal stacking of input vectors
            xx = x.reshape((self.side_length, self.side_length, -1))
            xx_f = sfft.rfftn(xx, axes=(0, 1), norm="ortho").reshape((-1, x.shape[1]))
        else:
            raise ValueError("Input does not have an appropriate number of dimensions.")
        return xx_f

    def adjoint(self, y_f):
        # todo 1 transform the adjoint call so that it can handle even size inputs
        # todo 2 transform the adjoint call so that it can handle multidimensional inputs
        yy_f = y_f.reshape(self.rdft_shape)
        # sfft.irfft2(yy_f, s=(self.side_length, self.side_length), norm="ortho").flatten()
        # tmp = .5 * (sfft.irfft2(yy_f, s=(self.side_length, self.side_length), norm="ortho") -
        #              (1 / np.sqrt(self.side_length)) * sfft.ifft(yy_f[:, 0], #sfft.irfft(yy_f[: self.side_length // 2 + 1, 0],
        #                                                           n=self.side_length,
        #                                                           norm='ortho').reshape((-1, 1))).real + \
        #        (1 / np.sqrt(self.side_length)) * sfft.ifft(yy_f[:, 0], norm='ortho').real.reshape((-1, 1))
        tmp = .5 * (sfft.irfft2(yy_f, s=(self.side_length, self.side_length), norm="ortho") + (
                1 / np.sqrt(self.side_length) * sfft.ifft(yy_f[:, 0], norm='ortho').real.reshape((-1, 1))))
        return tmp.flatten()

    def get_restricted_operator(self, column_indices: np.ndarray) -> DenseLinearOperator:
        column_indices = np.asarray(column_indices)
        new_indices = np.setdiff1d(column_indices, self.computed_columns_indices, assume_unique=True)
        if new_indices.size > 0 :
            basis_vectors = canonical_basis(new_indices, size=self.shape[1])
            new_columns = self(basis_vectors)

            self.computed_columns_indices = np.hstack([self.computed_columns_indices, new_indices])
            if self.mat is None:
                self.mat = new_columns
            else:
                self.mat = np.hstack([self.mat, new_columns])
        assert np.isin(column_indices, self.computed_columns_indices, invert=True).sum() == 0
        # Find the location of the requested indices within the already computed indices
        # we have self.computed_columns_indices[sort_indices] = column_indices
        sort_indices = np.where(column_indices.reshape((-1, 1)) == self.computed_columns_indices.reshape((1, -1)))[1]

        restricted_stacked_mat = np.vstack([self.mat[:, sort_indices].real,
                                            self.mat[:, sort_indices].imag])
        op = DenseLinearOperator(restricted_stacked_mat)
        op.lipschitz_cst = 1.
        op.diff_lipschitz_cst = 1.
        return op


class SubSampledDFT(RealInputDFT):
    def __init__(self, side_length: int, flat_frequency_indices: np.ndarray):
        self.flat_indices = flat_frequency_indices
        super(SubSampledDFT, self).__init__(side_length=side_length)
        self.shape = (self.flat_indices.shape[0], side_length ** 2)
        self.freq_indices = np.unravel_index(self.flat_indices, shape=self.rdft_shape)

    def __call__(self, x):
        return (super().__call__(x))[self.flat_indices, ...]

    def adjoint(self, y_f):
        full_freq = np.zeros(self.side_length * (self.side_length // 2 + 1), dtype=complex)
        full_freq[self.flat_indices] = y_f
        return super().adjoint(full_freq)


def canonical_basis(index, size: int) -> np.ndarray:
    """

    Parameters
    ----------
    index : Can be int or array of int
    size : size of the output canonical basis vector(s)

    Returns
    -------

    """
    index = np.asarray(index)
    single_input = False
    if index.ndim == 0:
        index = index[None]
        single_input = True

    tmp = np.zeros((size, index.shape[0]))
    tmp[index, np.arange(index.shape[0])] = 1.

    if single_input:
        return np.squeeze(tmp)
    else:
        return tmp