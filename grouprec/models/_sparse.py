"""CSR <-> torch glue for the deep models.

The user-item matrix is >99% sparse, so we hold it as ``scipy.sparse.csr_matrix`` and
densify **only the rows a batch touches**. A MovieLens-32M interaction matrix is ~50 GB
dense but ~400 MB as CSR; a 1024-row batch is ~80 MB dense, which is what actually
reaches the GPU.

``CsrRows`` is indexed exactly like the dense tensor it replaces -- ``M[i]``,
``M[a:b]``, ``M[[i, j, k]]``, ``M.shape`` -- and returns ``torch.Tensor`` each time, so
model code needs no changes.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy import sparse


class CsrRows:
    """A CSR matrix that yields dense ``torch`` rows on indexing.

    Parameters
    ----------
    matrix : scipy CSR user-item matrix.
    device : torch device the returned tensors land on.
    dtype : torch dtype of the returned tensors.
    """

    __slots__ = ("m", "device", "dtype")

    def __init__(self, matrix: sparse.csr_matrix, device, dtype=torch.float32) -> None:
        self.m = matrix.tocsr()
        self.device = device
        self.dtype = dtype

    @property
    def shape(self):
        return self.m.shape

    def __len__(self) -> int:
        return self.m.shape[0]

    def _dense(self, block) -> torch.Tensor:
        return torch.as_tensor(np.asarray(block.todense()), dtype=self.dtype,
                               device=self.device)

    def __getitem__(self, key) -> torch.Tensor:
        # Mirror dense-ndarray semantics exactly:
        #   M[i]        -> (n_items,)
        #   M[a:b]      -> (b-a, n_items)
        #   M[[i,j]]    -> (2, n_items)
        #   M[idx_2d]   -> (*idx_2d.shape, n_items)   <- CSR cannot index into >2D,
        #                  so flatten, gather rows, then restore the leading shape.
        if isinstance(key, (int, np.integer)):
            return self._dense(self.m[int(key)]).reshape(-1)
        if isinstance(key, (list, np.ndarray)):
            idx = np.asarray(key, dtype=np.int64)
            if idx.size == 0:
                return torch.zeros((*idx.shape, self.m.shape[1]), dtype=self.dtype,
                                   device=self.device)
            if idx.ndim >= 2:
                flat = self._dense(self.m[idx.reshape(-1)])
                return flat.reshape(*idx.shape, self.m.shape[1])
            return self._dense(self.m[idx])
        return self._dense(self.m[key])

    def to_torch_csr(self) -> torch.Tensor:
        """The whole matrix as a torch sparse CSR tensor (no densification).

        For ops that accept sparse operands (e.g. ``torch.sparse.mm``). Row-wise
        batching via ``__getitem__`` is preferred for training.
        """
        m = self.m.tocsr()
        return torch.sparse_csr_tensor(
            torch.as_tensor(m.indptr, dtype=torch.int64),
            torch.as_tensor(m.indices, dtype=torch.int64),
            torch.as_tensor(m.data, dtype=self.dtype),
            size=m.shape, device=self.device,
        )
