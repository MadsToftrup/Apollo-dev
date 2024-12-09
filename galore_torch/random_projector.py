import torch
import math
from typing import Any, Callable, Dict, Generator, Iterable, Optional, Sequence, Union


def stable_randn(
    shape: Union[int, Sequence[int]],
    seed: int,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[torch.dtype] = torch.float32,
):
    if device is None:
        device = torch.device("cpu")
    generator = torch.Generator(device=device).manual_seed(seed)
    rn = torch.randn(shape, generator=generator, device=generator.device, dtype=dtype)
    return rn


def next_seed(seed: int, adv: int = 0xF):
    """
    This is a naive helper function to generate a new seed from the given seed.
    """
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(
        0, torch.iinfo(torch.int64).max, (adv,), generator=generator, device=generator.device
    ).tolist()[-1]


def split_seed(seed: int):
    generator = torch.Generator().manual_seed(seed)
    return tuple(
        torch.randint(0, torch.iinfo(torch.int64).max, (2,), generator=generator, device=generator.device).tolist()
    )


class GradientProjector:
    def __init__(
        self, rank, update_proj_gap=200, alpha=1.0, proj_type="std", seed=0
    ):
        # This is a lazy implementation as we store the projection matrix instead of re-generation every iteration
        self.rank = rank
        self.update_proj_gap = update_proj_gap
        self.alpha = alpha
        self.proj_type = proj_type

        self.ortho_matrix = None
        self.seed = seed

    def project(self, full_rank_grad, iter):

        if self.proj_type == "std":
            if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
                if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                    self.ortho_matrix = self.get_orthogonal_matrix(
                        full_rank_grad, self.rank, type="right", seed=self.seed
                    )
                    self.seed = next_seed(self.seed)
                low_rank_grad = torch.matmul(full_rank_grad, self.ortho_matrix.t())
            else:
                if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                    self.ortho_matrix = self.get_orthogonal_matrix(
                        full_rank_grad, self.rank, type="left", seed=self.seed
                    )
                    self.seed = next_seed(self.seed)
                    
                low_rank_grad = torch.matmul(self.ortho_matrix.t(), full_rank_grad)
        elif self.proj_type == "reverse_std":
            if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
                if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                    self.ortho_matrix = self.get_orthogonal_matrix(
                        full_rank_grad, self.rank, type="left", seed=self.seed
                    )
                    self.seed = next_seed(self.seed)
                    
                low_rank_grad = torch.matmul(self.ortho_matrix.t(), full_rank_grad)
            else:
                if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                    self.ortho_matrix = self.get_orthogonal_matrix(
                        full_rank_grad, self.rank, type="right", seed=self.seed
                    )
                    self.seed = next_seed(self.seed)
                low_rank_grad = torch.matmul(full_rank_grad, self.ortho_matrix.t())
        elif self.proj_type == "right":
            if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                self.ortho_matrix = self.get_orthogonal_matrix(
                    full_rank_grad, self.rank, type="right", seed=self.seed
                )
                self.seed = next_seed(self.seed)
            low_rank_grad = torch.matmul(full_rank_grad, self.ortho_matrix.t())
        elif self.proj_type == "left":
            if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                self.ortho_matrix = self.get_orthogonal_matrix(
                    full_rank_grad, self.rank, type="left", seed=self.seed
                )
                self.seed = next_seed(self.seed)
            low_rank_grad = torch.matmul(self.ortho_matrix.t(), full_rank_grad)
        elif self.proj_type == "full":
            if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                self.ortho_matrix = self.get_orthogonal_matrix(
                    full_rank_grad, self.rank, type="full", seed=self.seed
                )
                self.seed = next_seed(self.seed)
            low_rank_grad = (
                torch.matmul(self.ortho_matrix[0].t(), full_rank_grad)
                @ self.ortho_matrix[1].t()
            )

        return low_rank_grad

    # random low rank projection
    def get_orthogonal_matrix(self, weights, rank, type, seed):
        module_params = weights

        if module_params.data.dtype != torch.float:
            float_data = False
            original_type = module_params.data.dtype
            original_device = module_params.data.device
            matrix = module_params.data.float()
        else:
            float_data = True
            matrix = module_params.data

        if type == "left":
            proj = stable_randn(
                (matrix.shape[0], rank), seed=seed, device=matrix.device, dtype=matrix.dtype
            ) / math.sqrt(rank)
            if not float_data:
                proj = proj.to(original_device).type(original_type)
            return proj
        elif type == "right":
            proj = stable_randn(
                (rank, matrix.shape[1]), seed=seed, device=matrix.device, dtype=matrix.dtype
            ) / math.sqrt(rank)
            if not float_data:
                proj = proj.to(original_device).type(original_type)
            return proj
        elif type == "full":
            raise NotImplementedError("full rank projection is not implemented yet")
        else:
            raise ValueError("type should be left, right or full")
