import torch
from typing import List


def transform_basis(B: List[List[float]], C: List[List[float]]) -> List[List[float]]:
    """Return the change-of-basis matrix **P = C⁻¹ B**.

    - *B*, *C* may be 2×2 or 3×3 nested lists.
    - Result is rounded to 4 decimals and returned as a nested list.
    """
    # Your implementation here
    B_T = torch.tensor(B).float()
    C_inv_t = torch.linalg.inv(torch.tensor(C).float())
    return (C_inv_t@B_T).tolist()