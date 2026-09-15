import torch

def transform_matrix(A, T, S) -> torch.Tensor:
    """
    Perform the change-of-basis transform T⁻¹ A S and round to 3 decimals using PyTorch.
    Inputs A, T, S can be Python lists, NumPy arrays, or torch Tensors.
    Returns a 2×2 tensor or tensor(-1.) if T or S is singular.
    """
    A_t = torch.as_tensor(A, dtype=torch.float)
    T_t = torch.as_tensor(T, dtype=torch.float)
    S_t = torch.as_tensor(S, dtype=torch.float)
    # Your implementation here
    if A_t.size !=A_t.size or S_t.size !=S_t.size:
        return -1
    try:
        T_t_1 = torch.linalg.inv(T_t)
        _ = torch.linalg.inv(S_t)
        result = T_t_1@A@S
        return result
    except  torch.linalg.LinAlgError:
        return -1
    
    
