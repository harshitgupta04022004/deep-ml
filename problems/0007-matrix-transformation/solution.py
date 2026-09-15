import numpy as np

def transform_matrix(
    A: list[list[int|float]], 
    T: list[list[int|float]], 
    S: list[list[int|float]]
) -> list[list[int|float]] | int:

    A_arr = np.array(A)
    T_arr = np.array(T)
    S_arr = np.array(S)

    if T_arr.shape != T_arr.shape or S_arr.shape != S_arr.shape:
        return -1

    try:
        T_inv = np.linalg.inv(T_arr)
        _ = np.linalg.inv(S_arr) 
        
        result = T_inv @ A_arr @ S_arr
        return result.tolist()
        
    except np.linalg.LinAlgError:
        return -1
