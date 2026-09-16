import numpy as np

def transform_basis(B: list[list[int]], C: list[list[int]]) -> list[list[float]]:
	B_t = np.array(B)
	C_inv_t = np.linalg.inv(np.array(C))
	return C_inv_t@B_t