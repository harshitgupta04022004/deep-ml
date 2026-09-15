import numpy as np
def solve_jacobi(A: np.ndarray, b: np.ndarray, n: int) -> list:
	
	D = np.diag(A)
	R = A-np.diag(D)
	x = np.zeros(A.shape[0])
	for _ in range(n):
		x = (b-R @ x)/D
	return x.tolist()
