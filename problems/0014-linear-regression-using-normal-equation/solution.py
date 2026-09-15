import numpy as np
def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
	# Your code here, make sure to round
	X_a = np.array(X)
	y_a = np.array(y)
	theta = np.linalg.inv(X_a.T@X_a)@X_a.T@y_a



	return theta