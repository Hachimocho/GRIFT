from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern


class AceV1():
    def __init__(self):
        super().__init__()
        
    def meta_optimize(self, scores, untested_combinations):
        pass
    
    # define the kernel function
    kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0))

    # create the GP model
    gp = GaussianProcessRegressor(kernel=kernel, alpha=0.1)

    # fit the GP model to the data
    gp.fit(X, y)

    # make predictions on untested combinations
    untested_combinations = ...
    predicted_scores, predicted_uncertainty = gp.predict(untested_combinations, return_std=True)