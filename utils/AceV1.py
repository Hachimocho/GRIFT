from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C    
    
class ACE():
    def __init__(self):
        pass
    
    def meta_optimize(self, unfinished_sweeps, finished_sweeps):
        #This function takes a list of unfinished sweeps and a list of finished sweeps and their scores, and uses a gaussian process regressor to predict the performance of the unfinished sweeps.
        # Create a list of the modules used in each sweep
        modules_used = []
        for sweep in finished_sweeps + unfinished_sweeps:
            modules_used.append([module.__name__ for module in sweep[1]["parameters"]])
        # Create a kernel for the gaussian process
        kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
        # Create a gaussian process regressor
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
        # Train the gaussian process regressor on the finished sweeps
        X = np.array(modules_used[:len(finished_sweeps)])
        y = np.array([sweep[0] for sweep in finished_sweeps])
        gpr.fit(X, y)

        # Predict the performance of the unfinished sweeps
        X_pred = np.array(modules_used[len(finished_sweeps):])
        y_pred = gpr.predict(X_pred)

        # Sort the unfinished sweeps by their predicted performance
        sorted_sweeps = sorted(zip(y_pred, unfinished_sweeps), reverse=True)

        # Return the sorted list of unfinished sweeps
        return [sweep[1] for sweep in sorted_sweeps]