import numpy as np
from scipy.spatial.distance import cdist

# Creates outliers for a SPECIFIC CLASS
# Scales the Gaussian so that features with greater variance are scaled proportionally: outliers can theoretically be in any direction
def infeasExam(X: np.array, 
                                 n_art: int,    # The number of artificial outliers to generate
                                 alpha: float,  # The multiplier
                                 epsilon: float, 
                                 mu: np.array = None, Sigma: np.array = None):
    """
    Generate artificial outliers by perturbing the original instances in X.

    Parameters:
        X (numpy.ndarray): The original instances (normal data points).
        n_art (int): The number of artificial outliers to generate.
        alpha (float): Scaling factor for the perturbation.
        epsilon (float): Minimum distance to be considered an outlier.
        
        mu (float): Mean of the normal distribution used for perturbation. 0 by default.
        sigma (float): Standard deviation of the normal distribution used for perturbation. Diagonal matrix of feature stdev's by default.

    Returns:
        numpy.ndarray: A 2D array containing artificial outliers
    """
    n, d = X.shape

    # Default parameters: zero-mean normal scaled to match standard deviation of each feature in the data
    if mu is None:
        mu = np.zeros((d,))
    if Sigma is None:
        feature_stdevs = np.std(X, axis=0)  # compute the stdev for each feature
        # print(f"feature_stdevs = {feature_stdevs}")
        Sigma = np.diag(feature_stdevs)

    # Apply a perturbation using alpha and the given distribution
    def perturb(instance: np.array) -> np.array:
        noise = np.random.multivariate_normal(mu, Sigma)
        return instance + alpha * noise

    # Determine whether this is a sufficient outlier using epsilon
    def is_sufficient_outlier(possible_outlier: np.array) -> bool:
        distances = cdist([possible_outlier], X, metric='euclidean')
        dist = np.min(distances)        # distance to closest normal instance
        return dist >= epsilon

    # Container for artificial outliers
    ArtOuts = []
    
    # Generate the first set of artificial outliers based on each normal point
    for i in range(n):
        art_i = perturb(X[i])

        if is_sufficient_outlier(art_i):
            ArtOuts.append(art_i)
    
    # If we already have enough, randomly sample n_art from these
    # If we don't have enough yet, repeat the process atop these artificial outliers
    if len(ArtOuts) >= n_art:
        row_indices = np.random.choice(len(ArtOuts), size = n_art, replace = False)
        X_art = np.array(ArtOuts)[row_indices]
    else:
        while len(ArtOuts) < n_art:
            out_i = ArtOuts[np.random.randint(0, len(ArtOuts))] # Randomly sample an existing artificial outlier
            
            art_i = perturb(out_i)              # Perturb
            if is_sufficient_outlier(art_i):    # Add if valid
                ArtOuts.append(art_i)

        X_art = np.array(ArtOuts)
    
    return X_art