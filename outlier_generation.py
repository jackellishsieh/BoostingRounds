import numpy as np
from scipy.spatial.distance import cdist

# Creates outliers for a SPECIFIC CLASS
# Scales the Gaussian so that features with greater variance are scaled proportionally: outliers can theoretically be in any direction
def generate_artificial_outliers(X: np.array, 
                                 n_art: int,    # The number of artificial outliers to generate
                                 alpha: float,  # The multiplier
                                 epsilon: float, 
                                 mu: np.array = None, Sigma: np.array = None):
    """
    Generate artificial outliers by perturbing the original instances in X.

    Parameters:
        X (numpy.ndarray): The original instances (normal data points).
        nart (int): The number of artificial outliers to generate.
        mu (float): Mean of the normal distribution used for perturbation.
        sigma (float): Standard deviation of the normal distribution used for perturbation.
        alpha (float): Scaling factor for the perturbation.
        epsilon (float): Minimum distance to be considered an outlier.

    Returns:
        numpy.ndarray: A 2D array containing both original and artificial outliers.
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
    
    # Step 1: Generate the first set of artificial outliers
    for i in range(n):
        # Step 2: Create artificial instance by perturbing this instance
        art_i = perturb(X[i])
        
        # Step 3: If this qualifies as an outlier, add
        if is_sufficient_outlier(art_i):
            ArtOuts.append(art_i)
    
    # Step 4a: if we already have enough, randomly sample n_art from these
    if len(ArtOuts) >= n_art:
        row_indices = np.random.choice(len(ArtOuts), size = n_art, replace = False)
        X_art = np.array(ArtOuts)[row_indices]
    
    # Step 4b: if we don't have enough, repeat the process atop these artificial outliers
    else:
        while len(ArtOuts) < n_art:
            # Randomly sample an existing artificial outlier
            out_i = ArtOuts[np.random.randint(0, len(ArtOuts))]

            # Perturb
            art_i = perturb(out_i)

            # Add if valid
            if is_sufficient_outlier(art_i):
                ArtOuts.append(art_i)

        X_art = np.array(ArtOuts)
    
    # Return the artificial outliers
    return X_art