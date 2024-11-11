import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial.distance import mahalanobis

# Custom gauss tail method (bad, doesn't really work as expected)
def customGaussTail(X, n_art, M):
    """
    Generate n_art outliers sampled from a Gaussian distribution 
    with a Mahalanobis distance of at least M from the mean.

    Parameters:
    X (ndarray): An nxd array representing n data points of d dimensions.
    n_art (int): Number of outliers to generate.
    M (float): Minimum Mahalanobis distance of outliers from the mean.

    Returns:
    ndarray: An n_art x d array of generated outliers.
    """
    
    # Step 1: Estimate the Gaussian distribution (mean and covariance)
    mean = np.mean(X, axis=0)  # Mean of the data
    cov = np.cov(X, rowvar=False)  # Covariance matrix of the data
    inv_cov = np.linalg.inv(cov)  # Inverse of the covariance matrix

    # Prepare to store the generated outliers
    outliers = []

    # Step 2: Generate outliers using rejection sampling
    while len(outliers) < n_art:
        # Sample from the Gaussian distribution
        sample = np.random.multivariate_normal(mean, cov)
        
        # Step 3: Compute the Mahalanobis distance of the sample from the mean
        dist = mahalanobis(sample, mean, inv_cov)       # must provide the inverse
        
        # Accept the sample if the Mahalanobis distance is at least M
        if dist >= M:
            outliers.append(sample)
    
    return np.array(outliers)


# Create outliers using margin sampling
def marginSample(X, n_art, factor = 3):
    _, d = X.shape
    
    # Calculate the mean (µ_i) and standard deviation (σ_i) for each feature
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)
    
    # Generate outliers
    outliers = np.zeros((n_art, d))    

    for i in range(d):
        for k in range(n_art):
            # Generate each feature just by sampling

            outliers[k, i] = np.random.normal(loc=means[i], scale=stds[i])

    return outliers

# Create outliers using the Gaussian-Tail method
def gaussTail(X, n_art, factor = 3):
    _, d = X.shape
    
    # Calculate the mean (µ_i) and standard deviation (σ_i) for each feature
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)
    
    # Define the gap region (within ±3σ_i around µ_i)
    lower_bounds = means - factor * stds
    upper_bounds = means + factor * stds

    # Generate outliers
    outliers = np.zeros((n_art, d))    

    for i in range(d):
        for k in range(n_art):
            # Generate each feature by rejection sampling
        
            while True:
                outliers[k, i] = np.random.normal(loc=means[i], scale=stds[i])

                if outliers[k, i] <= lower_bounds[i] or outliers[k, i] >= upper_bounds[i]:
                    break
    
    return outliers


# Creates outliers using the hyperrectangle method
def unifBox(X, n_art, factor = 1):             # factor = expand the boundary from the center by this much
    # Compute the min and max values for each column (dimension)
    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)
    
    lengths = max_vals - min_vals

    lower_bounds = min_vals - (factor - 1) / 2 * lengths
    upper_bounds = max_vals + (factor - 1) / 2 * lengths

    # Randomly sample n_art vectors from the bounding hyperrectangle
    sampled_vectors = np.random.uniform(lower_bounds, upper_bounds, (n_art, X.shape[1]))
    
    return sampled_vectors

# Creates outliers for a SPECIFIC CLASS using the Infeas algorithm
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