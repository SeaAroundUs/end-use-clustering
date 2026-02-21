from sklearn.decomposition import NMF
import pandas as pd

def dimension_reduction(X, n_components=4, init="nndsvd", random_state=0):
    """
    Perform Non-negative Matrix Factorization (NMF) for dimensionality reduction.

    Parameters:
    X (array-like): The input data matrix (n_samples, n_features).
    n_components (int): The number of components to keep. Default is 4.
    init (str): Method used to initialize the procedure. Must be 'random' or 'nndsvd'. Default is 'nndsvd'.
    random_state (int): Random state for reproducibility. Default is 0.

    Returns:
    W (DataFrame): The matrix of shape (n_samples, n_components) containing the mixture of components for each sample with values normalized per sample.
    H (DataFrame): The matrix of shape (n_components, n_features) containing the definition of each component in terms of the original features.
    """
    model = NMF(n_components=n_components, init=init, random_state=random_state)
    W = model.fit_transform(X)
    H = model.components_

    # Get country and category names for DataFrame indexing
    countries = X.index
    categories = X.columns

    # W (country mixtures)
    W_df = pd.DataFrame(W,
                        index=countries,
                        columns=[f"Type {i+1}" for i in range(n_components)])
    
    # Normalize W to percentages per country
    W_df = W_df.div(W_df.sum(axis=1), axis=0)

    # H (component definitions)
    H_df = pd.DataFrame(H,
                        columns=categories,
                        index=[f"Type {i+1}" for i in range(n_components)])
    
    return W_df, H_df