import numpy as np
from numpy.linalg import svd

def mp_median_exact(ncols, nrows):
    beta = ncols / nrows
    beta_minus = (1 - np.sqrt(beta))**2
    beta_plus  = (1 + np.sqrt(beta))**2
    
    # numeric integral of MP density
    def mp_density(x):
        return np.sqrt((beta_plus - x)*(x - beta_minus)) / (2*np.pi*beta*x)
    
    # find median via bisection
    lo, hi = beta_minus, beta_plus
    for _ in range(30):
        mid = (lo + hi) / 2
        xs = np.linspace(beta_minus, mid, 2000)
        ys = mp_density(xs)
        area = np.trapz(ys, xs)
        if area < 0.5:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def optimal_svht_coef(beta):
    """
    Optimal Hard Threshold coefficient (sigma unknown case)
    From Gavish & Donoho 2014.
    """
    if beta == 1:
        return 4 / np.sqrt(3)
    elif beta < 1:
        return optimal_svht_coef(1/beta) / np.sqrt(beta)

    # numerical formula approximation
    return np.sqrt(2*(beta + 1) + 8*beta/((beta+1) + np.sqrt(beta**2 + 14*beta + 1)))


def impute_column_mean(Z):
    """Impute missing (NaN/inf) values using column means."""
    Z = np.asarray(Z, dtype=float)
    Z_clean = Z.copy()

    # Replace infinities with NaN
    Z_clean[~np.isfinite(Z_clean)] = np.nan

    col_means = np.nanmean(Z_clean, axis=0)

    # Handle columns that are entirely NaN
    col_means = np.where(np.isnan(col_means), 0.0, col_means)

    # Broadcast replace
    inds = np.where(np.isnan(Z_clean))
    Z_clean[inds] = np.take(col_means, inds[1])

    return Z_clean


def estimate_optimal_dim(Z):
    """
    Estimate the optimal number of latent dimensions
    using the Optimal Hard Thresholding method (OHT).

    Z: matrix (e.g., z-score matrix), shape (n_samples, n_genes)
    """
    Z = np.asarray(Z)
    
    if not np.all(np.isfinite(Z)):
        Z = impute_column_mean(Z)
    
    n, p = Z.shape

    if p/n > 1:
        Z = Z.T
        n, p = Z.shape

    s = svd(Z, compute_uv=False)
    beta = p / n

    mp_med = mp_median_exact(p, n)
    coef = optimal_svht_coef(beta) / np.sqrt(mp_med)

    cutoff = coef * np.median(s)

    idx = np.where(s > cutoff)[0]

    if len(idx) == 0:
        rank = 2
    else:
        rank = idx[-1] + 1

    return rank, s, cutoff


def compute_zscore(adata, layers=["X_transformed","X_prepro","X_sf_norm","X_raw"]):
    
    for layer in layers:
        try:
            X = adata.layers[layer]
            print(f"Computing OHT Z-scores from layer {layer}")
            break
        except:
            continue
    
    row_means = np.nanmean(X, axis=1, keepdims=True)
    row_sds   = np.nanstd(X, axis=1, keepdims=True)

    Z = (X - row_means) / row_sds
    
    return Z