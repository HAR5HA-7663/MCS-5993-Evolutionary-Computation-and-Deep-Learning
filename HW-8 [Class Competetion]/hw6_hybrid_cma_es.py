# HW6: Hybrid CMA-ES (μ+λ) Optimization Algorithm
# This implementation is designed for the class competition

import numpy as np

# Global RNG for reproducibility
rng = np.random.default_rng(42)

def set_seed(seed: int = 42):
    """Reset the RNG for consistent results across benchmark runs."""
    global rng
    rng = np.random.default_rng(seed)


def clip_to_bounds(x: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """
    Clip a candidate vector x to the given bounds.
    bounds has shape (dim, 2), where bounds[j, 0] is the lower bound
    and bounds[j, 1] is the upper bound for dimension j.
    """
    return np.clip(x, bounds[:, 0], bounds[:, 1])


def init_population(mu: int, bounds: np.ndarray) -> np.ndarray:
    """
    Initialize a population of mu parents uniformly inside the bounds.
    Returns an array of shape (mu, dim).
    """
    dim = bounds.shape[0]
    low = bounds[:, 0]
    high = bounds[:, 1]
    # sample uniformly for each dimension
    pop = rng.uniform(low=low, high=high, size=(mu, dim))
    return pop


def evaluate_population(pop: np.ndarray, objective) -> np.ndarray:
    """
    Evaluate all individuals in the population with the given objective function.
    Returns a 1D array of fitness values (lower is better).
    """
    # Homework objective functions are usually defined to take a 1D vector
    # and return a scalar, so I just loop over rows here.
    return np.array([objective(ind) for ind in pop])


def init_sigmas(bounds: np.ndarray, sigma_global_scale: float = 0.3):
    """
    Initialize global and local sigmas based on the variable ranges.
    Returns (sigma_g, sigma_local) where:
    - sigma_g is a scalar
    - sigma_local is a 1D array of shape (dim,)
    """
    dim = bounds.shape[0]
    ranges = bounds[:, 1] - bounds[:, 0]
    # avoid zero range just in case
    ranges = np.where(ranges == 0, 1.0, ranges)
    
    # global sigma starts as a fraction of average range
    sigma_g = sigma_global_scale * np.mean(ranges)
    
    # local sigmas start as a fraction of each dimension's range
    sigma_local = 0.3 * ranges
    
    return sigma_g, sigma_local


def update_local_sigmas_from_parents(parents: np.ndarray,
                                     sigma_local: np.ndarray,
                                     alpha_corr: float = 0.5,
                                     min_sigma: float = 1e-8) -> np.ndarray:
    """
    Update local sigmas σ_j using the covariance and correlation matrix
    of the current parent population.
    
    - Diagonal of covariance -> variance per dimension
    - Correlation structure -> increase σ_j when a dimension is strongly
      correlated with others.
    
    Returns the updated sigma_local.
    """
    mu, dim = parents.shape
    
    if mu < 2:
        # not enough parents to estimate covariance; just return existing sigmas
        return sigma_local
    
    # subtract mean so covariance focuses on variation
    centered = parents - np.mean(parents, axis=0, keepdims=True)
    
    # covariance matrix: shape (dim, dim)
    cov = np.cov(centered.T)
    
    # ensure symmetry and numerical stability
    cov = (cov + cov.T) / 2.0
    
    # diagonal -> variance, force non-negative
    var = np.diag(cov)
    var = np.maximum(var, 0.0)
    
    # base local sigma from variance
    base_sigma = np.sqrt(var + 1e-12)
    
    # build correlation matrix safely
    std = np.sqrt(var + 1e-12)
    denom = np.outer(std, std)
    # avoid division by zero
    denom = np.where(denom == 0.0, 1e-12, denom)
    corr = cov / denom
    
    # clip correlations to [-1, 1] just in case
    corr = np.clip(corr, -1.0, 1.0)
    
    # for each dimension, compute average absolute correlation with others
    avg_abs_corr = np.mean(np.abs(corr), axis=1)
    
    # scale base_sigma by (1 + alpha_corr * avg_abs_corr)
    new_sigma_local = base_sigma * (1.0 + alpha_corr * avg_abs_corr)
    
    # combine with previous sigma_local using a simple smoothing factor
    beta = 0.5  # 0 -> old only, 1 -> new only
    sigma_local = (1.0 - beta) * sigma_local + beta * new_sigma_local
    
    # enforce a minimum sigma to avoid freezing
    sigma_local = np.maximum(sigma_local, min_sigma)
    
    return sigma_local


def update_global_sigma(sigma_g: float,
                        success_rate: float,
                        target_success: float = 0.2,
                        a_inc: float = 1.2,
                        b_dec: float = 0.85,
                        min_sigma: float = 1e-8,
                        max_sigma: float = 1e6) -> float:
    """
    Update the global sigma σg using a 1/5-style success rule.
    
    - If success_rate > target_success, increase sigma by factor a_inc.
    - If success_rate < target_success, decrease sigma by factor b_dec.
    
    The result is clamped between min_sigma and max_sigma.
    """
    if success_rate > target_success:
        sigma_g *= a_inc
    elif success_rate < target_success:
        sigma_g *= b_dec
    
    sigma_g = max(min_sigma, min(max_sigma, sigma_g))
    
    return sigma_g


def hw6_hybrid_cma_es(objective,
                      bounds: np.ndarray,
                      dim: int,
                      n_generations: int = 200,
                      mu: int = 4,
                      lam: int = 10,
                      seed: int | None = None):
    """
    Hybrid CMA-ES (μ+λ) algorithm used for HW6.
    
    Parameters
    ----------
    objective : callable
        Objective function to minimize. Takes a 1D numpy array of length dim.
    bounds : np.ndarray
        Array of shape (dim, 2) with lower and upper bounds for each dimension.
    dim : int
        Dimension of the search space.
    n_generations : int
        Number of generations to run.
    mu : int
        Number of parents.
    lam : int
        Number of offspring per generation.
    seed : int or None
        Optional seed to reset the RNG for this run.
    
    Returns
    -------
    best_f : float
        Best fitness value found.
    best_x : np.ndarray
        Corresponding best solution vector.
    history : list[float]
        History of best fitness per generation.
    """
    if seed is not None:
        set_seed(seed)
    
    # make sure bounds have the right shape
    bounds = np.asarray(bounds, dtype=float)
    assert bounds.shape == (dim, 2)
    
    # initialize parents
    parents = init_population(mu, bounds)
    parent_f = evaluate_population(parents, objective)
    
    # initialize sigmas
    sigma_g, sigma_local = init_sigmas(bounds)
    
    # track best solution seen so far
    best_idx = np.argmin(parent_f)
    best_f = float(parent_f[best_idx])
    best_x = parents[best_idx].copy()
    
    history = [best_f]
    
    for gen in range(n_generations):
        # ----- generate offspring -----
        offspring = np.empty((lam, dim))
        
        for i in range(lam):
            # choose a parent index at random
            p_idx = rng.integers(0, mu)
            parent = parents[p_idx]
            
            # per-dimension step sizes
            step_sizes = sigma_g * sigma_local
            
            # Gaussian mutation
            step = rng.normal(loc=0.0, scale=step_sizes, size=dim)
            child = parent + step
            
            # clip into bounds
            child = clip_to_bounds(child, bounds)
            
            offspring[i] = child
        
        # evaluate offspring
        offspring_f = evaluate_population(offspring, objective)
        
        # ----- success rate for 1/5 rule -----
        # I define success as offspring better than the median parent fitness
        parent_baseline = np.median(parent_f)
        successes = np.sum(offspring_f < parent_baseline)
        success_rate = successes / max(1, lam)
        
        # update global sigma
        sigma_g = update_global_sigma(sigma_g, success_rate)
        
        # ----- combine parents and offspring for (μ+λ) selection -----
        combined = np.vstack([parents, offspring])
        combined_f = np.concatenate([parent_f, offspring_f])
        
        # sort by fitness (ascending, since we minimize)
        idx = np.argsort(combined_f)
        combined = combined[idx]
        combined_f = combined_f[idx]
        
        # keep best μ as new parents
        parents = combined[:mu]
        parent_f = combined_f[:mu]
        
        # update local sigmas from new parents
        sigma_local = update_local_sigmas_from_parents(parents, sigma_local)
        
        # track global best
        if parent_f[0] < best_f:
            best_f = float(parent_f[0])
            best_x = parents[0].copy()
        
        history.append(best_f)
    
    return best_f, best_x, history


# Wrapper function for integration with ESalgorithms10funcs.ipynb
def HW6(f, bounds, dim, n_generations=200, seed=None, **kwargs):
    """
    Competition-ready HW6 wrapper with multiple restarts for maximum robustness.
    
    Strategy:
    - Runs 2-3 independent restarts to avoid local minima
    - Adapts parameters based on search space characteristics
    - Uses larger population and more generations for better exploration
    - Returns the best result across all restarts
    
    This is designed to maximize performance on unknown/surprise functions.
    """
    # Convert bounds to numpy array
    bounds = np.asarray(bounds, dtype=float)
    
    # Adaptive parameter tuning based on search space
    range_size = np.mean(bounds[:, 1] - bounds[:, 0])
    
    # Determine number of restarts based on search space complexity
    if range_size > 100:  # Large search space - need more exploration
        n_restarts = 3
        gen_per_run = 350
        mu_val = 6
        lam_val = 15
    elif range_size > 10:  # Medium search space
        n_restarts = 2
        gen_per_run = 300
        mu_val = 5
        lam_val = 12
    else:  # Small search space
        n_restarts = 2
        gen_per_run = 250
        mu_val = 4
        lam_val = 10
    
    # Override with provided n_generations if explicitly set
    if n_generations != 200:
        gen_per_run = n_generations
        # Scale restarts if generations are limited
        if n_generations < 150:
            n_restarts = 1  # Single run if very limited time
    
    # Store best result across all restarts
    best_overall = float('inf')
    
    # Run multiple independent restarts
    for restart in range(n_restarts):
        # Use different seed for each restart to ensure diversity
        restart_seed = (seed + restart * 10000) if seed is not None else restart * 10000
        
        try:
            best_f, best_x, history = hw6_hybrid_cma_es(
                objective=f,
                bounds=bounds,
                dim=dim,
                n_generations=gen_per_run,
                mu=mu_val,
                lam=lam_val,
                seed=restart_seed
            )
            
            # Track best result across all restarts
            if best_f < best_overall:
                best_overall = best_f
        
        except Exception as e:
            # If one restart fails, continue with others
            print(f"Warning: Restart {restart} failed: {e}", flush=True)
            continue
    
    # If all restarts failed, fall back to single run with defaults
    if best_overall == float('inf'):
        best_f, _, _ = hw6_hybrid_cma_es(
            objective=f,
            bounds=bounds,
            dim=dim,
            n_generations=n_generations,
            mu=4,
            lam=10,
            seed=seed
        )
        return best_f
    
    return best_overall


if __name__ == "__main__":
    # Quick test on sphere function
    def sphere(x: np.ndarray) -> float:
        return float(np.sum(x**2))
    
    dim_test = 5
    bounds_test = np.array([[-5.0, 5.0]] * dim_test)
    
    print("Testing HW6 Hybrid CMA-ES on Sphere function...")
    best_f_test, best_x_test, history_test = hw6_hybrid_cma_es(
        objective=sphere,
        bounds=bounds_test,
        dim=dim_test,
        n_generations=200,
        mu=4,
        lam=10,
        seed=123,
    )
    
    print(f"Test best fitness: {best_f_test}")
    print(f"Test best solution: {best_x_test}")
    print(f"Initial fitness: {history_test[0]:.6f}")
    print(f"Final fitness: {history_test[-1]:.6f}")
    print(f"Improvement: {history_test[0] - history_test[-1]:.6f}")

