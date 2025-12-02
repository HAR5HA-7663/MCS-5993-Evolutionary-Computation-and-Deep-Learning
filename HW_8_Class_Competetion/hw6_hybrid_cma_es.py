# ==============================================================================
# HW6: Hybrid CMA-ES (μ+λ) Optimization Algorithm
# ==============================================================================
# Course: MCS-5993 Evolutionary Computation and Deep Learning
# Assignment: HW6 - Algorithm Competition
# Student: Harsha Yellela
# 
# I implement a hybrid CMA-ES-style Evolution Strategy that combines:
# - (μ+λ) population strategy with global and local step-size adaptation
# - Per-dimension step sizes learned from covariance matrix
# - 1/5 success rule for global step size control
# - Multiple restarts for competition robustness
# ==============================================================================

import numpy as np

# ==============================================================================
# Global RNG Setup
# ==============================================================================

# I use a global RNG for reproducibility across benchmark runs
rng = np.random.default_rng(42)


def set_seed(seed: int = 42):
    """
    Reset the global RNG with a new seed for reproducible experiments.
    
    Parameters
    ----------
    seed : int
        Random seed value for the RNG
    """
    global rng
    rng = np.random.default_rng(seed)


# ==============================================================================
# Helper Functions
# ==============================================================================


def clip_to_bounds(x: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """
    I implement this function to clip candidate solutions to valid bounds.
    
    This ensures all solutions stay within the problem's feasible region,
    which is important for bounded optimization problems.
    
    Parameters
    ----------
    x : np.ndarray
        Candidate solution vector of shape (dim,)
    bounds : np.ndarray
        Bounds array of shape (dim, 2) where bounds[j, 0] is lower bound
        and bounds[j, 1] is upper bound for dimension j
        
    Returns
    -------
    np.ndarray
        Clipped solution vector within bounds
    """
    return np.clip(x, bounds[:, 0], bounds[:, 1])


def init_population(mu: int, bounds: np.ndarray) -> np.ndarray:
    """
    I initialize a population of mu parents uniformly distributed within bounds.
    
    This provides diverse starting points for the evolution process.
    Uniform initialization helps explore the search space evenly at the start.
    
    Parameters
    ----------
    mu : int
        Number of parents to initialize
    bounds : np.ndarray
        Bounds array of shape (dim, 2)
        
    Returns
    -------
    np.ndarray
        Population array of shape (mu, dim) with individuals within bounds
    """
    dim = bounds.shape[0]
    low = bounds[:, 0]
    high = bounds[:, 1]
    
    # Sample uniformly for each dimension independently
    pop = rng.uniform(low=low, high=high, size=(mu, dim))
    return pop


def evaluate_population(pop: np.ndarray, objective) -> np.ndarray:
    """
    I evaluate all individuals in the population with the objective function.
    
    This step computes fitness for each population member. Since objective
    functions typically take a 1D vector, I loop over rows to evaluate each
    individual separately.
    
    Parameters
    ----------
    pop : np.ndarray
        Population array of shape (mu, dim)
    objective : callable
        Objective function that takes 1D array and returns scalar (lower is better)
        
    Returns
    -------
    np.ndarray
        1D array of fitness values, one per individual
    """
    return np.array([objective(ind) for ind in pop])


# ==============================================================================
# Step-Size Initialization and Adaptation
# ==============================================================================


def init_sigmas(bounds: np.ndarray, sigma_global_scale: float = 0.3):
    """
    I initialize global and local step sizes based on variable ranges.
    
    This step sets initial mutation strengths. Global sigma controls overall
    exploration, while local sigmas allow per-dimension adaptation. I scale
    both relative to the search space size to ensure appropriate initial
    step sizes.
    
    Parameters
    ----------
    bounds : np.ndarray
        Bounds array of shape (dim, 2)
    sigma_global_scale : float
        Scale factor for global sigma (default: 0.3 of average range)
        
    Returns
    -------
    tuple
        (sigma_g, sigma_local) where:
        - sigma_g: scalar global step size
        - sigma_local: 1D array of per-dimension step sizes, shape (dim,)
    """
    dim = bounds.shape[0]
    ranges = bounds[:, 1] - bounds[:, 0]
    
    # Avoid zero range to prevent division issues
    ranges = np.where(ranges == 0, 1.0, ranges)
    
    # Global sigma starts as fraction of average range
    # This provides reasonable initial exploration scale
    sigma_g = sigma_global_scale * np.mean(ranges)
    
    # Local sigmas start as fraction of each dimension's range
    # This allows different dimensions to have different initial step sizes
    sigma_local = 0.3 * ranges
    
    return sigma_g, sigma_local


def update_local_sigmas_from_parents(parents: np.ndarray,
                                     sigma_local: np.ndarray,
                                     alpha_corr: float = 0.5,
                                     min_sigma: float = 1e-8) -> np.ndarray:
    """
    I update local step sizes σ_j using the covariance matrix of parents.
    
    This step learns which dimensions need more or less exploration by
    analyzing the parent population's distribution. Dimensions with high
    variance need larger steps for exploration, while dimensions with low
    variance are near optima and need smaller steps for refinement.
    
    I also consider correlation: dimensions correlated with others benefit
    from increased mutation to explore correlated regions more effectively.
    
    Parameters
    ----------
    parents : np.ndarray
        Parent population of shape (mu, dim)
    sigma_local : np.ndarray
        Current local step sizes of shape (dim,)
    alpha_corr : float
        Correlation scaling factor (default: 0.5)
    min_sigma : float
        Minimum allowed step size to avoid freezing (default: 1e-8)
        
    Returns
    -------
    np.ndarray
        Updated local step sizes of shape (dim,)
    """
    mu, dim = parents.shape
    
    # Need at least 2 parents to estimate covariance
    if mu < 2:
        return sigma_local
    
    # Center parents to focus on variation around mean
    centered = parents - np.mean(parents, axis=0, keepdims=True)
    
    # Compute covariance matrix to learn dimension relationships
    cov = np.cov(centered.T)
    
    # Ensure symmetry for numerical stability
    cov = (cov + cov.T) / 2.0
    
    # Extract variance per dimension (diagonal of covariance)
    var = np.diag(cov)
    var = np.maximum(var, 0.0)  # Force non-negative
    
    # Base local sigma from variance - high variance = need larger steps
    base_sigma = np.sqrt(var + 1e-12)
    
    # Build correlation matrix to detect dimension relationships
    std = np.sqrt(var + 1e-12)
    denom = np.outer(std, std)
    denom = np.where(denom == 0.0, 1e-12, denom)  # Avoid division by zero
    corr = cov / denom
    
    # Clip correlations to valid range
    corr = np.clip(corr, -1.0, 1.0)
    
    # For each dimension, compute average absolute correlation with others
    # Highly correlated dimensions benefit from increased mutation
    avg_abs_corr = np.mean(np.abs(corr), axis=1)
    
    # Scale base sigma by correlation factor
    new_sigma_local = base_sigma * (1.0 + alpha_corr * avg_abs_corr)
    
    # Smooth update to avoid oscillations
    # Beta controls how much we trust new values vs old values
    beta = 0.5  # 50% old, 50% new
    sigma_local = (1.0 - beta) * sigma_local + beta * new_sigma_local
    
    # Enforce minimum to prevent freezing
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
    I update global step size using the 1/5 success rule.
    
    This step adapts overall exploration based on how often mutations succeed.
    If success rate is high (>20%), we increase sigma to explore more.
    If success rate is low (<20%), we decrease sigma to refine locally.
    This implements the classic 1/5 success rule from ES literature.
    
    Parameters
    ----------
    sigma_g : float
        Current global step size
    success_rate : float
        Fraction of successful mutations (0.0 to 1.0)
    target_success : float
        Target success rate, typically 0.2 for 1/5 rule
    a_inc : float
        Multiplicative increase factor when above target (default: 1.2)
    b_dec : float
        Multiplicative decrease factor when below target (default: 0.85)
    min_sigma : float
        Minimum allowed step size (default: 1e-8)
    max_sigma : float
        Maximum allowed step size (default: 1e6)
        
    Returns
    -------
    float
        Updated global step size
    """
    # Increase if we're succeeding too often (need more exploration)
    if success_rate > target_success:
        sigma_g *= a_inc
    # Decrease if we're succeeding too rarely (need more refinement)
    elif success_rate < target_success:
        sigma_g *= b_dec
    
    # Clamp to valid range
    sigma_g = max(min_sigma, min(max_sigma, sigma_g))
    
    return sigma_g


# ==============================================================================
# Main HW6 Algorithm
# ==============================================================================


def hw6_hybrid_cma_es(objective,
                      bounds: np.ndarray,
                      dim: int,
                      n_generations: int = 200,
                      mu: int = 4,
                      lam: int = 10,
                      seed: int | None = None):
    """
    I implement the main HW6 hybrid CMA-ES (μ+λ) algorithm.
    
    This algorithm combines population-based selection with dual step-size
    adaptation. The key innovation is using both global and local step sizes,
    allowing independent per-dimension adaptation while maintaining overall
    convergence control through the global step size.
    
    Parameters
    ----------
    objective : callable
        Objective function to minimize. Takes 1D numpy array of length dim,
        returns scalar (lower is better).
    bounds : np.ndarray
        Array of shape (dim, 2) with [lower, upper] bounds per dimension.
    dim : int
        Dimension of the search space.
    n_generations : int
        Number of generations to run (default: 200).
    mu : int
        Number of parents to maintain (default: 4).
    lam : int
        Number of offspring to generate per generation (default: 10).
    seed : int or None
        Optional random seed for reproducibility.
        
    Returns
    -------
    tuple
        (best_f, best_x, history) where:
        - best_f: float, best fitness value found
        - best_x: np.ndarray, best solution vector found
        - history: list[float], best fitness per generation
    """
    # Set seed for reproducibility if provided
    if seed is not None:
        set_seed(seed)
    
    # Validate bounds shape
    bounds = np.asarray(bounds, dtype=float)
    assert bounds.shape == (dim, 2), "Bounds must have shape (dim, 2)"
    
    # Initialize parent population uniformly within bounds
    parents = init_population(mu, bounds)
    parent_f = evaluate_population(parents, objective)
    
    # Initialize step sizes based on search space ranges
    sigma_g, sigma_local = init_sigmas(bounds)
    
    # Track best solution seen so far
    best_idx = np.argmin(parent_f)
    best_f = float(parent_f[best_idx])
    best_x = parents[best_idx].copy()
    
    # Store history for convergence analysis
    history = [best_f]
    
    # Main evolution loop
    for gen in range(n_generations):
        # Generate λ offspring through mutation
        offspring = np.empty((lam, dim))
        
        for i in range(lam):
            # Select random parent for reproduction
            p_idx = rng.integers(0, mu)
            parent = parents[p_idx]
            
            # Compute per-dimension step sizes (global × local)
            # This allows each dimension to adapt independently
            step_sizes = sigma_g * sigma_local
            
            # Gaussian mutation with per-dimension scaling
            step = rng.normal(loc=0.0, scale=step_sizes, size=dim)
            child = parent + step
            
            # Ensure child stays within bounds
            child = clip_to_bounds(child, bounds)
            offspring[i] = child
        
        # Evaluate all offspring
        offspring_f = evaluate_population(offspring, objective)
        
        # Compute success rate for 1/5 rule
        # I define success as offspring better than median parent fitness
        # This is more robust than comparing to best parent
        parent_baseline = np.median(parent_f)
        successes = np.sum(offspring_f < parent_baseline)
        success_rate = successes / max(1, lam)
        
        # Update global step size using 1/5 rule
        sigma_g = update_global_sigma(sigma_g, success_rate)
        
        # (μ+λ) selection: combine parents and offspring, keep best μ
        combined = np.vstack([parents, offspring])
        combined_f = np.concatenate([parent_f, offspring_f])
        
        # Sort by fitness (ascending, since we minimize)
        idx = np.argsort(combined_f)
        combined = combined[idx]
        combined_f = combined_f[idx]
        
        # Select best μ as new parents
        parents = combined[:mu]
        parent_f = combined_f[:mu]
        
        # Update local step sizes from new parent distribution
        # This step learns which dimensions need more/less exploration
        sigma_local = update_local_sigmas_from_parents(parents, sigma_local)
        
        # Update global best if we found something better
        if parent_f[0] < best_f:
            best_f = float(parent_f[0])
            best_x = parents[0].copy()
        
        # Record history for convergence tracking
        history.append(best_f)
    
    return best_f, best_x, history


# ==============================================================================
# Competition Wrapper Function
# ==============================================================================


def HW6(f, bounds, dim, n_generations=200, seed=None, **kwargs):
    """
    I implement a competition-ready wrapper with multiple restarts.
    
    This wrapper runs multiple independent restarts to avoid local minima,
    which is crucial for competition settings. I adapt parameters based on
    search space characteristics: larger spaces get more exploration via
    bigger populations and more restarts.
    
    This design maximizes performance on unknown competition functions by
    balancing exploration and exploitation across multiple independent runs.
    
    Parameters
    ----------
    f : callable
        Objective function to minimize (takes 1D array, returns scalar).
    bounds : np.ndarray
        Bounds array of shape (dim, 2).
    dim : int
        Search space dimension.
    n_generations : int
        Number of generations per run (default: 200).
    seed : int or None
        Random seed for reproducibility.
    **kwargs
        Additional arguments (ignored for compatibility).
        
    Returns
    -------
    float
        Best fitness found across all restarts.
    """
    # Convert bounds to numpy array
    bounds = np.asarray(bounds, dtype=float)
    
    # Compute search space size to adapt parameters
    range_size = np.mean(bounds[:, 1] - bounds[:, 0])
    
    # Adaptive parameter tuning based on search space complexity
    # Larger spaces need more exploration (more restarts, bigger populations)
    if range_size > 100:  # Large search space
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
    
    # Override generations if explicitly set
    if n_generations != 200:
        gen_per_run = n_generations
        # Reduce restarts if generations are very limited
        if n_generations < 150:
            n_restarts = 1
    
    # Track best result across all restarts
    best_overall = float('inf')
    
    # Run multiple independent restarts
    for restart in range(n_restarts):
        # Use different seed for each restart to ensure diversity
        restart_seed = (seed + restart * 10000) if seed is not None else restart * 10000
        
        try:
            # Run one evolution process
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
    
    # Fallback to single run if all restarts failed
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


# ==============================================================================
# Testing
# ==============================================================================


if __name__ == "__main__":
    # Quick test on sphere function to verify implementation
    def sphere(x: np.ndarray) -> float:
        """Sphere function for testing."""
        return float(np.sum(x**2))
    
    dim_test = 5
    bounds_test = np.array([[-5.0, 5.0]] * dim_test)
    
    print("Testing HW6 Hybrid CMA-ES on Sphere function...")
    print(f"Dimension: {dim_test}, Generations: 200")
    
    best_f_test, best_x_test, history_test = hw6_hybrid_cma_es(
        objective=sphere,
        bounds=bounds_test,
        dim=dim_test,
        n_generations=200,
        mu=4,
        lam=10,
        seed=123,
    )
    
    print(f"\nResults:")
    print(f"  Best fitness: {best_f_test:.6e}")
    print(f"  Best solution: {best_x_test}")
    print(f"  Initial fitness: {history_test[0]:.6f}")
    print(f"  Final fitness: {history_test[-1]:.6e}")
    print(f"  Improvement: {history_test[0] - history_test[-1]:.6f}")
