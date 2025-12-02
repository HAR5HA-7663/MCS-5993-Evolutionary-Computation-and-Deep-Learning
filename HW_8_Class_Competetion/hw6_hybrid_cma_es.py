# HW6: Hybrid CMA-ES (μ+λ) Optimization Algorithm
# Course: MCS-5993 Evolutionary Computation and Deep Learning
# Assignment: HW6 - Algorithm Competition
# Student: Harsha Yellela

import numpy as np

# global RNG for reproducibility
rng = np.random.default_rng(42)

def set_seed(seed: int = 42):
    """reset RNG seed"""
    global rng
    rng = np.random.default_rng(seed)


def clip_to_bounds(x: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """clip solution to bounds"""
    return np.clip(x, bounds[:, 0], bounds[:, 1])


def init_population(mu: int, bounds: np.ndarray) -> np.ndarray:
    """initialize population of mu parents randomly within bounds"""
    dim = bounds.shape[0]
    low = bounds[:, 0]
    high = bounds[:, 1]
    pop = rng.uniform(low=low, high=high, size=(mu, dim))
    return pop


def evaluate_population(pop: np.ndarray, objective) -> np.ndarray:
    """evaluate all individuals in population"""
    return np.array([objective(ind) for ind in pop])


def init_sigmas(bounds: np.ndarray, sigma_global_scale: float = 0.3):
    """initialize global and local step sizes"""
    dim = bounds.shape[0]
    ranges = bounds[:, 1] - bounds[:, 0]
    ranges = np.where(ranges == 0, 1.0, ranges)  # avoid zero range
    
    # for high dimensions, slightly reduce initial step size for better convergence
    if dim >= 50:
        sigma_global_scale = 0.25  # smaller initial step for high dim
    elif dim >= 20:
        sigma_global_scale = 0.28
    
    # global sigma from average range
    sigma_g = sigma_global_scale * np.mean(ranges)
    
    # local sigmas from each dimension's range
    # scale down slightly for high dimensions
    local_scale = 0.25 if dim >= 50 else 0.3
    sigma_local = local_scale * ranges
    
    return sigma_g, sigma_local


def update_local_sigmas_from_parents(parents: np.ndarray,
                                     sigma_local: np.ndarray,
                                     alpha_corr: float = 0.5,
                                     min_sigma: float = 1e-8) -> np.ndarray:
    """update local step sizes using covariance matrix of parents"""
    mu, dim = parents.shape
    
    if mu < 2:
        return sigma_local
    
    # center parents
    centered = parents - np.mean(parents, axis=0, keepdims=True)
    
    # compute covariance matrix
    cov = np.cov(centered.T)
    cov = (cov + cov.T) / 2.0  # ensure symmetry
    
    # get variance per dimension
    var = np.diag(cov)
    var = np.maximum(var, 0.0)
    
    # base sigma from variance
    base_sigma = np.sqrt(var + 1e-12)
    
    # build correlation matrix
    std = np.sqrt(var + 1e-12)
    denom = np.outer(std, std)
    denom = np.where(denom == 0.0, 1e-12, denom)
    corr = cov / denom
    corr = np.clip(corr, -1.0, 1.0)
    
    # average correlation per dimension
    avg_abs_corr = np.mean(np.abs(corr), axis=1)
    
    # scale by correlation
    new_sigma_local = base_sigma * (1.0 + alpha_corr * avg_abs_corr)
    
    # smooth update
    beta = 0.5
    sigma_local = (1.0 - beta) * sigma_local + beta * new_sigma_local
    
    # minimum to avoid freezing
    sigma_local = np.maximum(sigma_local, min_sigma)
    
    return sigma_local


def update_global_sigma(sigma_g: float,
                        success_rate: float,
                        target_success: float = 0.2,
                        a_inc: float = 1.2,
                        b_dec: float = 0.85,
                        min_sigma: float = 1e-8,
                        max_sigma: float = 1e6) -> float:
    """update global step size using 1/5 rule"""
    if success_rate > target_success:
        sigma_g *= a_inc  # increase
    elif success_rate < target_success:
        sigma_g *= b_dec  # decrease
    
    sigma_g = max(min_sigma, min(max_sigma, sigma_g))
    return sigma_g


def hw6_hybrid_cma_es(objective,
                      bounds: np.ndarray,
                      dim: int,
                      n_generations: int = 200,
                      mu: int = 4,
                      lam: int = 10,
                      seed: int | None = None):
    """main HW6 hybrid CMA-ES algorithm"""
    if seed is not None:
        set_seed(seed)
    
    bounds = np.asarray(bounds, dtype=float)
    assert bounds.shape == (dim, 2)
    
    # initialize parents
    parents = init_population(mu, bounds)
    parent_f = evaluate_population(parents, objective)
    
    # initialize step sizes
    sigma_g, sigma_local = init_sigmas(bounds)
    
    # track best
    best_idx = np.argmin(parent_f)
    best_f = float(parent_f[best_idx])
    best_x = parents[best_idx].copy()
    
    history = [best_f]
    
    for gen in range(n_generations):
        # generate offspring
        offspring = np.empty((lam, dim))
        
        for i in range(lam):
            # pick random parent
            p_idx = rng.integers(0, mu)
            parent = parents[p_idx]
            
            # per-dimension step sizes
            step_sizes = sigma_g * sigma_local
            
            # mutate
            step = rng.normal(loc=0.0, scale=step_sizes, size=dim)
            child = parent + step
            child = clip_to_bounds(child, bounds)
            offspring[i] = child
        
        # evaluate offspring
        offspring_f = evaluate_population(offspring, objective)
        
        # compute success rate for 1/5 rule
        parent_baseline = np.median(parent_f)
        successes = np.sum(offspring_f < parent_baseline)
        success_rate = successes / max(1, lam)
        
        # update global sigma
        sigma_g = update_global_sigma(sigma_g, success_rate)
        
        # (μ+λ) selection
        combined = np.vstack([parents, offspring])
        combined_f = np.concatenate([parent_f, offspring_f])
        
        # sort by fitness
        idx = np.argsort(combined_f)
        combined = combined[idx]
        combined_f = combined_f[idx]
        
        # keep best μ
        parents = combined[:mu]
        parent_f = combined_f[:mu]
        
        # update local sigmas
        sigma_local = update_local_sigmas_from_parents(parents, sigma_local)
        
        # update best
        if parent_f[0] < best_f:
            best_f = float(parent_f[0])
            best_x = parents[0].copy()
        
        history.append(best_f)
    
    return best_f, best_x, history


def HW6(f, bounds, dim, n_generations=200, seed=None, **kwargs):
    """competition wrapper with multiple restarts"""
    bounds = np.asarray(bounds, dtype=float)
    
    # adapt parameters based on dimension and search space size
    range_size = np.mean(bounds[:, 1] - bounds[:, 0])
    
    # for high dimensions, use larger populations (but < 20 constraint)
    if dim >= 50:
        # high dimension: need more exploration (all values < 20 as required)
        n_restarts = 3
        mu_val = 8  # < 20
        lam_val = 18  # < 20 (conservative limit)
        gen_per_run = n_generations if n_generations != 200 else 350
    elif dim >= 20:
        # medium-high dimension (all values < 20)
        n_restarts = 3
        mu_val = 7  # < 20
        lam_val = 16  # < 20
        gen_per_run = n_generations if n_generations != 200 else 330
    elif range_size > 100:  # large space
        n_restarts = 3
        gen_per_run = 350
        mu_val = 6
        lam_val = 15
    elif range_size > 10:  # medium
        n_restarts = 2
        gen_per_run = 300
        mu_val = 5
        lam_val = 12
    else:  # small
        n_restarts = 2
        gen_per_run = 250
        mu_val = 4
        lam_val = 10
    
    if n_generations != 200:
        gen_per_run = n_generations
        if n_generations < 150:
            n_restarts = max(1, n_restarts - 1)
    
    best_overall = float('inf')
    
    # multiple restarts
    for restart in range(n_restarts):
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
            
            if best_f < best_overall:
                best_overall = best_f
        
        except Exception as e:
            print(f"Warning: Restart {restart} failed: {e}", flush=True)
            continue
    
    # fallback if all failed
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
    # quick test on sphere
    def sphere(x: np.ndarray) -> float:
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
