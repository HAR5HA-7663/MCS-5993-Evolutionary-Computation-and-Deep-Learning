# ==============================================================================
# ESAlgorithms10Funcs - Template for HW6 Integration
# Course: MCS-5993 Evolutionary Computation and Deep Learning
# Student: Harsha Yellela
# Purpose: Template showing how to integrate HW6 algorithm for competition
# ==============================================================================

import numpy as np
from hw6_hybrid_cma_es import HW6 as hw6_optimized

# ==============================================================================
# Benchmark Functions (10 functions from the competition table)
# ==============================================================================
# I implement these standard benchmark functions to test optimization algorithms.
# Each function has different characteristics: unimodal vs multimodal, different
# landscapes, and varying difficulty levels.

def sphere(x):
    """
    Sphere function - Unimodal test function.
    
    This function has a single global minimum at the origin and is useful for
    testing basic convergence behavior.
    
    Parameters
    ----------
    x : np.ndarray
        Input vector of any dimension
    
    Returns
    -------
    float
        Sum of squares: f(x) = Σ x_i²
    """
    return np.sum(x**2)


def rosenbrock(x):
    """
    Rosenbrock function - Unimodal valley function.
    
    This function has a narrow valley leading to the global minimum, making it
    challenging for algorithms that don't adapt well to curved landscapes.
    
    Parameters
    ----------
    x : np.ndarray
        Input vector of dimension at least 2
    
    Returns
    -------
    float
        Rosenbrock function value
    """
    return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)


def rastrigin(x):
    """
    Rastrigin function - Highly multimodal test function.
    
    This function has many local minima, making it challenging for algorithms
    to escape local optima and find the global minimum.
    
    Parameters
    ----------
    x : np.ndarray
        Input vector of any dimension
    
    Returns
    -------
    float
        Rastrigin function value
    """
    A = 10
    n = len(x)
    return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))


def ackley(x):
    """
    Ackley function - Multimodal test function.
    
    This function combines exponential and cosine terms to create a complex
    landscape with many local minima and a single global minimum.
    
    Parameters
    ----------
    x : np.ndarray
        Input vector of any dimension
    
    Returns
    -------
    float
        Ackley function value
    """
    n = len(x)
    sum_sq = np.sum(x**2)
    sum_cos = np.sum(np.cos(2 * np.pi * x))
    return -20 * np.exp(-0.2 * np.sqrt(sum_sq / n)) - np.exp(sum_cos / n) + 20 + np.e


def griewank(x):
    """
    Griewank function - Multimodal test function.
    
    This function has many local minima, but they become less pronounced
    as we move away from the origin, making the global minimum easier to find.
    
    Parameters
    ----------
    x : np.ndarray
        Input vector of any dimension
    
    Returns
    -------
    float
        Griewank function value
    """
    sum_sq = np.sum(x**2)
    prod_cos = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    return 1 + sum_sq / 4000 - prod_cos


def schwefel(x):
    """
    Schwefel function - Multimodal test function.
    
    This function has deceptive behavior where local minima are far from the
    global minimum, making it challenging for gradient-based methods.
    
    Parameters
    ----------
    x : np.ndarray
        Input vector of any dimension
    
    Returns
    -------
    float
        Schwefel function value
    """
    n = len(x)
    return 418.9829 * n - np.sum(x * np.sin(np.sqrt(np.abs(x))))


def levy(x):
    """
    Levy function - Multimodal test function.
    
    This function has many local minima and requires good exploration to find
    the global minimum at (1, 1, ..., 1).
    
    Parameters
    ----------
    x : np.ndarray
        Input vector of any dimension
    
    Returns
    -------
    float
        Levy function value
    """
    w = 1 + (x - 1) / 4
    term1 = np.sin(np.pi * w[0])**2
    term2 = np.sum((w[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1)**2))
    term3 = (w[-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[-1])**2)
    return term1 + term2 + term3


def zakharov(x):
    """
    Zakharov function - Unimodal test function.
    
    This function combines polynomial terms and is used to test algorithm
    performance on smooth, unimodal landscapes.
    
    Parameters
    ----------
    x : np.ndarray
        Input vector of any dimension
    
    Returns
    -------
    float
        Zakharov function value
    """
    sum1 = np.sum(x**2)
    sum2 = np.sum(0.5 * np.arange(1, len(x) + 1) * x)
    return sum1 + sum2**2 + sum2**4


def lunacek_bi_rastrigin(x):
    """
    Lunacek Bi-Rastrigin - Complex multimodal function.
    
    This is a simplified version that combines Rastrigin with a quadratic shift,
    creating a challenging multimodal landscape.
    
    Parameters
    ----------
    x : np.ndarray
        Input vector of any dimension
    
    Returns
    -------
    float
        Lunacek Bi-Rastrigin function value
    """
    return rastrigin(x) + np.sum((x - 2.5)**2)


def hybrid_composition(x):
    """
    Hybrid Composition - Very complex test function.
    
    This function combines multiple benchmark functions to create a complex
    landscape that tests algorithm robustness across different function types.
    
    Parameters
    ----------
    x : np.ndarray
        Input vector of any dimension
    
    Returns
    -------
    float
        Hybrid composition function value
    """
    return 0.3 * sphere(x) + 0.3 * rastrigin(x) + 0.4 * griewank(x)

# ==============================================================================
# Test Functions Dictionary with Bounds
# ==============================================================================
# I organize the test functions into a dictionary with their bounds and properties
# to make it easy to iterate over them during comparison experiments.

test_functions = {
    'Sphere': {
        'function': sphere,
        'bounds': np.array([[-5.12, 5.12], [-5.12, 5.12]]),
        'dim': 2,
        'global_min': 0.0
    },
    'Rosenbrock': {
        'function': rosenbrock,
        'bounds': np.array([[-2.048, 2.048], [-2.048, 2.048]]),
        'dim': 2,
        'global_min': 0.0
    },
    'Rastrigin': {
        'function': rastrigin,
        'bounds': np.array([[-5.12, 5.12], [-5.12, 5.12]]),
        'dim': 2,
        'global_min': 0.0
    },
    'Ackley': {
        'function': ackley,
        'bounds': np.array([[-32.768, 32.768], [-32.768, 32.768]]),
        'dim': 2,
        'global_min': 0.0
    },
    'Griewank': {
        'function': griewank,
        'bounds': np.array([[-600, 600], [-600, 600]]),
        'dim': 2,
        'global_min': 0.0
    },
    'Schwefel': {
        'function': schwefel,
        'bounds': np.array([[-500, 500], [-500, 500]]),
        'dim': 2,
        'global_min': 0.0
    },
    'Lunacek BiRstrgn': {
        'function': lunacek_bi_rastrigin,
        'bounds': np.array([[-5.12, 5.12], [-5.12, 5.12]]),
        'dim': 2,
        'global_min': 0.0
    },
    'Levy': {
        'function': levy,
        'bounds': np.array([[-10, 10], [-10, 10]]),
        'dim': 2,
        'global_min': 0.0
    },
    'Zakharov': {
        'function': zakharov,
        'bounds': np.array([[-5, 10], [-5, 10]]),
        'dim': 2,
        'global_min': 0.0
    },
    'Hybrid Composition': {
        'function': hybrid_composition,
        'bounds': np.array([[-5, 5], [-5, 5]]),
        'dim': 2,
        'global_min': 0.0
    }
}

# ==============================================================================
# Baseline Algorithm Implementations
# ==============================================================================
# I implement baseline algorithms for comparison to evaluate how my HW6 algorithm
# performs relative to standard approaches. These serve as benchmarks.

def random_search(f, bounds, dim, n_generations=200, seed=None):
    """
    Random Search baseline algorithm.
    
    This is the simplest baseline - it just samples random points and keeps
    the best one found. I use this to establish a lower bound on performance.
    
    Parameters
    ----------
    f : callable
        Objective function to minimize
    bounds : np.ndarray
        Array of shape (dim, 2) with [lower, upper] bounds
    dim : int
        Dimension of search space
    n_generations : int
        Number of generations (converted to evaluations)
    seed : int or None
        Random seed for reproducibility
    
    Returns
    -------
    float
        Best fitness found
    """
    if seed is not None:
        np.random.seed(seed)
    
    best_f = float('inf')
    best_x = None
    
    # Use similar evaluation budget to other algorithms for fair comparison
    n_evals = n_generations * 10
    for _ in range(n_evals):
        x = np.random.uniform(bounds[:, 0], bounds[:, 1])
        fx = f(x)
        if fx < best_f:
            best_f = fx
            best_x = x
    
    return best_f


def one_fifth_rule_es(f, bounds, dim, n_generations=200, seed=None):
    """
    1/5 Rule Evolution Strategy baseline.
    
    This algorithm uses a single parent with adaptive step size controlled by
    the 1/5 success rule. I implement this to compare against my population-based
    approach with dual step sizes.
    
    Parameters
    ----------
    f : callable
        Objective function to minimize
    bounds : np.ndarray
        Array of shape (dim, 2) with [lower, upper] bounds
    dim : int
        Dimension of search space
    n_generations : int
        Number of generations
    seed : int or None
        Random seed for reproducibility
    
    Returns
    -------
    float
        Best fitness found
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Initialize single parent
    x = np.random.uniform(bounds[:, 0], bounds[:, 1])
    sigma = 0.3 * np.mean(bounds[:, 1] - bounds[:, 0])
    
    best_f = f(x)
    best_x = x.copy()
    
    lam = 10
    for gen in range(n_generations):
        # Generate offspring
        successes = 0
        for _ in range(lam):
            x_new = x + sigma * np.random.randn(dim)
            x_new = np.clip(x_new, bounds[:, 0], bounds[:, 1])
            f_new = f(x_new)
            
            # Track successes for 1/5 rule
            if f_new < f(x):
                x = x_new
                successes += 1
                if f_new < best_f:
                    best_f = f_new
                    best_x = x_new.copy()
        
        # Adapt sigma using 1/5 rule: increase if success rate > 0.2, decrease otherwise
        success_rate = successes / lam
        if success_rate > 0.2:
            sigma *= 1.2  # Increase exploration
        elif success_rate < 0.2:
            sigma *= 0.85  # Increase exploitation
    
    return best_f


def mu_plus_lambda_es(f, bounds, dim, n_generations=200, seed=None):
    """
    (μ+λ) Evolution Strategy baseline.
    
    This algorithm maintains a population of μ parents and generates λ offspring
    each generation. It uses fixed step sizes, unlike my adaptive approach.
    I use this to compare against my (μ+λ) strategy with dual step sizes.
    
    Parameters
    ----------
    f : callable
        Objective function to minimize
    bounds : np.ndarray
        Array of shape (dim, 2) with [lower, upper] bounds
    dim : int
        Dimension of search space
    n_generations : int
        Number of generations
    seed : int or None
        Random seed for reproducibility
    
    Returns
    -------
    float
        Best fitness found
    """
    if seed is not None:
        np.random.seed(seed)
    
    mu = 4
    lam = 10
    
    # Initialize population
    population = [np.random.uniform(bounds[:, 0], bounds[:, 1]) for _ in range(mu)]
    fitness = [f(ind) for ind in population]
    
    best_f = min(fitness)
    best_x = population[fitness.index(best_f)].copy()
    
    for gen in range(n_generations):
        # Generate offspring from random parents
        offspring = []
        for _ in range(lam):
            parent = population[np.random.randint(mu)]
            # Use fixed sigma (not adaptive like my algorithm)
            sigma = 0.3 * np.mean(bounds[:, 1] - bounds[:, 0])
            child = parent + sigma * np.random.randn(dim)
            child = np.clip(child, bounds[:, 0], bounds[:, 1])
            offspring.append(child)
        
        # Evaluate offspring
        offspring_fitness = [f(ind) for ind in offspring]
        
        # Select best μ from parents + offspring (elitist selection)
        combined = population + offspring
        combined_fitness = fitness + offspring_fitness
        
        indices = np.argsort(combined_fitness)[:mu]
        population = [combined[i] for i in indices]
        fitness = [combined_fitness[i] for i in indices]
        
        if fitness[0] < best_f:
            best_f = fitness[0]
            best_x = population[0].copy()
    
    return best_f


def mu_plus_lambda_es_variant(f, bounds, dim, n_generations=200, seed=None):
    """
    (μ+λ) ES variant baseline.
    
    This is similar to the standard (μ+λ)-ES but included as a variant for
    comparison. Currently it's identical to the standard version.
    
    Parameters
    ----------
    f : callable
        Objective function to minimize
    bounds : np.ndarray
        Array of shape (dim, 2) with [lower, upper] bounds
    dim : int
        Dimension of search space
    n_generations : int
        Number of generations
    seed : int or None
        Random seed for reproducibility
    
    Returns
    -------
    float
        Best fitness found
    """
    return mu_plus_lambda_es(f, bounds, dim, n_generations, seed)

# ==============================================================================
# HW6 Algorithm Integration
# ==============================================================================
# I import my HW6 algorithm implementation which uses dual step sizes (global
# and local) with correlation learning. This is the main algorithm I'm comparing.

def HW6(f, bounds, dim, n_generations=200, seed=None):
    """
    HW6: Hybrid CMA-ES (μ+λ) Algorithm.
    
    This is my competition algorithm that combines:
    - (μ+λ) Evolution Strategy population structure
    - Global step size σ_g adapted via 1/5 success rule
    - Local step sizes σ_local (one per dimension) learned from covariance matrix
    - Adaptive mutation combining global and local information
    
    For competition, I use multiple restarts (2-3 runs) with adaptive parameters
    based on search space size to maximize robustness and avoid local minima.
    
    Parameters
    ----------
    f : callable
        Objective function to minimize
    bounds : np.ndarray
        Array of shape (dim, 2) with [lower, upper] bounds
    dim : int
        Dimension of search space
    n_generations : int
        Number of generations
    seed : int or None
        Random seed for reproducibility
    
    Returns
    -------
    float
        Best fitness found across all restarts
    """
    return hw6_optimized(f, bounds, dim, n_generations, seed)

# ==============================================================================
# Algorithm Dictionary
# ==============================================================================
# I organize all algorithms into a dictionary for easy iteration during comparison.

algorithms = {
    'Random Search': random_search,
    '1/5 Rule ES': one_fifth_rule_es,
    '(μ+λ)-ES': mu_plus_lambda_es,
    '(μ+λ)-ES Vari': mu_plus_lambda_es_variant,
    'HW6': HW6,
}

# ==============================================================================
# Comparison Framework
# ==============================================================================
# I implement functions to compare all algorithms on all test functions and
# display results in a formatted table matching the assignment requirements.

def compare_algorithms(algorithms, test_functions, n_runs=10, n_generations=200):
    """
    Compare all algorithms on all test functions.
    
    I run each algorithm multiple times (n_runs) on each function and compute
    the mean performance. This provides statistical reliability for comparison.
    
    Parameters
    ----------
    algorithms : dict
        Dictionary mapping algorithm names to functions
    test_functions : dict
        Dictionary of test functions with bounds and properties
    n_runs : int
        Number of independent runs per algorithm per function
    n_generations : int
        Number of generations per run
    
    Returns
    -------
    dict
        Nested dictionary: results[function_name][algorithm_name] = mean_fitness
    """
    results = {}
    
    for func_name, func_info in test_functions.items():
        print(f"\nTesting on {func_name}...")
        results[func_name] = {}
        
        for alg_name, alg_func in algorithms.items():
            print(f"  Running {alg_name}...", end=' ')
            
            # Run multiple times for statistical reliability
            runs = []
            for run in range(n_runs):
                try:
                    best_f = alg_func(
                        f=func_info['function'],
                        bounds=func_info['bounds'],
                        dim=func_info['dim'],
                        n_generations=n_generations,
                        seed=run
                    )
                    runs.append(best_f)
                except Exception as e:
                    print(f"Error: {e}")
                    runs.append(float('inf'))
            
            results[func_name][alg_name] = np.mean(runs)
            print(f"Mean: {results[func_name][alg_name]:.6e}")
    
    return results


def print_comparison_table(results):
    """
    Print comparison results in formatted table.
    
    I format the results to match the assignment table format, showing all
    algorithm performances and identifying winners for each function.
    
    Parameters
    ----------
    results : dict
        Results dictionary from compare_algorithms
    """
    print("\n" + "="*120)
    print(f"{'D=2':<20} ALGORITHM COMPARISON SUMMARY")
    print("="*120)
    
    # Header row with all algorithm names
    header = f"{'Function':<20}"
    for alg_name in algorithms.keys():
        header += f"{alg_name:<20}"
    header += f"{'Winner':<20}"
    print(header)
    print("-"*120)
    
    # Count wins for each algorithm
    wins = {alg: 0 for alg in algorithms.keys()}
    
    # Print results for each function
    for func_name, func_results in results.items():
        row = f"{func_name:<20}"
        
        # Find best result (lowest fitness = winner)
        best_result = min(func_results.values())
        winner = [alg for alg, res in func_results.items() if res == best_result][0]
        wins[winner] += 1
        
        # Add all algorithm results to row
        for alg_name in algorithms.keys():
            result = func_results[alg_name]
            row += f"{result:<20.6e}"
        
        row += f"{winner:<20}"
        print(row)
    
    # Print summary with total wins
    print("-"*120)
    row = f"{'TOTAL WINS':<20}"
    for alg_name in algorithms.keys():
        row += f"{wins[alg_name]:<20}"
    print(row)
    
    # Print overall winner
    overall_winner = max(wins.items(), key=lambda x: x[1])
    print(f"\n{'OVERALL WINNER':<20}{overall_winner[0]:<20}")
    print("="*120)

# ==============================================================================
# Main Execution
# ==============================================================================
# I run the full comparison when this script is executed directly.

if __name__ == "__main__":
    print("Running Algorithm Comparison for HW6...")
    print(f"Testing {len(test_functions)} functions with {len(algorithms)} algorithms")
    
    # Run full comparison
    results = compare_algorithms(
        algorithms=algorithms,
        test_functions=test_functions,
        n_runs=10,  # Number of runs per algorithm per function
        n_generations=200  # Generations per run
    )
    
    # Print formatted table
    print_comparison_table(results)
    
    print("\n✅ Comparison complete!")
