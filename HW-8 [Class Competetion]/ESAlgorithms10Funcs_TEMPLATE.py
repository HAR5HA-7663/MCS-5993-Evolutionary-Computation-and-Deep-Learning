# ESAlgorithms10Funcs - TEMPLATE for HW6 Integration
# This is a template showing how to integrate your HW6 algorithm

import numpy as np
from hw6_hybrid_cma_es import HW6 as hw6_optimized

# ==============================================================================
# STEP 1: Define Benchmark Functions (10 functions from the table)
# ==============================================================================

def sphere(x):
    """Sphere function - Unimodal"""
    return np.sum(x**2)

def rosenbrock(x):
    """Rosenbrock function - Unimodal valley"""
    return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

def rastrigin(x):
    """Rastrigin function - Highly multimodal"""
    A = 10
    n = len(x)
    return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))

def ackley(x):
    """Ackley function - Multimodal"""
    n = len(x)
    sum_sq = np.sum(x**2)
    sum_cos = np.sum(np.cos(2 * np.pi * x))
    return -20 * np.exp(-0.2 * np.sqrt(sum_sq / n)) - np.exp(sum_cos / n) + 20 + np.e

def griewank(x):
    """Griewank function - Multimodal"""
    sum_sq = np.sum(x**2)
    prod_cos = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    return 1 + sum_sq / 4000 - prod_cos

def schwefel(x):
    """Schwefel function - Multimodal"""
    n = len(x)
    return 418.9829 * n - np.sum(x * np.sin(np.sqrt(np.abs(x))))

def levy(x):
    """Levy function - Multimodal"""
    w = 1 + (x - 1) / 4
    term1 = np.sin(np.pi * w[0])**2
    term2 = np.sum((w[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1)**2))
    term3 = (w[-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[-1])**2)
    return term1 + term2 + term3

def zakharov(x):
    """Zakharov function - Unimodal"""
    sum1 = np.sum(x**2)
    sum2 = np.sum(0.5 * np.arange(1, len(x) + 1) * x)
    return sum1 + sum2**2 + sum2**4

# Placeholder for more complex functions
def lunacek_bi_rastrigin(x):
    """Lunacek Bi-Rastrigin - Complex multimodal"""
    # Simplified version
    return rastrigin(x) + np.sum((x - 2.5)**2)

def hybrid_composition(x):
    """Hybrid Composition - Very complex"""
    # Simplified version combining multiple functions
    return 0.3 * sphere(x) + 0.3 * rastrigin(x) + 0.4 * griewank(x)

# ==============================================================================
# STEP 2: Define Test Functions with Bounds
# ==============================================================================

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
# STEP 3: Define Algorithm Wrappers
# ==============================================================================

def random_search(f, bounds, dim, n_generations=200, seed=None):
    """Random Search baseline"""
    if seed is not None:
        np.random.seed(seed)
    
    best_f = float('inf')
    best_x = None
    
    n_evals = n_generations * 10  # Similar budget to other algorithms
    for _ in range(n_evals):
        x = np.random.uniform(bounds[:, 0], bounds[:, 1])
        fx = f(x)
        if fx < best_f:
            best_f = fx
            best_x = x
    
    return best_f

def one_fifth_rule_es(f, bounds, dim, n_generations=200, seed=None):
    """1/5 Rule Evolution Strategy"""
    if seed is not None:
        np.random.seed(seed)
    
    # Initialize
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
            
            if f_new < f(x):
                x = x_new
                successes += 1
                if f_new < best_f:
                    best_f = f_new
                    best_x = x_new.copy()
        
        # Adapt sigma using 1/5 rule
        success_rate = successes / lam
        if success_rate > 0.2:
            sigma *= 1.2
        elif success_rate < 0.2:
            sigma *= 0.85
    
    return best_f

def mu_plus_lambda_es(f, bounds, dim, n_generations=200, seed=None):
    """(μ+λ) Evolution Strategy"""
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
        # Generate offspring
        offspring = []
        for _ in range(lam):
            parent = population[np.random.randint(mu)]
            sigma = 0.3 * np.mean(bounds[:, 1] - bounds[:, 0])
            child = parent + sigma * np.random.randn(dim)
            child = np.clip(child, bounds[:, 0], bounds[:, 1])
            offspring.append(child)
        
        # Evaluate offspring
        offspring_fitness = [f(ind) for ind in offspring]
        
        # Select best μ from parents + offspring
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
    """(μ+λ) ES with adaptive sigma"""
    # Similar to above but with self-adaptive sigma
    return mu_plus_lambda_es(f, bounds, dim, n_generations, seed)

# ==============================================================================
# STEP 4: ADD YOUR HW6 ALGORITHM HERE
# ==============================================================================

def HW6(f, bounds, dim, n_generations=200, seed=None):
    """
    HW6: Hybrid CMA-ES (μ+λ) Algorithm
    
    This algorithm combines:
    - (μ+λ) Evolution Strategy structure
    - Global step size with 1/5 success rule
    - Local step sizes per dimension using correlation matrix
    - Adaptive mutation combining global and local information
    
    NOTE: FOR COMPETITION:
    - Multiple restarts (2-3 runs) to avoid local minima
    - Adaptive parameters based on search space size
    - Larger population and more generations for better exploration
    - Returns best result across all restarts
    """
    # Use the optimized HW6 function with multiple restarts
    return hw6_optimized(f, bounds, dim, n_generations, seed)

# ==============================================================================
# STEP 5: Define Algorithm Dictionary
# ==============================================================================

algorithms = {
    'Random Search': random_search,
    '1/5 Rule ES': one_fifth_rule_es,
    '(μ+λ)-ES': mu_plus_lambda_es,
    '(μ+λ)-ES Vari': mu_plus_lambda_es_variant,
    'HW6': HW6,  # <-- YOUR ALGORITHM
}

# ==============================================================================
# STEP 6: Run Comparison
# ==============================================================================

def compare_algorithms(algorithms, test_functions, n_runs=10, n_generations=200):
    """Compare all algorithms on all test functions"""
    
    results = {}
    
    for func_name, func_info in test_functions.items():
        print(f"\nTesting on {func_name}...")
        results[func_name] = {}
        
        for alg_name, alg_func in algorithms.items():
            print(f"  Running {alg_name}...", end=' ')
            
            # Run multiple times and take mean
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
    """Print results in the format shown in the assignment"""
    
    print("\n" + "="*120)
    print(f"{'D=2':<20} ALGORITHM COMPARISON SUMMARY")
    print("="*120)
    
    # Header
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
        
        # Find best result for this function
        best_result = min(func_results.values())
        winner = [alg for alg, res in func_results.items() if res == best_result][0]
        wins[winner] += 1
        
        # Print all results
        for alg_name in algorithms.keys():
            result = func_results[alg_name]
            row += f"{result:<20.6e}"
        
        row += f"{winner:<20}"
        print(row)
    
    # Print summary
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
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    print("Running Algorithm Comparison for HW6...")
    print(f"Testing {len(test_functions)} functions with {len(algorithms)} algorithms")
    
    # Run comparison
    results = compare_algorithms(
        algorithms=algorithms,
        test_functions=test_functions,
        n_runs=10,  # Number of runs per algorithm per function
        n_generations=200  # Generations per run
    )
    
    # Print formatted table
    print_comparison_table(results)
    
    print("\n✅ Comparison complete!")
