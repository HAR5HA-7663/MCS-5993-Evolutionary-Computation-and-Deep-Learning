# ESAlgorithms10Funcs - Template for HW6 Integration
# Course: MCS-5993 Evolutionary Computation and Deep Learning
# Student: Harsha Yellela

import numpy as np
from hw6_hybrid_cma_es import HW6 as hw6_optimized

# Benchmark Functions (10 functions from competition table)

def sphere(x):
    """Sphere function - unimodal"""
    return np.sum(x**2)

def dixon_price(x):
    """Dixon-Price function - Competition function"""
    x = np.array(x)
    n = len(x)
    
    # first term (x1 - 1)^2
    term1 = (x[0] - 1)**2
    
    # loop part, vectorized
    # Formula: sum( i * (2*x_i^2 - x_{i-1})^2 ) for i from 2 to n
    indices = np.arange(2, n + 1)
    term2 = np.sum(indices * (2 * x[1:]**2 - x[:-1])**2)
    
    return term1 + term2

def rosenbrock(x):
    """Rosenbrock function - unimodal valley"""
    return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)


def rastrigin(x):
    """Rastrigin function - highly multimodal"""
    A = 10
    n = len(x)
    return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))


def ackley(x):
    """Ackley function - multimodal"""
    n = len(x)
    sum_sq = np.sum(x**2)
    sum_cos = np.sum(np.cos(2 * np.pi * x))
    return -20 * np.exp(-0.2 * np.sqrt(sum_sq / n)) - np.exp(sum_cos / n) + 20 + np.e


def griewank(x):
    """Griewank function - multimodal"""
    sum_sq = np.sum(x**2)
    prod_cos = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    return 1 + sum_sq / 4000 - prod_cos


def schwefel(x):
    """Schwefel function - multimodal"""
    n = len(x)
    return 418.9829 * n - np.sum(x * np.sin(np.sqrt(np.abs(x))))


def levy(x):
    """Levy function - multimodal"""
    w = 1 + (x - 1) / 4
    term1 = np.sin(np.pi * w[0])**2
    term2 = np.sum((w[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1)**2))
    term3 = (w[-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[-1])**2)
    return term1 + term2 + term3


def zakharov(x):
    """Zakharov function - unimodal"""
    sum1 = np.sum(x**2)
    sum2 = np.sum(0.5 * np.arange(1, len(x) + 1) * x)
    return sum1 + sum2**2 + sum2**4


def lunacek_bi_rastrigin(x):
    """Lunacek Bi-Rastrigin - simplified version"""
    return rastrigin(x) + np.sum((x - 2.5)**2)


def hybrid_composition(x):
    """Hybrid Composition - combination of multiple functions"""
    return 0.3 * sphere(x) + 0.3 * rastrigin(x) + 0.4 * griewank(x)

# Test Functions Dictionary with Bounds

test_functions = {
    'Dixon_Price': {
        'function': dixon_price,
        'bounds': np.array([[-10, 10], [-10, 10]]),
        'dim': 2,
        'global_min': 1e-5
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

# Baseline Algorithms

def random_search(f, bounds, dim, n_generations=200, seed=None):
    """Random Search baseline"""
    if seed is not None:
        np.random.seed(seed)
    
    best_f = float('inf')
    best_x = None
    
    n_evals = n_generations * 10
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
    
    # initialize
    x = np.random.uniform(bounds[:, 0], bounds[:, 1])
    sigma = 0.3 * np.mean(bounds[:, 1] - bounds[:, 0])
    
    best_f = f(x)
    best_x = x.copy()
    
    lam = 10
    for gen in range(n_generations):
        # generate offspring
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
        
        # adapt sigma using 1/5 rule
        success_rate = successes / lam
        if success_rate > 0.2:
            sigma *= 1.2  # increase
        elif success_rate < 0.2:
            sigma *= 0.85  # decrease
    
    return best_f


def mu_plus_lambda_es(f, bounds, dim, n_generations=200, seed=None):
    """(μ+λ) Evolution Strategy"""
    if seed is not None:
        np.random.seed(seed)
    
    mu = 4
    lam = 10
    
    # initialize population
    population = [np.random.uniform(bounds[:, 0], bounds[:, 1]) for _ in range(mu)]
    fitness = [f(ind) for ind in population]
    
    best_f = min(fitness)
    best_x = population[fitness.index(best_f)].copy()
    
    for gen in range(n_generations):
        # generate offspring
        offspring = []
        for _ in range(lam):
            parent = population[np.random.randint(mu)]
            sigma = 0.3 * np.mean(bounds[:, 1] - bounds[:, 0])
            child = parent + sigma * np.random.randn(dim)
            child = np.clip(child, bounds[:, 0], bounds[:, 1])
            offspring.append(child)
        
        # evaluate offspring
        offspring_fitness = [f(ind) for ind in offspring]
        
        # select best μ from parents + offspring
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
    """(μ+λ) ES variant"""
    return mu_plus_lambda_es(f, bounds, dim, n_generations, seed)

# HW6 Algorithm Integration

def HW6(f, bounds, dim, n_generations=200, seed=None):
    """HW6: Hybrid CMA-ES (μ+λ) Algorithm"""
    return hw6_optimized(f, bounds, dim, n_generations, seed)

# Algorithm Dictionary

algorithms = {
    'Random Search': random_search,
    '1/5 Rule ES': one_fifth_rule_es,
    '(μ+λ)-ES': mu_plus_lambda_es,
    '(μ+λ)-ES Vari': mu_plus_lambda_es_variant,
    'HW6': HW6,
}

# Comparison Framework

def compare_algorithms(algorithms, test_functions, n_runs=10, n_generations=200):
    """Compare all algorithms on all test functions"""
    results = {}
    
    for func_name, func_info in test_functions.items():
        print(f"\nTesting on {func_name}...")
        results[func_name] = {}
        
        for alg_name, alg_func in algorithms.items():
            print(f"  Running {alg_name}...", end=' ')
            
            # run multiple times for average
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


def print_comparison_table(results, dim=2):
    """Print comparison results in formatted table"""
    print("\n" + "="*120)
    print(f"{f'D={dim}':<20} ALGORITHM COMPARISON SUMMARY")
    print("="*120)
    
    # header
    header = f"{'Function':<20}"
    for alg_name in algorithms.keys():
        header += f"{alg_name:<20}"
    header += f"{'Winner':<20}"
    print(header)
    print("-"*120)
    
    # count wins
    wins = {alg: 0 for alg in algorithms.keys()}
    
    # print results for each function
    for func_name, func_results in results.items():
        row = f"{func_name:<20}"
        
        # find best result
        best_result = min(func_results.values())
        winner = [alg for alg, res in func_results.items() if res == best_result][0]
        wins[winner] += 1
        
        # print all algorithm results
        for alg_name in algorithms.keys():
            result = func_results[alg_name]
            row += f"{result:<20.6e}"
        
        row += f"{winner:<20}"
        print(row)
    
    # print summary
    print("-"*120)
    row = f"{'TOTAL WINS':<20}"
    for alg_name in algorithms.keys():
        row += f"{wins[alg_name]:<20}"
    print(row)
    
    # overall winner
    overall_winner = max(wins.items(), key=lambda x: x[1])
    print(f"\n{'OVERALL WINNER':<20}{overall_winner[0]:<20}")
    print("="*120)

# Main Execution

def main(dim=50, max_evaluations=5000, runs=5):
    """competition main function"""
    # convert max_evaluations to n_generations
    # using average mu=5, lam=12 for estimation
    avg_pop_size = 17  # mu + lambda average
    n_generations = max(1, max_evaluations // avg_pop_size)
    
    # create all 10 functions with specified dimension bounds
    competition_functions = {
        'Dixon_Price': {
            'function': dixon_price,
            'bounds': np.array([[-10, 10]] * dim),
            'dim': dim,
            'global_min': 1e-5
        },
        'Rosenbrock': {
            'function': rosenbrock,
            'bounds': np.array([[-2.048, 2.048]] * dim),
            'dim': dim,
            'global_min': 0.0
        },
        'Rastrigin': {
            'function': rastrigin,
            'bounds': np.array([[-5.12, 5.12]] * dim),
            'dim': dim,
            'global_min': 0.0
        },
        'Ackley': {
            'function': ackley,
            'bounds': np.array([[-32.768, 32.768]] * dim),
            'dim': dim,
            'global_min': 0.0
        },
        'Griewank': {
            'function': griewank,
            'bounds': np.array([[-600, 600]] * dim),
            'dim': dim,
            'global_min': 0.0
        },
        'Schwefel': {
            'function': schwefel,
            'bounds': np.array([[-500, 500]] * dim),
            'dim': dim,
            'global_min': 0.0
        },
        'Lunacek BiRstrgn': {
            'function': lunacek_bi_rastrigin,
            'bounds': np.array([[-5.12, 5.12]] * dim),
            'dim': dim,
            'global_min': 0.0
        },
        'Levy': {
            'function': levy,
            'bounds': np.array([[-10, 10]] * dim),
            'dim': dim,
            'global_min': 0.0
        },
        'Zakharov': {
            'function': zakharov,
            'bounds': np.array([[-5, 10]] * dim),
            'dim': dim,
            'global_min': 0.0
        },
        'Hybrid Composition': {
            'function': hybrid_composition,
            'bounds': np.array([[-5, 5]] * dim),
            'dim': dim,
            'global_min': 0.0
        }
    }
    
    print(f"Running Competition: dim={dim}, max_evaluations={max_evaluations}, runs={runs}")
    print(f"Using {n_generations} generations (estimated from {max_evaluations} max evaluations)")
    
    # run comparison
    results = compare_algorithms(
        algorithms=algorithms,
        test_functions=competition_functions,
        n_runs=runs,
        n_generations=n_generations
    )
    
    # print formatted table with dimension
    print_comparison_table(results, dim=dim)
    
    print("\n✅ Competition comparison complete!")
    return results

if __name__ == "__main__":
    # use competition parameters
    main(dim=50, max_evaluations=5000, runs=5)
