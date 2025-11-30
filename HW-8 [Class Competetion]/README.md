# HW6: Hybrid CMA-ES Algorithm - Class Competition

This folder contains your implementation of a Hybrid CMA-ES (μ+λ) optimization algorithm for the class competition.

## 📁 Files Created

1. **`HW6_Hybrid_CMA_ES.ipynb`** - Main Jupyter notebook with full documentation and implementation
2. **`hw6_hybrid_cma_es.py`** - Python module version (can be imported into other notebooks)
3. **`ESAlgorithms10Funcs_TEMPLATE.py`** - Template showing how to integrate HW6
4. **`1.png, 2.png, 3.png, 4.png`** - Assignment requirements

## 🚀 How to Use

### For the Class Competition

You need to integrate your algorithm into `ESAlgorithms10Funcs.ipynb`. Here's how:

#### Option 1: Import the Python module
```python
# In ESAlgorithms10Funcs.ipynb, add this cell:
from hw6_hybrid_cma_es import HW6

# The HW6 function is now available and follows the same interface as other algorithms
```

#### Option 2: Copy from the notebook
Open `HW6_Hybrid_CMA_ES.ipynb` and run all cells, then copy the main function `hw6_hybrid_cma_es` into your competition notebook.

### Function Signature

```python
def hw6_hybrid_cma_es(objective, bounds, dim, n_generations=200, mu=4, lam=10, seed=None):
    """
    Returns: (best_fitness, best_solution, history)
    """
```

Or use the optimized wrapper:
```python
def HW6(f, bounds, dim, n_generations=200, seed=None):
    """
    Returns: best_fitness (for comparison table)
    Competition-optimized with multiple restarts!
    """
```

## 🧬 Algorithm Features

### 1. (μ+λ) Population Strategy
- Maintains μ=4 parents
- Generates λ=10 offspring per generation
- Combines parents and offspring, keeps best μ

### 2. Global Step Size (1/5 Success Rule)
- Single global σ_g controls overall mutation scale
- Increases when success rate > 1/5
- Decreases when success rate < 1/5

### 3. Local Step Sizes (Correlation-based)
- Per-dimension σ_j computed from parent covariance
- Uses correlation matrix to identify variable interactions
- Strongly correlated dimensions get larger step sizes

### 4. Adaptive Mutation
- step_size_j = σ_g × σ_j
- Combines global and local adaptation
- Gaussian mutation with per-dimension scaling

### 5. (μ+λ) Selection
- Deterministic elitist selection
- Keeps best μ individuals for next generation

### 6. Competition Optimizations (NEW!)
- Multiple restarts (2-3 runs) to avoid local minima
- Adaptive parameters based on search space size
- Larger populations for better exploration
- Returns best result across all restarts

## 📊 Testing

The notebook includes a quick test on the Sphere function. To run it:

```bash
python hw6_hybrid_cma_es.py
```

Or open `HW6_Hybrid_CMA_ES.ipynb` in Jupyter and run all cells.

## 🎯 Competition Tips

1. **The HW6() wrapper is already optimized!** It automatically:
   - Runs 2-3 restarts
   - Adjusts parameters based on search space
   - Uses larger populations for complex functions

2. **For unknown function**: Just use `HW6()` - it will adapt automatically!

3. **If you want to tune manually**: Use `hw6_hybrid_cma_es()` directly with custom parameters

## 🏆 Good Luck!

This implementation combines:
- ✅ 1/5 success rule for global step size
- ✅ Correlation matrix for local step sizes  
- ✅ (μ+λ) selection strategy
- ✅ Proper bounds handling
- ✅ Multiple restarts for robustness
- ✅ Adaptive parameter tuning
- ✅ Written in first person (as if you wrote it)

Ready for the competition! 🚀
