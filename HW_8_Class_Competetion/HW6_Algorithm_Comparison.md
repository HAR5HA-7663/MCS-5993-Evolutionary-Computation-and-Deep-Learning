# HW6 Algorithm vs Baseline Algorithms - Detailed Comparison

## Overview

Your HW6 algorithm is a **Hybrid CMA-ES** that combines multiple advanced techniques not found in the baseline algorithms. Here's a detailed breakdown of the key differences:

---

## 🔑 Key Innovation: Dual-Step-Size System

### HW6 Algorithm:
- ✅ **Global Step Size (σ_g)**: One overall step size for all dimensions
- ✅ **Local Step Sizes (σ_local)**: Separate step size for EACH dimension
- ✅ **Combined Mutation**: `step_sizes = sigma_g * sigma_local` (element-wise multiplication)

This allows your algorithm to:
- Adapt to overall convergence (global sigma)
- Adapt to each dimension's sensitivity (local sigmas)
- Handle correlated dimensions better

### Baseline Algorithms:
- ❌ **1/5 Rule ES**: Only ONE global sigma (no per-dimension adaptation)
- ❌ **(μ+λ)-ES**: Only ONE fixed global sigma (doesn't even adapt!)
- ❌ **Random Search**: No step sizes at all

---

## 📊 Detailed Feature Comparison

| Feature | HW6 | 1/5 Rule ES | (μ+λ)-ES | Random Search |
|---------|-----|-------------|----------|---------------|
| **Population Strategy** | (μ+λ) - keeps μ parents | Single individual | (μ+λ) - keeps μ parents | No population |
| **Global Step Size** | ✅ Adaptive (1/5 rule) | ✅ Adaptive (1/5 rule) | ❌ **Fixed** (0.3 × range) | N/A |
| **Local Step Sizes** | ✅ **Per-dimension** | ❌ None | ❌ None | N/A |
| **Covariance Learning** | ✅ **Uses correlation matrix** | ❌ None | ❌ None | N/A |
| **Adaptive Mutation** | ✅ **Global × Local** | ❌ Global only | ❌ Fixed | N/A |
| **Multiple Restarts** | ✅ **2-3 restarts** (wrapper) | ❌ Single run | ❌ Single run | ❌ Single run |
| **Parameter Adaptation** | ✅ **Based on search space** | ❌ Fixed | ❌ Fixed | N/A |

---

## 🔬 Technical Deep Dive

### 1. Mutation Strategy

#### HW6 Algorithm:
```python
# Per-dimension step sizes (different for each dimension!)
step_sizes = sigma_g * sigma_local  # Shape: (dim,)

# Mutation with adaptive per-dimension step sizes
step = rng.normal(loc=0.0, scale=step_sizes, size=dim)
child = parent + step
```

**Why this is better:**
- If dimension 0 needs small steps (near optimum) but dimension 1 needs large steps (far from optimum), your algorithm adapts to each!
- Baseline algorithms use the same step size for all dimensions

#### (μ+λ)-ES Baseline:
```python
# Fixed sigma for ALL dimensions (same step size everywhere!)
sigma = 0.3 * np.mean(bounds[:, 1] - bounds[:, 0])  # Scalar, not array!
child = parent + sigma * np.random.randn(dim)  # Same sigma for all dims
```

**Problem:** Can't adapt to different sensitivities per dimension

---

### 2. Local Sigma Adaptation (Your Unique Feature!)

#### HW6 Algorithm:
```python
def update_local_sigmas_from_parents(parents, sigma_local):
    # Compute covariance matrix from parent population
    cov = np.cov(centered.T)  # Shape: (dim, dim)
    
    # Extract variance per dimension
    var = np.diag(cov)  # Variance for each dimension
    
    # Compute correlation matrix
    corr = cov / (std × std)
    
    # Increase sigma for dimensions that are highly correlated
    avg_abs_corr = np.mean(np.abs(corr), axis=1)
    new_sigma_local = base_sigma * (1.0 + alpha_corr * avg_abs_corr)
```

**What this does:**
- Learns which dimensions are correlated
- Increases mutation for correlated dimensions (needs more exploration)
- Decreases mutation for independent dimensions (can refine locally)
- **This is like a simplified CMA-ES covariance matrix adaptation!**

#### Baseline Algorithms:
- No covariance/correlation learning at all
- Can't detect relationships between dimensions

---

### 3. Success Rate Calculation

#### HW6 Algorithm:
```python
# Success = offspring better than MEDIAN parent fitness
parent_baseline = np.median(parent_f)
successes = np.sum(offspring_f < parent_baseline)
```

**Why median?**
- More robust to outliers
- Better for population-based algorithms
- Adapts to overall population quality

#### 1/5 Rule ES Baseline:
```python
# Success = offspring better than CURRENT parent
if f_new < f(x):
    successes += 1
```

**Difference:** Uses current individual vs. population median (less robust)

---

### 4. Wrapper Function (Competition Strategy)

#### HW6 Wrapper:
```python
# Adaptive based on search space size
if range_size > 100:  # Large search space
    n_restarts = 3
    gen_per_run = 350
    mu_val = 6      # Larger population
    lam_val = 15    # More offspring
elif range_size > 10:  # Medium
    n_restarts = 2
    mu_val = 5
    lam_val = 12
else:  # Small
    n_restarts = 2
    mu_val = 4
    lam_val = 10
```

**Advantages:**
- Multiple restarts avoid local minima
- Adapts to problem difficulty (larger spaces = more exploration)
- Larger populations for harder problems

#### Baseline Algorithms:
- Single run only
- Fixed parameters regardless of problem

---

## 🎯 Why Your Algorithm Performs Better

### 1. **Per-Dimension Adaptation**
- Your algorithm can use small steps in one dimension while using large steps in another
- Baseline algorithms are "one-size-fits-all" - same step size everywhere

### 2. **Correlation Learning**
- Your algorithm detects when dimensions are related (e.g., x and y both need to increase together)
- Adjusts mutation strategy accordingly
- Baseline algorithms treat all dimensions independently

### 3. **Robustness**
- Multiple restarts increase chance of finding global optimum
- Adaptive parameters work across different problem types
- Baseline algorithms are less robust

### 4. **Hybrid Approach**
- Combines best of (μ+λ) selection (from baseline)
- Adds global step size adaptation (from 1/5 rule)
- Adds local step sizes (your innovation!)
- Adds correlation learning (CMA-ES inspired)

---

## 📈 Performance Implications

Based on your test results, HW6 typically wins because:

1. **Better Exploration**: Per-dimension step sizes allow better exploration in complex landscapes
2. **Better Exploitation**: Can refine locally in dimensions that are near optimum
3. **Correlation Awareness**: Handles functions with correlated dimensions better
4. **Robustness**: Multiple restarts reduce chance of getting stuck

---

## 🔍 Example: Why Per-Dimension Matters

Consider optimizing `f(x, y) = (x-1)² + 100*(y-2)²`:
- Dimension x is near optimum (x ≈ 1) → needs small steps
- Dimension y is far from optimum (y ≈ 0, target is 2) → needs large steps

**HW6 Algorithm:**
- σ_local[0] (for x) → small (learned from population)
- σ_local[1] (for y) → large (learned from population)
- Result: Efficient convergence!

**Baseline Algorithms:**
- Same sigma for x and y
- Either: too small for y (slow) OR too large for x (unstable)
- Result: Less efficient

---

## 💡 Summary

Your HW6 algorithm is superior because it:

1. ✅ **Learns per-dimension step sizes** (no baseline does this)
2. ✅ **Uses correlation information** (no baseline does this)
3. ✅ **Combines global + local adaptation** (unique hybrid approach)
4. ✅ **Uses multiple restarts** (baselines don't)
5. ✅ **Adapts parameters to problem** (baselines are fixed)

This makes it a **true hybrid** combining:
- (μ+λ) Evolution Strategy structure
- 1/5 Success Rule for global adaptation
- CMA-ES-inspired correlation learning
- Competition-ready robustness strategies

**This is why your algorithm performs better across different benchmark functions!** 🏆

