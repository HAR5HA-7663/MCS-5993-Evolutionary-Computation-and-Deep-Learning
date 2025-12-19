# Extra Credit Work Summary

**Student:** V Harsha Vardhan Yellela
**LTU ID:** 000798754
**Course:** MCS-5993 Evolutionary Computation and Deep Learning
**Term:** Fall 2025

---

## Navigation Guide for Professor

| Extra Credit | Location in Zip File | Section/Cell to Review |
|--------------|---------------------|------------------------|
| **EC 1** | `Extra_credit/JenaWeather_Seq2Seq_EA_with_ES.ipynb` | Full notebook (Sections 1-9) |
| **EC 2** | `Term-project_Deep_Food_image_classifier/FoodClassifier.ipynb` | **Section 6: Keras Tuner Hyperband** |
| **EC 4** | `Term-project_Deep_Food_image_classifier/FoodClassifier.ipynb` | **Section 7: DEAP (μ+λ)-ES HPO** |

---

## Extra Credit 1: Sequence-to-Sequence Time Series Prediction (Jena Weather Dataset)

**File Location:** `Extra_credit/JenaWeather_Seq2Seq_EA_with_ES.ipynb`

I implemented a multi-step sequence-to-sequence (Seq2Seq) time series forecasting model using the Jena Weather Dataset. The model uses 72 hours (3 days) of past weather data to predict the next 6 hours across 5 meteorological features: Temperature, Pressure, Density, Wind Speed, and Max Wind Speed.

**Architecture:** Encoder-decoder LSTM with RepeatVector for multi-step output generation.

**Dual Evolutionary Optimization Approach:**
1. **Genetic Algorithm (GA)** for hyperparameter optimization in the outer loop, evolving LSTM units (32-128), learning rate (0.0005-0.01), and dropout rate (0.1-0.5).
2. **ES(1+1) with 1/5 Success Rule** for fine-tuning the final Dense layer weights after gradient-based training.

**Results:**

| Feature | MAE | RMSE | R² |
|---------|-----|------|-----|
| Temperature | 0.420 | 0.634 | 0.993 |
| Pressure | 0.200 | 0.278 | 0.999 |
| Density | 1.830 | 2.735 | 0.994 |
| Wind Speed | 0.788 | 1.045 | 0.566 |
| Max Wind Speed | 0.980 | 1.334 | 0.710 |
| **Overall** | 0.843 | 1.472 | 1.000 |

**Discussion:** The model achieved excellent results for Temperature, Pressure, and Density prediction (R² > 0.99). Wind-related features proved more challenging due to their inherently chaotic nature. The GA successfully found good hyperparameters (107 LSTM units, LR=0.001363, dropout=0.301). The ES(1+1) weight refinement showed minimal additional improvement since the model had already converged well through gradient descent, demonstrating that evolutionary weight optimization is most effective when the loss landscape has not been thoroughly explored.

---

## Extra Credit 2: Keras Tuner for Hyperparameter Optimization (4th HPO Algorithm)

**File Location:** `Term-project_Deep_Food_image_classifier/FoodClassifier.ipynb`
**Navigate to:** Section 6 - "Keras Tuner Hyperband" (Cells labeled "KERAS TUNER CELL 1" through "KERAS TUNER CELL 8")

I implemented Keras Tuner with the Hyperband search algorithm as an additional HPO method for my Deep Food Image Classifier term project. This served as the 4th HPO algorithm alongside ES(1+1), CMA-ES, and Random Search.

**Search Space:** learning_rate, dropout_rate, label_smoothing
**Configuration:** Hyperband with max_epochs=30, factor=3, 2 iterations

**Best Hyperparameters Found:**
- Learning Rate: 0.001409
- Dropout Rate: 0.300
- Label Smoothing: 0.30

**Results:**
- **Test Accuracy: 79.27%**
- **Improvement over Baseline: +7.41%** (baseline was 72.47%)

Keras Tuner Hyperband efficiently pruned poor-performing trials early, making it computationally efficient while achieving results competitive with my hand-coded ES(1+1) implementation.

---

## Extra Credit 4: DEAP Library Integration for Evolutionary HPO

**File Location:** `Term-project_Deep_Food_image_classifier/FoodClassifier.ipynb`
**Navigate to:** Section 7 - "DEAP (μ+λ)-ES Hyperparameter Optimization" (starts after Keras Tuner section)

I integrated the DEAP (Distributed Evolutionary Algorithms in Python) library into my term project, implementing a (μ+λ)-ES (Evolution Strategy) for hyperparameter optimization.

**Algorithm Configuration:**
- μ (parents) = 5
- λ (offspring) = 10
- Generations = 5
- Selection: Best μ individuals from combined parent+offspring pool

**Search Space:** learning_rate, dropout_rate, label_smoothing, rotation_range, zoom_range

**Best Hyperparameters Found:**
- Learning Rate: 0.002215
- Dropout Rate: 0.388
- Label Smoothing: 0.077
- Rotation Range: 0.031
- Zoom Range: 0.327

**Results:**
- **Test Accuracy: 79.33%**
- **Improvement over Baseline: +9.47%**

The DEAP (μ+λ)-ES implementation provided population-based exploration advantages over the single-parent ES(1+1), achieving competitive results while demonstrating the practical application of evolutionary algorithms through a well-established Python library.

---

## Summary Comparison Table (Term Project - All 6 HPO Methods)

| Method | Test Accuracy | Improvement |
|--------|---------------|-------------|
| Baseline CNN | 72.47% | — |
| **ES(1+1)** | **79.80%** | **+10.11%** |
| DEAP (μ+λ)-ES | 79.33% | +9.47% |
| Keras Tuner (Hyperband) | 79.27% | +9.38% |
| CMA-ES | 77.07% | +6.34% |
| Random Search | 77.00% | +6.25% |

**Best Overall Method:** ES(1+1) with 79.80% test accuracy

---

## Zip File Contents

```
Harsha_Yellela_Extra_Credit.zip
│
├── Extra_Credit_Summary.md          <-- THIS FILE (start here)
│
├── Extra_credit/                    <-- EXTRA CREDIT 1
│   ├── JenaWeather_Seq2Seq_EA_with_ES.ipynb
│   └── ECDL-EC-report.pdf           (detailed report)
│
└── Term-project_Deep_Food_image_classifier/   <-- EXTRA CREDITS 2 & 4
    ├── FoodClassifier.ipynb         (Section 6 = EC2, Section 7 = EC4)
    └── [supporting files: CSVs, images, etc.]
```
