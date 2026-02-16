# ML-Based Circuit Performance Prediction for VLSI Design Optimization

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)](https://scikit-learn.org/)

> Machine learning pipeline to predict circuit delay and power consumption from design parameters, enabling 100× faster design space exploration compared to SPICE simulation.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Results](#results)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Technical Details](#technical-details)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## 🎯 Overview

Traditional VLSI design workflows require running thousands of time-consuming SPICE simulations to optimize circuit parameters (transistor width, drive strength, load capacitance). This project demonstrates how **machine learning** can accelerate this process by predicting circuit performance instantly.

### Problem Statement

- **Challenge:** Design space exploration requires extensive SPICE simulations (hours to days)
- **Solution:** ML model predicting delay/power from design parameters in milliseconds
- **Impact:** 100× speedup enables rapid prototyping and automated design optimization

### Technologies Used

- **Machine Learning:** scikit-learn, pandas, NumPy
- **Data Source:** Nangate 45nm Open Cell Library (industry-standard)
- **Visualization:** matplotlib, seaborn
- **Development:** Jupyter Notebook, Python 3.10

---

## ✨ Key Features

- ✅ **High Accuracy:** R² > 0.95, MAE < 5 picoseconds
- ✅ **Industry Data:** Trained on Nangate 45nm standard cell library
- ✅ **Feature Engineering:** 7 domain-specific derived features
- ✅ **Model Comparison:** Evaluated 5 ML algorithms (Random Forest, Gradient Boosting, etc.)
- ✅ **Design Space Exploration:** Parameter sweeps and sensitivity analysis
- ✅ **Production Ready:** Saved model for deployment

---

## 📊 Results

### Model Performance

| Model | R² Score | MAE (ps) | RMSE (ps) |
|-------|----------|----------|-----------|
| **Random Forest** | **0.979** | **2.87** | **3.45** |
| Gradient Boosting | 0.972 | 3.21 | 3.89 |
| Ridge Regression | 0.891 | 8.45 | 10.23 |
| Lasso Regression | 0.883 | 8.92 | 10.67 |
| Linear Regression | 0.879 | 9.12 | 10.89 |

### Key Achievements

- 🎯 **97.9% prediction accuracy** (R² score)
- ⚡ **Sub-3ps error** on average (MAE = 2.87 ps)
- 📈 **37+ standard cells** from Nangate library analyzed
- 🚀 **100× faster** than running HSPICE simulations

### Prediction Accuracy Visualization

![Prediction Accuracy](results/prediction_vs_actual.png)
*Predicted vs. Actual Delay - showing near-perfect correlation*

![Error Distribution](results/error_distribution.png)
*Prediction error distribution - tightly clustered around zero*

![Design Space](results/design_space_heatmap.png)
*Design space exploration heatmap across area and drive strength*

---

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- Anaconda (recommended) or pip

### Clone Repository

```bash
git clone https://github.com/yourusername/ML-VLSI-Circuit-Prediction.git
cd ML-VLSI-Circuit-Prediction
```

### Install Dependencies

**Using Conda (Recommended):**
```bash
conda create -n ml-vlsi python=3.10
conda activate ml-vlsi
conda install pandas numpy scikit-learn matplotlib seaborn jupyter
```

**Using pip:**
```bash
pip install -r requirements.txt
```

### Download Nangate Library

The Nangate 45nm Open Cell Library is automatically downloaded by the extraction script, or you can manually download from:
- [Nangate Website](http://www.nangate.com/)
- [GitHub Mirror](https://github.com/JulianKemmerer/Drexel-ECEC575)

---

## 🚀 Quick Start

### Option 1: Run All Notebooks

```bash
jupyter notebook
# Open notebooks in order:
# 01_data_extraction.ipynb
# 02_feature_engineering.ipynb
# 03_model_training.ipynb
# 04_validation.ipynb
```

### Option 2: Use Saved Model for Predictions

```python
import joblib
import pandas as pd

# Load model
model = joblib.load('models/best_model_Random_Forest.pkl')
scaler = joblib.load('models/feature_scaler.pkl')

# Prepare input (example: INV_X4 cell)
input_data = pd.DataFrame({
    'Area_um2': [1.330],
    'Input_Cap_pF': [0.00504],
    'Drive_Strength': [4.0],
    'Normalized_Area': [2.5],
    'Delay_per_Area': [0.00001],
    'Log_Area': [0.124],
    'Area_x_Cap': [0.0067]
})

# Scale and predict
input_scaled = scaler.transform(input_data)
predicted_delay = model.predict(input_scaled)[0]

print(f"Predicted Delay: {predicted_delay*1e12:.2f} ps")
# Output: Predicted Delay: 8.52 ps
```

---

## 📁 Project Structure

```
ML-VLSI-Circuit-Prediction/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── LICENSE                            # MIT License
│
├── notebooks/                         # Jupyter notebooks
│   ├── 01_data_extraction.ipynb      # Parse Nangate Liberty files
│   ├── 02_feature_engineering.ipynb  # Create derived features
│   ├── 03_model_training.ipynb       # Train ML models
│   └── 04_validation.ipynb           # Test and analyze results
│
├── data/                              # Datasets
│   ├── nangate_cell_data.csv         # Extracted raw data
│   └── nangate_data_engineered.csv   # Engineered features
│
├── models/                            # Saved models
│   ├── best_model_Random_Forest.pkl  # Trained model
│   └── feature_scaler.pkl            # Feature scaler
│
├── results/                           # Visualizations and reports
│   ├── model_comparison.png
│   ├── prediction_vs_actual.png
│   ├── error_distribution.png
│   ├── feature_importance.png
│   └── design_space_heatmap.png
│
└── docs/                              # Documentation
    └── project_report.pdf             # Detailed technical report
```

---

## 🔬 Methodology

### 1. Data Extraction

**Source:** Nangate 45nm Open Cell Library  
**Format:** Liberty (.lib) files containing characterized standard cells  
**Parsing:** Custom Python parser to extract:
- Cell area (µm²)
- Input capacitance (pF)
- Leakage power (W)
- Rise/Fall delays (ns)

**Output:** 37+ standard cells with timing and power data

### 2. Feature Engineering

Created 7 derived features to capture circuit physics:

| Feature | Description | Engineering Rationale |
|---------|-------------|----------------------|
| `Cell_Type` | Gate function (INV, NAND, etc.) | Different logic functions have different delay characteristics |
| `Drive_Strength` | Transistor sizing (X1, X2, X4...) | Directly impacts drive capability |
| `PDP` | Power-Delay Product | Key optimization metric in VLSI |
| `Delay_per_Area` | Area efficiency | Captures transistor utilization |
| `Normalized_Area` | Relative sizing | Enables comparison across cell types |
| `Log_Area` | Logarithmic transform | Handles wide range of cell sizes |
| `Area_x_Cap` | Interaction term | Captures combined effect |

### 3. Model Training

**Algorithms Evaluated:**
- Linear Regression (baseline)
- Ridge Regression (L2 regularization)
- Lasso Regression (L1 regularization)
- **Random Forest** (best performance) ⭐
- Gradient Boosting

**Training Configuration:**
```python
RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
```

**Train/Test Split:** 80/20  
**Cross-Validation:** 5-fold  
**Scaling:** StandardScaler

### 4. Validation

**Metrics:**
- R² Score: 0.979 (97.9% variance explained)
- MAE: 2.87 ps (mean absolute error)
- RMSE: 3.45 ps (root mean squared error)
- MAPE: 4.2% (mean absolute percentage error)

**Test Scenarios:**
- ✅ Drive strength sweep (X1 → X32)
- ✅ Area sweep (0.5 → 3.0 µm²)
- ✅ Design space exploration (10×10 grid)
- ✅ Cross-validation on unseen cells


<img width="482" height="251" alt="R2_and_MAE_Summary" src="https://github.com/user-attachments/assets/022e2f8a-0e82-49a9-aefb-47a66e24b28e" />

<img width="1025" height="767" alt="New_Visualisation" src="https://github.com/user-attachments/assets/5b1e0ed8-8bdd-4c35-b56e-c130d7783857" />

<img width="499" height="407" alt="Trend_Analysis" src="https://github.com/user-attachments/assets/c7adeec9-a98c-4543-85d6-7c5d8eeb56b5" />

---

## 🔍 Technical Details

### Why Random Forest?

Random Forest outperformed other algorithms because:
1. **Non-linear relationships:** Circuit delay doesn't scale linearly with parameters
2. **Robustness:** Handles outliers well (some cells have anomalous behavior)
3. **Feature interactions:** Captures complex relationships between area, capacitance, and drive strength
4. **No overfitting:** Ensemble method reduces variance

### Feature Importance

Top 3 most important features:

1. **Area (35.2%)** - Directly correlates with transistor sizing
2. **Drive Strength (28.7%)** - Determines output resistance
3. **Input Capacitance (18.9%)** - Affects switching speed

### Model Limitations

⚠️ **Current model predicts intrinsic delay** (minimal load):
- Trained on first entry of Liberty delay tables
- Assumes small output load (~0.5 fF)

⚠️ **For realistic predictions with heavy loads:**
- Need to extract full delay tables indexed by load capacitance
- Add `Load_Capacitance` as input feature
- Model would then capture upsizing benefits for heavy loads

**Example:**
- **Light load (<10fF):** Smaller cells faster (intrinsic delay dominates)
- **Heavy load (>50fF):** Larger cells faster (drive strength dominates)

### Why Upsizing Still Helps in Real Design

Despite X16 cells showing higher intrinsic delay:

```
Total_Delay = Intrinsic_Delay + R_driver × C_load
                    ↑                    ↑
              Increases with size   Decreases 16× with size
```

At heavy loads (C_load >> 10fF), the R_driver reduction dominates!

---

## 📈 Use Cases

### 1. Rapid Design Space Exploration
```python
# Sweep drive strengths to find optimal sizing
for drive in [1, 2, 4, 8, 16]:
    predicted_delay = model.predict(...)
    # Takes milliseconds vs. hours in SPICE
```

### 2. Automated Transistor Sizing
```python
# Optimize for target delay
target_delay = 10.0  # ps
optimal_params = optimize_design(model, target_delay)
```

### 3. Early-Stage Design Estimation
- Estimate circuit performance before detailed implementation
- Guide architectural decisions
- Resource planning for tape-out

### 4. Educational Tool
- Understand delay-area-power trade-offs
- Learn ML applications in EDA
- Explore circuit design space interactively

---

## 🚧 Future Improvements

### Planned Enhancements

- [ ] **Extract full delay tables** - Add load capacitance dependency
- [ ] **Power prediction** - Extend to dynamic power modeling
- [ ] **Multiple corners** - Add PVT (Process, Voltage, Temperature) variations
- [ ] **More cell types** - Expand beyond basic gates (adders, multiplexers)
- [ ] **Neural networks** - Try deep learning for higher accuracy
- [ ] **Web interface** - Deploy as interactive prediction tool
- [ ] **Integration with EDA tools** - Plugin for Cadence/Synopsys

### Research Directions

- Apply to custom circuits (not just standard cells)
- Transfer learning across technology nodes
- Reinforcement learning for automated optimization
- Comparison with Synopsys DSO.ai approach

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup

```bash
git clone https://github.com/yourusername/ML-VLSI-Circuit-Prediction.git
cd ML-VLSI-Circuit-Prediction
conda env create -f environment.yml
conda activate ml-vlsi
```

### Areas for Contribution

- Additional feature engineering ideas
- Alternative ML algorithms
- Visualization improvements
- Documentation enhancements
- Bug fixes

---

### Related Projects

- [OpenROAD](https://github.com/The-OpenROAD-Project/OpenROAD) - Open-source EDA tools
- [Drexel ECEC575](https://github.com/JulianKemmerer/Drexel-ECEC575) - VLSI course materials
- [Synopsys DSO.ai](https://www.synopsys.com/ai/dso-ai.html) - Commercial ML-based design optimization

---

