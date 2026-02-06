# Neural Network-Based Anomaly Detection & Correction in Financial Data

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.10+](https://img.shields.io/badge/TensorFlow-2.10+-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Project Overview

This project implements a context-aware anomaly detection and correction system for financial credit data using deep learning. The system is designed to identify and correct contextual anomalies (not just extreme values) and demonstrably improves downstream credit risk modeling.

### Key Features

- **Contextual Anomaly Detection**: Identifies inconsistencies between financial attributes (e.g., low income + high credit limit)
- **Missing Value Strategy**: Treats missing data as informative signals rather than noise
- **Neural Network Architecture**: Autoencoder for pattern learning + Isolation Forest for validation
- **Smart Correction**: Context-aware reconstruction using learned manifolds
- **Proven Impact**: +3.3% improvement in downstream credit scoring (LR) and +1.8% (LightGBM)

## 📊 Dataset

**Home Credit Default Risk** (Kaggle Competition)
- 307,511 loan applications
- 122 original features
- 20 key financial features selected
- Known anomalies for validation (DAYS_EMPLOYED bug)

## 🏗️ Architecture

```
Input Data (20 features)
    ↓
[Missing Flag Creation + Mean Imputation]
    ↓
[StandardScaler]
    ↓
┌──────────────────────────────────────┐
│  Autoencoder (Primary - 70%)         │
│  ├─ Encoder: [20→64→32→16→8]       │
│  └─ Decoder: [8→16→32→64→20]       │
└──────────────────────────────────────┘
    ↓
[Reconstruction Error] → Anomaly Score
    ↓
┌──────────────────────────────────────┐
│  Isolation Forest (Secondary - 30%)  │
│  └─ 100 trees, contamination=0.1    │
└──────────────────────────────────────┘
    ↓
[Ensemble: 0.7*AE + 0.3*IF]
    ↓
[Context-Aware Correction]
    ↓
Corrected Data + Confidence Scores
```

## 📁 Project Structure

```
financial-anomaly-detection/
├── src/
│   ├── preprocessing.py      # Data preprocessing pipeline
│   ├── models.py             # Autoencoder & ensemble models
│   ├── correction.py         # Smart anomaly correction
│   ├── evaluation.py         # Intrinsic & downstream metrics
│   └── visualization.py      # Plot utilities
├── notebooks/
│   ├── 01_EDA.ipynb          # Exploratory data analysis
│   ├── 02_Training.ipynb     # Model training
│   ├── 03_Evaluation.ipynb   # Full evaluation
│   └── 04_Demo.ipynb         # Quick demonstration
├── data/                     # Dataset (gitignored)
├── models/                   # Trained models (gitignored)
├── results/
│   ├── figures/             # Visualizations
│   └── metrics/             # Evaluation results
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/financial-anomaly-detection.git
cd financial-anomaly-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Download Dataset

Download from [Kaggle](https://www.kaggle.com/c/home-credit-default-risk):
```bash
# Using Kaggle API
kaggle competitions download -c home-credit-default-risk
unzip application_train.csv.zip -d data/
```

### Run Full Pipeline

```python
from src.preprocessing import FinancialPreprocessor
from src.models import AnomalyDetectionSystem
from src.evaluation import Evaluator

# 1. Preprocess
preprocessor = FinancialPreprocessor()
X, y, features = preprocessor.fit_transform('data/application_train.csv')

# 2. Train
detector = AnomalyDetectionSystem()
detector.fit(X[y == 0])  # Train on normal samples only

# 3. Detect & Correct
results = detector.detect_and_correct(X)

# 4. Evaluate
evaluator = Evaluator()
metrics = evaluator.evaluate_all(X, results['corrected'], y)
```

## 📈 Results

### Intrinsic Performance (Anomaly Detection)

| Metric | Score |
|--------|-------|
| Precision | 0.78 |
| Recall | 0.85 |
| F1-Score | 0.81 |
| ROC-AUC | 0.88 |

### Downstream Impact (Credit Scoring)

| Model | Original AUC | Corrected AUC | Improvement |
|-------|--------------|---------------|-------------|
| Logistic Regression | 0.721 | 0.745 | **+3.3%** ✨ |
| LightGBM | 0.758 | 0.772 | **+1.8%** ✨ |

> **Note**: In financial industry, even 1-2% AUC improvement is considered significant.

## 🔬 Key Scientific Contributions

1. **Contextual Anomaly Definition**: Anomalies defined as inconsistencies between features, not just extreme values
2. **Missing Value Strategy**: Binary flagging + mean imputation preserves information for neural networks
3. **Proven Downstream Impact**: Measurable improvement in credit risk models

## 📊 Visualizations

### Anomaly Distribution
![Anomaly Distribution](results/figures/anomaly_distribution.png)

### Downstream Impact
![Impact](results/figures/downstream_impact.png)

### Feature Space (t-SNE)
![t-SNE](results/figures/tsne_visualization.png)

## 🛠️ Technical Details

### Preprocessing Strategy

```python
# Critical: Missing values are informative signals!
1. Create binary flags: is_missing_feature
2. Impute with mean (preserves distribution for StandardScaler)
3. Apply StandardScaler (Z-score normalization)
```

### Why Autoencoder?

- **Learns contextual patterns**: "For this income, what credit is normal?"
- **Non-linear manifold**: Captures complex feature relationships
- **Reconstruction = Correction**: Projects anomalies onto normal manifold

### Ensemble Rationale

- **Autoencoder (70%)**: Contextual pattern learning
- **Isolation Forest (30%)**: Distributional outlier detection
- **Complementary**: Different perspectives on anomalies

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@software{financial_anomaly_detection_2024,
  author = {Your Name},
  title = {Neural Network-Based Anomaly Detection and Correction in Financial Data},
  year = {2024},
  url = {https://github.com/yourusername/financial-anomaly-detection}
}
```

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

- **Author**: Your Name
- **Email**: your.email@example.com
- **LinkedIn**: [Your Profile](https://linkedin.com/in/yourprofile)

## 🙏 Acknowledgments

- Home Credit Group for the dataset
- Kaggle community for valuable insights
- Research papers on anomaly detection in financial data

---

**⭐ Star this repository if you find it helpful!**
