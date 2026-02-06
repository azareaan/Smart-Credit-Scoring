# Fuzzy Credit Scoring for Risk Assessment in FinTech Systems
### A Hybrid Deep Learning & Fuzzy Logic Approach

This repository presents a professional-grade credit risk assessment system. It combines **Unsupervised Deep Learning (Autoencoders)** to detect behavioral anomalies and a **Fuzzy Inference System (FIS)** to calculate a final, explainable risk score.



## 🚀 Overview
Traditional credit scoring often fails to capture "logical inconsistencies" in borrower behavior. This project solves that by:
1. **Anomaly Detection Core:** Using a PyTorch-based Autoencoder to learn the patterns of "Good Borrowers" (Target=0).
2. **Logic Inconsistency Features:** Engineering specific features (e.g., External Source Variance) that highlight conflicts in credit reports.
3. **Fuzzy Risk Layer:** An expert system that takes AI-generated anomaly scores and traditional credit metrics to provide a human-readable Risk Index (0-100).

## 📂 Project Structure
```text
├── data/               # Raw dataset (Home Credit Default Risk)
├── src/                # Source modules
│   ├── features.py     # Feature engineering & Inconsistency logic
│   ├── model.py        # PyTorch Autoencoder architecture
│   ├── train.py        # Semi-supervised training loop
│   └── fuzzy_logic.py  # Fuzzy Inference System (Mamdani FIS)
├── outputs/            # Generated results and visualizations
├── main.py             # Step 1: Run Anomaly Detection
└── main_fuzzy.py       # Step 2: Run Fuzzy Scoring
```

🛠️ Installation

    Clone this repository.

    Ensure you have Python 3.9+ installed.

    Place application_train.csv in the data/ folder.

    Install dependencies:
    Bash

    pip install -r requirements.txt

📈 Methodology
1. The Autoencoder (AE)

The model is trained only on "Repayers" (Target=0). By using a restricted bottleneck (4 dimensions), it acts as a regularizer. When the model encounters a "Default" profile or a "Logical Inconsistency," the Reconstruction Error (MSE) spikes, flagging it as an anomaly.
2. Fuzzy Inference System (FIS)

We use the Mamdani method to map:

    Input 1: AI Anomaly Score (High/Medium/Low)

    Input 2: External Source Mean (High/Medium/Low)

    Output: Final Risk Score (0-100)

📊 Results

    Unsupervised AUC: ~0.65 (Significant improvement over standard baseline).

    Explainability: Unlike black-box models, the Fuzzy system provides clear reasons for every risk score generated.

🤝 Acknowledgments

Built for academic research in FinTech Risk Systems. This project utilizes the Home Credit Default Risk dataset.


---