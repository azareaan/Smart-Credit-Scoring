# Usage Guide

## Quick Start (5 minutes)

### 1. Installation

```bash
git clone https://github.com/yourusername/financial-anomaly-detection.git
cd financial-anomaly-detection
pip install -r requirements.txt
```

### 2. Download Data

```bash
# Using Kaggle API (recommended)
kaggle competitions download -c home-credit-default-risk
unzip application_train.csv.zip -d data/

# Or manually download from:
# https://www.kaggle.com/c/home-credit-default-risk/data
```

### 3. Run Example

```bash
python example.py
```

That's it! Results will be in `results/` folder.

---

## Detailed Usage

### Option 1: Using the Complete Pipeline

```python
from src import (
    FinancialPreprocessor,
    AnomalyDetectionSystem,
    Evaluator,
    plot_downstream_impact
)

# Load & preprocess
preprocessor = FinancialPreprocessor()
X, y, features = preprocessor.fit_transform('data/application_train.csv')

# Train
detector = AnomalyDetectionSystem(input_dim=20)
detector.fit(X[y == 0], X_all=X)

# Detect & correct
results = detector.detect_and_correct(X)

# Evaluate
evaluator = Evaluator()
metrics = evaluator.evaluate_all(
    X, results['corrected'], y,
    y_true_anomaly=(df['DAYS_EMPLOYED'] == 365243).values,
    y_pred_flags=results['anomaly_mask'].astype(int),
    y_pred_scores=results['scores']
)
```

### Option 2: Step-by-Step (Custom Pipeline)

#### Step 1: Preprocessing

```python
from src.preprocessing import FinancialPreprocessor

preprocessor = FinancialPreprocessor()

# Fit on training data
X_train, y_train, features = preprocessor.fit_transform('data/application_train.csv')

# Transform new data (using fitted preprocessor)
X_test = preprocessor.transform(df_test)

# Access feature names
print(preprocessor.feature_names)

# Access encoders
print(preprocessor.encoders.keys())
```

#### Step 2: Train Autoencoder

```python
from src.models import AutoencoderModel

# Initialize
autoencoder = AutoencoderModel(input_dim=20, latent_dim=8)

# Build architecture
autoencoder.build()
print(autoencoder.model.summary())

# Train (on normal samples only)
X_normal = X[y == 0]
history = autoencoder.fit(
    X_normal,
    epochs=50,
    batch_size=512,
    validation_split=0.2,
    verbose=1
)

# Save
autoencoder.save('models/autoencoder')
```

#### Step 3: Train Isolation Forest

```python
from src.models import IsolationForestModel

# Initialize
iso_forest = IsolationForestModel(
    n_estimators=100,
    contamination=0.1
)

# Train (on all data)
iso_forest.fit(X)

# Save
iso_forest.save('models/isolation_forest')
```

#### Step 4: Ensemble Detection

```python
from src.models import AnomalyDetectionSystem

# Initialize with custom weights
detector = AnomalyDetectionSystem(
    input_dim=20,
    ae_weight=0.7,
    if_weight=0.3
)

# Or load pre-trained
detector.load('models/')

# Predict
results = detector.predict(X)
print(f"Detected {results['flags'].sum()} anomalies")

# Detect and correct
correction_results = detector.detect_and_correct(X)
X_corrected = correction_results['corrected']
```

#### Step 5: Correction

```python
# Access correction details
anomaly_mask = correction_results['anomaly_mask']
confidence = correction_results['confidence']

# Analyze corrections
n_corrected = anomaly_mask.sum()
avg_confidence = confidence.mean()

print(f"Corrected {n_corrected} records")
print(f"Average confidence: {avg_confidence:.3f}")

# Identify low-confidence corrections
low_conf_idx = np.where(confidence < 0.5)[0]
print(f"Low confidence corrections: {len(low_conf_idx)}")
```

#### Step 6: Evaluation

```python
from src.evaluation import Evaluator

evaluator = Evaluator()

# Intrinsic evaluation
intrinsic = evaluator.evaluate_detection(
    y_true_anomaly=known_anomalies,
    y_pred_flags=results['flags'],
    y_pred_scores=results['scores']
)

# Downstream evaluation
downstream = evaluator.evaluate_downstream(
    X_original=X,
    X_corrected=X_corrected,
    y_target=y
)

# Save results
evaluator.save_results('results/metrics/evaluation.json')

# Generate report
report = evaluator.generate_report()
print(report)
```

#### Step 7: Visualization

```python
from src.visualization import (
    plot_anomaly_distribution,
    plot_detection_metrics,
    plot_downstream_impact,
    plot_feature_space_2d,
    create_evaluation_summary
)

# Anomaly distribution
plot_anomaly_distribution(
    results['scores'],
    results['flags'],
    detector.ensemble_threshold,
    save_path='results/figures/distribution.png'
)

# Detection metrics
plot_detection_metrics(
    intrinsic,
    save_path='results/figures/metrics.png'
)

# Downstream impact
plot_downstream_impact(
    downstream,
    save_path='results/figures/impact.png'
)

# Feature space (t-SNE)
plot_feature_space_2d(
    X, X_corrected,
    anomaly_mask,
    method='tsne',
    save_path='results/figures/tsne.png'
)

# Complete summary
create_evaluation_summary(
    intrinsic,
    downstream,
    save_path='results/figures/summary.png'
)
```

---

## Advanced Usage

### Custom Autoencoder Architecture

```python
from tensorflow import keras
from tensorflow.keras import layers

def build_custom_autoencoder(input_dim):
    # Custom encoder
    encoder_input = layers.Input(shape=(input_dim,))
    x = layers.Dense(128, activation='relu')(encoder_input)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='relu')(x)
    encoded = layers.Dense(16, activation='relu')(x)
    
    encoder = keras.Model(encoder_input, encoded)
    
    # Custom decoder
    decoder_input = layers.Input(shape=(16,))
    x = layers.Dense(64, activation='relu')(decoder_input)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation='relu')(x)
    decoded = layers.Dense(input_dim, activation='linear')(x)
    
    decoder = keras.Model(decoder_input, decoded)
    
    # Full model
    ae_input = layers.Input(shape=(input_dim,))
    encoded_repr = encoder(ae_input)
    reconstructed = decoder(encoded_repr)
    autoencoder = keras.Model(ae_input, reconstructed)
    
    autoencoder.compile(optimizer='adam', loss='mse')
    
    return autoencoder, encoder, decoder

# Use custom architecture
from src.models import AutoencoderModel

ae = AutoencoderModel(input_dim=20)
ae.model, ae.encoder, ae.decoder = build_custom_autoencoder(20)
ae.fit(X_normal)
```

### Custom Ensemble Weights

```python
# Test different weight combinations
results_list = []

for ae_weight in [0.5, 0.6, 0.7, 0.8, 0.9]:
    if_weight = 1 - ae_weight
    
    detector = AnomalyDetectionSystem(
        input_dim=20,
        ae_weight=ae_weight,
        if_weight=if_weight
    )
    
    # ... train and evaluate ...
    
    results_list.append({
        'weights': (ae_weight, if_weight),
        'f1': f1_score,
        'auc': roc_auc
    })

# Find best weights
best = max(results_list, key=lambda x: x['f1'])
print(f"Best weights: {best['weights']}")
```

### Batch Processing for Large Datasets

```python
# Process in chunks
chunk_size = 50000
n_chunks = len(df) // chunk_size + 1

all_results = []

for i in range(n_chunks):
    start = i * chunk_size
    end = min((i + 1) * chunk_size, len(df))
    
    df_chunk = df.iloc[start:end]
    X_chunk = preprocessor.transform(df_chunk)
    
    results_chunk = detector.detect_and_correct(X_chunk)
    all_results.append(results_chunk)

# Combine results
X_corrected_full = np.vstack([r['corrected'] for r in all_results])
```

### Integration with Fuzzy Logic System

```python
# Prepare output for fuzzy credit scoring
output_package = {
    'features': X_corrected,
    'feature_names': preprocessor.feature_names,
    'anomaly_scores': results['scores'],
    'correction_confidence': results['confidence'],
    'original_target': y
}

# Save for fuzzy project
import pickle
with open('fuzzy_input_data.pkl', 'wb') as f:
    pickle.dump(output_package, f)

print("✓ Data ready for fuzzy credit scoring system")
```

---

## Troubleshooting

### Issue: "FileNotFoundError: application_train.csv"

**Solution:** Ensure data file is in correct location:
```bash
mkdir -p data
# Download and place application_train.csv in data/
```

### Issue: "ValueError: Input contains NaN"

**Solution:** Check preprocessing:
```python
# Debug
print(f"NaN in X: {np.isnan(X).any()}")
print(f"Features with NaN: {[f for f, has_nan in zip(features, np.isnan(X).any(axis=0)) if has_nan]}")

# Fix: Ensure proper preprocessing
preprocessor = FinancialPreprocessor()
X, y, features = preprocessor.fit_transform('data/application_train.csv')
assert not np.isnan(X).any(), "Still has NaN!"
```

### Issue: Low downstream improvement

**Possible causes:**
1. Not enough anomalies detected
2. Threshold too strict/loose
3. Correction not effective

**Debug:**
```python
# Check detection rate
print(f"Anomaly rate: {results['flags'].mean():.2%}")

# Try different threshold
percentiles = [90, 92, 94, 95, 96, 98]
for p in percentiles:
    threshold = np.percentile(results['scores'], p)
    flags = (results['scores'] > threshold).astype(int)
    print(f"Percentile {p}: {flags.sum()} anomalies")
```

### Issue: Training is slow

**Solutions:**
1. Reduce epochs: `epochs=30` instead of `50`
2. Increase batch size: `batch_size=1024` instead of `512`
3. Use GPU: Ensure TensorFlow sees GPU
4. Reduce training data: Sample 50% for faster iterations

```python
# Check GPU
import tensorflow as tf
print(f"GPU available: {tf.config.list_physical_devices('GPU')}")

# Fast training mode
detector.fit(X_normal, epochs=20, batch_size=1024, verbose=0)
```

---

## Tips & Best Practices

### 1. Always Monitor Training

```python
from src.visualization import plot_training_history

history = autoencoder.fit(X_normal, epochs=50, verbose=1)
plot_training_history(history)
```

### 2. Save Intermediate Results

```python
# Save after each major step
preprocessor.scaler  # Save this
detector.save('models/checkpoint')
np.save('data/X_preprocessed.npy', X)
```

### 3. Version Your Experiments

```python
import datetime

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = f'models/experiment_{timestamp}'

detector.save(save_dir)
evaluator.save_results(f'{save_dir}/metrics.json')
```

### 4. Document Hyperparameters

```python
config = {
    'ae_weight': 0.7,
    'if_weight': 0.3,
    'epochs': 50,
    'batch_size': 512,
    'latent_dim': 8,
    'threshold_percentile': 95
}

import json
with open('results/config.json', 'w') as f:
    json.dump(config, f, indent=2)
```

---

## Next Steps

After completing basic usage:

1. **Experiment with hyperparameters**
2. **Try custom architectures**
3. **Integrate with fuzzy logic system**
4. **Deploy as API** (see docs/DEPLOYMENT.md)
5. **Contribute improvements** (see CONTRIBUTING.md)

---

## Support

- **Documentation**: Check docs/ folder
- **Examples**: See notebooks/ folder
- **Issues**: GitHub Issues
- **Questions**: Open a discussion

Happy detecting! 🎯
