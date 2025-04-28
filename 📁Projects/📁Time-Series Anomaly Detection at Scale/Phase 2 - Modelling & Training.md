[Whole Plan::[[Plan (deepseek)]]]
[Phase 1::[[Phase 1 - Data Preparation & Ingestion | Data Preparation & Ingestion]]]
[Phase 3::[[Phase 3 - Scaling & Streaming Deployment | Scaling & Streaming Deployment]]]

---

**Objective**: *Build and optimize scalable anomaly detection models for time-series data.*  

---

#### **Task 1: Algorithm Selection & Baseline Models**  
**What to Do**:  
1. **Benchmark Algorithms**:  
   - Test NAB-recommended models:  
     - **Hierarchical Temporal Memory (HTM)** via `nupic` (for streaming compatibility).  
     - **Isolation Forest** (scikit-learn) for unsupervised detection.  
     - **LSTM Autoencoder** (PyTorch/TensorFlow) for deep learning.  
   - Evaluate baselines like **Prophet** (Facebook) or **SARIMA**.  
2. **Define Metrics**:  
   - Use NAB’s scoring (prioritizes early detection) with **F1-score**, **precision**, **recall**.  

**Tools**:  
- `nupic`, `scikit-learn`, `PyTorch`, `sktime` (time-series ML).  
- **Distributed Training**: `Dask-ML` (parallelize scikit-learn), `Horovod` (PyTorch/TensorFlow).  

**Roadblocks**:  
- HTM’s steep learning curve.  
- LSTM training instability (vanishing gradients).  

**Debugging**:  
- Start with Isolation Forest for quick iteration.  
- For LSTMs: Use gradient clipping, layer normalization.  

**Example Prompt**:  
*“Compare F1-scores of HTM vs. LSTM on the NAB ‘realTraffic’ dataset.”*  

---

#### **Task 2: Scalable Training Pipeline**  
**What to Do**:  
1. **Distribute Training**:  
   - Use `Dask-ML` to parallelize scikit-learn models across clusters.  
   - For deep learning: Use `Horovod` with PyTorch or `TensorFlow Distributed Strategy`.  
2. **Windowed Data**:  
   - Split time-series into sliding windows (e.g., 1-hour windows with 5-minute steps).  

**Tools**:  
- **Small Data**: `scikit-learn` (single-node).  
- **Large Data**: `Dask-ML`, `PySpark MLlib`, or `Ray Train`.  
- **Windowing**: `tsfresh` for feature extraction, `sktime` for splits.  

**Roadblocks**:  
- High memory usage with large windows.  
- Slow distributed training setup.  

**Debugging**:  
- Reduce window size or stride.  
- Profile with `dask.diagnostics.ProgressBar` or `PySpark UI`.  

**Learn**:  
- [Distributed Training with Dask-ML](https://ml.dask.org/)  
- [Time-Series Windowing Best Practices](https://towardsdatascience.com/using-sliding-windows-to-generate-training-data-for-time-series-9d4e9e34e5d5)  

---

#### **Task 3: Hyperparameter Tuning**  
**What to Do**:  
1. **Define Search Space**:  
   - For HTM: `activationThreshold`, `initialPermanence`.  
   - For LSTM: `hidden_units`, `learning_rate`, `dropout`.  
2. **Optimize at Scale**:  
   - Use distributed hyperparameter tuning frameworks.  

**Tools**:  
- **Single Node**: `GridSearchCV` (scikit-learn), `Optuna`.  
- **Distributed**: `Ray Tune`, `Spark Trials` (MLflow + Spark).  

**Roadblocks**:  
- Tuning takes days to converge.  
- Overfitting to validation splits.  

**Debugging**:  
- Use Bayesian optimization (Optuna) instead of grid search.  
- Apply time-series cross-validation (`TimeSeriesSplit`).  

**Example Prompt**:  
*“Run a hyperparameter sweep for LSTM hidden_units (32, 64, 128) and dropout (0.1, 0.2).”*  

---

#### **Task 4: Model Validation & Thresholding**  
**What to Do**:  
1. **Dynamic Thresholding**:  
   - Set anomaly thresholds using percentile-based methods (e.g., 99th percentile of reconstruction error).  
2. **Validate on NAB Metrics**:  
   - Use the official NAB scoring script to evaluate detection latency and false positives.  

**Tools**:  
- Thresholding: `scipy.stats.scoreatpercentile`, `tsmoothie`.  
- NAB Evaluation: Clone and adapt the NAB `run.py` script.  

**Roadblocks**:  
- Thresholds too sensitive (high false positives).  
- NAB scoring penalizes late detections.  

**Debugging**:  
- Adjust thresholds based on domain needs (e.g., lower recall for critical systems).  
- Use `MLflow` to track threshold-impacted metrics.  

**Learn**:  
- [NAB Scoring Methodology](https://arxiv.org/abs/1510.03336)  

---

#### **Task 5: Model Serialization & Tracking**  
**What to Do**:  
1. **Export Models**:  
   - Save trained models to `models/trained/` (e.g., HTM as JSON, LSTM as `.pt` or `.h5`).  
2. **Track Experiments**:  
   - Log hyperparameters, metrics, and artifacts with `MLflow` or `Weights & Biases`.  

**Tools**:  
- Serialization: `pickle` (scikit-learn), `torch.save()` (PyTorch), `tf.saved_model` (TensorFlow).  
- Tracking: `MLflow`, `DVC`.  

**Roadblocks**:  
- Model format incompatibility (e.g., PyTorch ↔ TensorFlow).  
- Experiment metadata not reproducible.  

**Debugging**:  
- Use `ONNX` for cross-framework model portability.  
- Version control `configs/model_config.yaml` with Git.  

**Example Prompt**:  
*“Log LSTM training metrics (loss, F1-score) to MLflow with a ‘noisy_synthetic’ experiment tag.”*  

---

### **Phase 2 Success Criteria**  
Proceed to Phase 3 if:  
✅ Models are validated using NAB scoring (F1-score > baseline).  
✅ Best hyperparameters are logged in `configs/model_config.yaml`.  
✅ All models are serialized to `models/trained/` with metadata.  

---

### **Phase 2 Roadmap Summary**  
| Task                   | Tools               | Key Outputs                          |  
|------------------------|---------------------|--------------------------------------|  
| Algorithm Selection    | nupic, PyTorch      | Baseline model performance reports   |  
| Distributed Training   | Dask-ML, Ray Train  | Trained models (HTM, LSTM, etc.)     |  
| Hyperparameter Tuning  | Optuna, Ray Tune    | Optimized model parameters           |  
| Validation & Thresholding | NAB, scipy      | Anomaly thresholds and F1-scores     |  
| Model Tracking          | MLflow              | Versioned models and metrics         |  

---

### **What to Learn Next**  
- **HTM Theory**: [Numenta’s HTM School](https://numenta.com/htm-school/)  
- **LSTM for Anomalies**: [LSTM Autoencoder Guide](https://towardsdatascience.com/lstm-autoencoder-for-anomaly-detection-e1f4f2ee7ccf)  

If stuck, ask for hints on:  
- Reducing LSTM overfitting with dropout.  
- Debugging silent failures in distributed training.