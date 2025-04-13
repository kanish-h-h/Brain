
---
## Prompts

### For Whole Plan
`"I’m starting Project 1: Time-Series Anomaly Detection at Scale using the Numenta Anomaly Benchmark. I need a step-by-step roadmap to build this project from scratch. Do NOT provide code—instead: Break the project into phases (e.g., data prep, modeling, deployment). Suggest specific tools (Dask/PySpark for 10M+ rows, Kafka/Flink for streaming). Highlight likely roadblocks (e.g., memory issues, latency bottlenecks). Offer debugging strategies for those roadblocks. I’ll code everything myself. If I get stuck, I’ll ask for a hint. Start with Phase 1."`

### For Each Phase
```
"I’m working on **Project 1: Time-Series Anomaly Detection at Scale using Numenta Anomaly Benchmark**. For **[Phase X: e.g., Modeling & Training]**:

1. Break this phase into **specific tasks** (e.g., algorithm selection, distributed training).
    
2. Suggest **tools/frameworks** optimized for scale (e.g., Dask/Spark for data, Flink/Kafka for streaming).
    
3. Highlight **likely roadblocks** (e.g., latency, overfitting, state management).
    
4. Provide **debugging strategies** for those roadblocks.  
    Do NOT provide code—focus on architecture, tools, and troubleshooting. If I get stuck, I’ll ask for hints."
```

---

Here’s your step-by-step roadmap, structured to address scale, tooling, and common pitfalls:

---

### **Phase 1: Data Preparation & Ingestion**
**Objective**: Acquire, clean, and structure time-series data for anomaly detection at scale.

1. **Data Sourcing**  
   - Download the [Numenta Anomaly Benchmark (NAB) dataset](https://github.com/numenta/NAB).  
   - For scale testing (>10M rows), generate synthetic time-series data with anomalies using tools like `tsmoothie` or `synthetic_data` (Python).  

2. **Data Formatting**  
   - Ensure timestamps are parsed correctly (use `pandas`/`dask` for timezone handling).  
   - Validate numeric values (e.g., no NaN/infinity in critical fields).  

3. **Scalable Storage**  
   - Use **Parquet** (columnar storage) for large datasets.  
   - Tools: **Dask** (for out-of-core processing) or **PySpark** (for distributed clusters).  

4. **Roadblocks & Debugging**:  
   - **Memory issues with large files**:  
     - Debug: Process data in chunks (e.g., `pandas.read_csv(chunksize=1e6)`).  
     - Tool: Switch to Dask for lazy evaluation.  
   - **Inconsistent timestamps**:  
     - Debug: Use `pandas.to_datetime()` with `utc=True` and `errors="coerce"`.  
   - **Missing synthetic anomalies**:  
     - Debug: Inject anomalies using statistical methods (e.g., spike injection, trend shifts).  

---

### **Phase 2: Modeling & Training**
**Objective**: Build and train anomaly detection models optimized for time-series data.

1. **Algorithm Selection**  
   - Start with NAB-recommended models:  
     - **Hierarchical Temporal Memory (HTM)** (via `nupic`).  
     - Alternatives: **Isolation Forest**, **LSTM Autoencoders**, or **Prophet** (Facebook).  

2. **Scalable Training**  
   - For large datasets:  
     - Use **Dask-ML** or **PySpark MLlib** for distributed training.  
     - For streaming: **Apache Flink ML** (stateful processing).  

3. **Hyperparameter Tuning**  
   - Tools: **Ray Tune** (distributed hyperparameter optimization) or **Optuna**.  

4. **Roadblocks & Debugging**:  
   - **High training latency**:  
     - Debug: Profile with `cProfile` or `PySpark UI`; reduce feature dimensions.  
   - **Poor anomaly recall**:  
     - Debug: Adjust detection thresholds or use NAB’s scoring mechanism.  
   - **Model overfitting**:  
     - Debug: Use time-series cross-validation (e.g., `TimeSeriesSplit` in sklearn).  

---

### **Phase 3: Scaling & Streaming**  
**Objective**: Deploy the model for real-time/streaming inference.  

1. **Streaming Pipeline**  
   - Ingest data via **Apache Kafka** (high-throughput messaging).  
   - Process streams with **Apache Flink** (low-latency windowed computations).  

2. **Model Serving**  
   - Deploy models as APIs with **FastAPI** or **TensorFlow Serving**.  
   - For low-latency inference: Use **ONNX Runtime** or **TorchScript**.  

3. **Roadblocks & Debugging**:  
   - **Latency spikes in streaming**:  
     - Debug: Check parallelism in Flink/Kafka; optimize batch sizes.  
   - **State management in streaming**:  
     - Debug: Use Flink’s checkpointing or Kafka’s log-based retention.  
   - **Model cold starts**:  
     - Debug: Pre-warm models with historical data.  

---

### **Phase 4: Evaluation & Optimization**  
**Objective**: Validate performance and optimize for production.  

1. **NAB Scoring**  
   - Use NAB’s scoring system (weights early detection).  
   - Metrics: **F1-score**, **precision**, **recall**, and **time-to-detect**.  

2. **Optimization**  
   - Prune models with **TensorFlow Model Optimization Toolkit**.  
   - Quantize models for edge deployment (e.g., **TFLite**).  

3. **Roadblocks & Debugging**:  
   - **Score vs. real-world mismatch**:  
     - Debug: Augment NAB data with domain-specific anomalies.  
   - **High false positives**:  
     - Debug: Adjust anomaly thresholds dynamically.  

---

### **Phase 5: Deployment & Monitoring**  
**Objective**: Ship the system and ensure reliability.  

1. **Containerization**  
   - Use **Docker** for environment consistency.  
   - Orchestrate with **Kubernetes** (auto-scaling).  

2. **Monitoring**  
   - Track latency/accuracy with **Prometheus** + **Grafana**.  
   - Log anomalies with **ELK Stack** (Elasticsearch, Logstash, Kibana).  

3. **Roadblocks & Debugging**:  
   - **Deployment crashes**:  
     - Debug: Check resource limits (CPU/memory) in Kubernetes.  
   - **Model drift**:  
     - Debug: Retrain models periodically; use **MLflow** for tracking.  

---

### **Critical Tools Recap**:  
- **Data Prep**: Dask, PySpark, Parquet  
- **Modeling**: Nupic (HTM), PyTorch, Ray Tune  
- **Streaming**: Kafka, Flink  
- **Deployment**: FastAPI, Docker, Kubernetes  

**Start with Phase 1**, and if you hit a snag (e.g., Dask memory errors), ask for a hint!