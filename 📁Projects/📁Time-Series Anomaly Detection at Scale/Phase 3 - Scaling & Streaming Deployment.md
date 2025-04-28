[Whole Plan::[[Plan (deepseek)]]]
[Phase 2::[[Phase 2 - Modelling & Training | Modelling & Training]]]
[Phase 4::[[Phase 4 - Evaluation & Optimization | Evaluation & Optimization]]]

---
  
**Objective**: *Deploy models for real-time anomaly detection with low latency and high throughput.*  

---

#### **Task 1: Streaming Pipeline Setup**  
**What to Do**:  
1. **Ingest Data via Kafka**:  
   - Set up Kafka producers to stream time-series data (e.g., from `data/processed/`).  
   - Use **Avro** schema for efficient serialization.  
2. **Process Streams with Flink**:  
   - Build Flink jobs for:  
     - Windowing (e.g., 5-minute tumbling windows).  
     - Feature engineering (e.g., rolling averages).  

**Tools**:  
- **Streaming**: Apache Kafka (producers/consumers), Apache Flink (DataStream API).  
- **Serialization**: Apache Avro, Confluent Schema Registry.  

**Roadblocks**:  
- High latency in windowed aggregations.  
- Schema mismatches breaking the pipeline.  

**Debugging**:  
- Check Kafka consumer lag with `kafka-consumer-groups` CLI.  
- Validate schemas with `kafka-avro-console-consumer`.  

**Example Prompt**:  
*“Create a Flink job to compute 5-minute rolling averages of sensor data from Kafka topic ‘raw-sensors’.”*  

---

#### **Task 2: Model Serving in Real-Time**  
**What to Do**:  
1. **Deploy Models as APIs**:  
   - Serve models (e.g., LSTM) via **FastAPI** or **TensorFlow Serving**.  
2. **Integrate with Flink**:  
   - Call model APIs from Flink jobs for inference (e.g., using `AsyncIO`).  

**Tools**:  
- **Serving**: FastAPI, TensorFlow Serving, ONNX Runtime.  
- **Optimization**: Model pruning/quantization with `TensorFlow Lite` or `OpenVINO`.  

**Roadblocks**:  
- High inference latency (>100ms).  
- Model versioning conflicts.  

**Debugging**:  
- Profile latency with `Py-Spy` or `cProfile`.  
- Use a model registry (MLflow) to track versions.  

**Learn**:  
- [Low-Latency Model Serving](https://www.tensorflow.org/tfx/guide/serving)  

---

#### **Task 3: State Management & Checkpointing**  
**What to Do**:  
1. **Handle Stateful Operations**:  
   - Track anomaly counts or thresholds in Flink state (e.g., `ValueState`).  
2. **Enable Checkpointing**:  
   - Configure Flink to save state to **S3** or **HDFS** for fault tolerance.  

**Tools**:  
- **State**: Flink’s `StateBackend` (RocksDB for large state).  
- **Storage**: S3, MinIO, HDFS.  

**Roadblocks**:  
- State recovery failures after crashes.  
- Growing state size slowing down jobs.  

**Debugging**:  
- Test state recovery with manual savepoints.  
- Use TTL (time-to-live) for stale state.  

**Example Prompt**:  
*“Configure Flink to checkpoint state every 10 minutes to S3 for the ‘anomaly-counts’ job.”*  

---

#### **Task 4: Monitoring & Alerting**  
**What to Do**:  
1. **Track Metrics**:  
   - Monitor throughput (events/sec), latency (p95), and anomaly rates.  
2. **Set Alerts**:  
   - Trigger alerts for SLA breaches (e.g., latency > 1s).  

**Tools**:  
- **Monitoring**: Prometheus + Grafana, Flink’s built-in metrics.  
- **Alerting**: Alertmanager, PagerDuty.  

**Roadblocks**:  
- Metrics overload (too many dashboards).  
- False alarms due to seasonal spikes.  

**Debugging**:  
- Use Prometheus’s `rate()` to filter outliers.  
- Correlate alerts with business KPIs (e.g., user traffic).  

**Learn**:  
- [Flink Metrics Guide](https://nightlies.apache.org/flink/flink-docs-release-1.17/docs/ops/metrics/)  

---

### **Phase 3 Success Criteria**  
Proceed to Phase 4 if:  
✅ End-to-end latency < 100ms (from Kafka ingestion to anomaly alert).  
✅ Flink jobs process 10K+ events/sec without backpressure.  
✅ Model APIs return inferences with >95% uptime.  

---

### **Phase 3 Roadmap Summary**  
| Task                   | Tools               | Key Outputs                          |  
|------------------------|---------------------|--------------------------------------|  
| Streaming Pipeline     | Kafka, Flink        | Real-time data ingestion & windowing |  
| Model Serving          | FastAPI, ONNX       | Low-latency inference API endpoints  |  
| State Management       | Flink StateBackend  | Checkpointed state in S3/HDFS        |  
| Monitoring             | Prometheus, Grafana | Dashboards for throughput/latency    |  

---

### **What to Learn Next**  
- **Flink Internals**: [Stateful Stream Processing](https://www.ververica.com/blog/apache-flink-stream-processing)  
- **Kafka Tuning**: [Optimizing Kafka for Latency](https://www.confluent.io/blog/configure-kafka-to-minimize-latency/)  

If stuck, ask for hints on:  
- Diagnosing Kafka consumer lag.  
- Reducing Flink checkpointing overhead.