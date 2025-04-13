#prompt 
If you want me to **guide you through projects without giving direct code**, but instead provide **structured approaches, debugging tips, and high-level roadmaps**, use this prompt:

> **"I'm working on a [specific ML project, e.g., 'Real-Time Anomaly Detection System']. I need a step-by-step roadmap to build this project from scratch. Do NOT provide full code—instead, break it into phases, suggest key algorithms/tools, highlight potential roadblocks, and offer debugging strategies. If I get stuck, I’ll ask for hints. Start by outlining the project’s structure."**

**Example Interaction:**

- **You:** _"I’m building a Federated Learning system for medical data. Give me a step-by-step plan."_
    
- **Me:**
    
    1. **Phase 1 (Data Setup)** – Use PySyft to simulate hospitals with partitioned COVID-19 X-ray datasets.
        
    2. **Phase 2 (Modeling)** – Implement FedAvg with differential privacy (use Opacus).
        
    3. **Phase 3 (Debugging)** – If gradients vanish, check clipping thresholds.
        
    4. **Phase 4 (Evaluation)** – Compare federated vs. centralized model AUC-ROC.