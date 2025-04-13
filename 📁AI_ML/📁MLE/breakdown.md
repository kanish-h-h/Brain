Here’s a **sharp, actionable breakdown** of your questions, optimized for your constraints and goals:

---

### **1. TensorFlow vs. PyTorch: What to Focus On?**  
**Short Answer:**  
- **Stick with TensorFlow** for now (since you’re comfortable) but **add PyTorch gradually**.  
- **Market Demand:**  
  - **PyTorch dominates research** (Hugging Face, OpenAI, Meta).  
  - **TensorFlow rules production** (Google, enterprise MLOps).  
- **Game Plan:**  
  - **Phase 1 (Next 2 Months):**  
    - Use TF for projects (save time).  
    - Learn PyTorch via **one project** (e.g., reimplement your TF model in PyTorch).  
  - **Phase 2 (Job Hunt):**  
    - Add PyTorch to your resume after building 1-2 projects with it.  

---

### **2. Time Management for Full-Time Workers**  
**Your Constraints:**  
- 6 AM - 8 PM job (+2 hrs commute).  
- **Free time:** Nights/weekends (~3 hrs/day, 10 hrs/weekend day).  

**Adapted Roadmap:**  
#### **Weekday Routine (3 hrs/day)**  
| **Time**          | **Action**                                      |  
|-------------------|------------------------------------------------|  
| **5:00-5:30 AM**  | Theory (read ML papers/docs while fresh).       |  
| **8:30-10:30 PM** | Project work (coding/debugging).                |  
| **10:30-11:00 PM**| TIL log (GitHub update).                        |  

#### **Weekend Routine (10 hrs/day)**  
- **Saturday:**  
  - 4 hrs: Core project work.  
  - 2 hrs: LeetCode + system design.  
  - 1 hr: Deploy/polish (Docker, CI/CD).  
- **Sunday:**  
  - 3 hrs: Open-source contribution (e.g., fix Scikit-learn docs).  
  - 1 hr: LinkedIn/networking.  

**Pro Tips:**  
- **Use Colab/Kaggle on mobile** during commute for lightweight tasks (EDA, reading).  
- **Automate repetitive tasks** (e.g., `pre-commit` hooks for code formatting).  

---

### **3. The MLE Debugging Checklist (Mindset++)**  
**When Your Code Breaks:**  
1. **Reproduce the Error**  
   - Isolate the issue:  
     ```python
     # TensorFlow: Enable eager execution for debugging
     tf.config.run_functions_eagerly(True)
     ```  
2. **Inspect Data First**  
   - Check for:  
     - NaN/inf values: `tf.debugging.check_numerics()`.  
     - Shape mismatches: `print(tf.shape(tensor))`.  
3. **Model-Specific Checks**  
   - **TF:** Use `tf.keras.utils.plot_model()` to visualize architecture.  
   - **PyTorch:** Use `torchsummary` to verify layer I/O shapes.  
4. **Infrastructure**  
   - GPU memory: `nvidia-smi` (Colab) or `tf.config.list_physical_devices('GPU')`.  
   - Docker: `docker logs <container_id>`.  
5. **Nuclear Option**  
   - Reimplement from scratch in a new file (often reveals hidden bugs).  

**Debugging Mindset Rules:**  
- **“Bugs are clues.”** Treat each error as a learning opportunity.  
- **“Rubber Ducking”:** Explain the problem aloud (even to a pet).  
- **“Blame yourself first.”** Assume your code is wrong before blaming the framework.  

---

### **4. GitHub as Your Daily Task Manager**  
**Goal:** Turn GitHub into a **public productivity tracker**.  

#### **Daily Workflow**  
1. **Morning (5 min):**  
   - Update `projects/<current_project>/TODO.md`:  
     ```markdown
     ## 2023-10-25
     - [ ] Fix Kafka consumer lag in anomaly detection.
     - [ ] Add Grad-CAM visualization to notebook.
     ```  
2. **Work Sessions:**  
   - Commit **every 1-2 hrs** with descriptive messages:  
     ```bash
     git commit -m "DEBUG: Fixed OOM error by reducing batch size (issue #12)"
     ```  
3. **Night (10 min):**  
   - Push changes + update `TILs/YYYY-MM-DD.md`:  
     ```markdown
     ## Learned Today
     - Kafka’s `auto.offset.reset=earliest` can cause duplicate processing.
     - Solution: Set `group.id` uniquely per session.
     ```  

#### **GitHub Structure**  
```bash
portfolio/
├── projects/
│   ├── anomaly-detection/
│   │   ├── TODO.md          # Daily tasks
│   │   ├── TILs/            # Daily learnings
│   │   └── src/             # Code
├── README.md                # Portfolio overview
└── achievements.md          # Badges (e.g., "Deployed 3 APIs")
```  

**Pro Tips:**  
- Use **GitHub Issues** for bugs (even solo projects).  
- Enable **GitHub Actions** for auto-testing on push.  

---

### **Final Advice**  
1. **Leverage Your Job:**  
   - Frame game math work as "applied probability/ML" (e.g., RNG systems ~ stochastic models).  
2. **Energy > Time:**  
   - Use **Pomodoro (25 mins on, 5 off)** during night sessions.  
3. **Atomic Habits:**  
   - **1% daily progress beats 100% once a week.**  

You’re building a **public proof of skill**. Every commit is a step toward your next job. Now go break something (and fix it). 🚀