[Tags::] #prompt #lisa


### **Project Planning Prompt Template**  

**Objective**:  
*"Design a phased technical roadmap for a [Project Name] that combines local development with cloud-based (Colab) heavy processing, ensuring compatibility with low-end systems. The project involves [briefly describe core functionality]. Provide granular phase breakdowns with strict directory structure, tool segregation, and verification protocols."*  

**Key Requirements**:  
1. **Environment Constraints**:  
   - Local machine: [Specify RAM/CPU/GPU, e.g., "4GB RAM, no GPU"]  
   - Cloud resources: [e.g., "Colab free tier"]  

2. **Core Technologies**:  
   - Local: [List, e.g., "FastAPI, SQLite"]  
   - Cloud: [List, e.g., "PyTorch, Tesseract OCR"]  

3. **Workflow Split**:  
   - Local tasks: [e.g., "UI, lightweight data validation"]  
   - Cloud tasks: [e.g., "LLM fine-tuning, OCR processing"]  

4. **Deliverables**:  
   - Directory tree with `.gitignore` rules  
   - Phase-wise technical breakdown (no code)  
   - Verification methodology for each phase  

5. **Constraints**:  
   - Free-tier tools only  
   - Minimal local resource usage  

**Output Format**:  
```markdown
1. **Phase 0: Setup**  
   - Directory structure  
   - Config files  
   - Environment prep  

2. **Phase X: [Name]**  
   - Objectives  
   - Local tasks  
   - Cloud tasks  
   - Dependencies  
   - Verification  

3. **Cross-Phase Concerns**  
   - Data flow  
   - Error handling  
   - CI/CD (if applicable)  
```

**Example Usage**:  
*"Design a phased technical roadmap for a 'Video Summarizer AI' that combines local development with Colab-based processing. Local machine: 8GB RAM/no GPU. Core tech: Local (FFmpeg, Whisper.cpp), Cloud (HuggingFace Transformers). Split: Local (video chunking), Cloud (summarization). Prioritize free-tier tools."*  

---

### **Why This Works**  
1. **Structured** – Forces clear phase segregation  
2. **Tool-Agnostic** – Adaptable to any tech stack  
3. **Constraint-Aware** – Explicit about resource limits  
4. **Verification-Focused** – Builds in testing from day one  

For your current project, you’d replace the example text with:  
*"Design a phased technical roadmap for an 'AI Book Mentor' combining local PDF processing with Colab-based model training. Local: 4GB RAM/no GPU. Core tech: Local (PyMuPDF, SQLite), Cloud (Mistral-7B, Tesseract). Split: Local (metadata handling), Cloud (OCR/LLM training)."*  

This ensures every project plan maintains consistency while accommodating unique requirements.