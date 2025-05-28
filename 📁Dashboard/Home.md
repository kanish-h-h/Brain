---

## cssclasses: home  
banner: "![[1025a70dbc96689d04d6bc6552f683bf~2.jpg]]"  
banner_y: 0.46  
banner_x: 0.60  

## 🎯 Today

-  Review 1‑Month Goal
    
-  Push Skin‑sight Phase 2
    
-  Tidy inbox
    

---

## 🗄️ Recent Files

```dataview
list
from ""
sort file.mtime desc
limit 15
```

---

## 🚀 Active Projects

```dataview
table file.mtime as "Updated"
from "📁Projects"
where contains(file.tags,"active")
sort file.mtime desc
```

---

### Dashboard

- [[🏚️ AIML]]
    

### Builder’s Crucible

- [[The Builders’ Crucible (12 Projects That Force You to Master ML Engineering) | 12 Projects]]
    
- [[Optimzation for Constriants | Opt]]
    
- [[Approach Roadmap | Roadmap]]
    
- [[📁AI_ML/📁MLE/breakdown]]
    
- [[The Ultimate GitHub Mastery Guide for Machine Learning Engineers | MLE + GitHub]]
    

### Skills & Study

- [[1 - Month Goal]]
    
- [[Python + ML + Maths]]
    
- [[Learn Machine Learning]]
    

---

### Assignments

- [[📁AI_ML/Prerequisites]]
    
- [[numpy-exercises]] · [[pandas-exercises]] · [[matplotlib-exercises]]
    

### Python Cheatsheets

- [[Python]]
    

### Docker / DevOps

- [[docker_mysql]]
    

### PyTorch Quick Ref

- [[0. Fundamentals]] · [[1. Workflow-Fundamentals]] · [[2. Neural-Network-Classification]]
    

### SQL

- [[Introduction to SQL]] · [[Intermediate SQL]]
    

### Data Viz

- [[NUMpy]] · [[PANDAS]] · [[MATPLOTLIB]]
    

---

- [[Statistics Fundamentals]]
    
- [[Normal Distribution]] · [[log-normal Distribution]] · [[Gamma Distribution]] · [[Beta Distribution]] · [[Triangular Distribution]]
    
- [[Bayes Theorem]]
    
- [[Visualizing and Summarizing Data]]
    

---

#### PAR Sheets

- [[PAR Sheets]] · [[Button + Features]] · [[par sheet analysis]]
    

#### Randomness

- [[Random]] · [[secrets]] · [[PseudRandom Number Generators]] · [[Random Seeding]]
    
- Crypto‑Random: [[Cryptography Random]]
    

---

- **Ideas:** [[Projects Ideas]]
    
- **PRNG GAN:** [[1. Understanding the problem]] · [[2. Designing Generator and Discriminator Models]] · [[3. Defining Loss Functions and Optimizers]] · [[4. Training The GAN]]
    
- **Cholera:** [[Cholera - Documentation]] · [[Cholera 1]] · [[Cholera 2]] · [[Cholera 3]]
    
- **ACCLIP:** [[ACCLIP - Documentation]] · [[ACCLIP mission]] · [[ACCLIP mission Details]]
    
- **Daymet:** [[Daymet - Documentation]]
    
- **Mobile Apps:** [[Environment-Setup]]
    

---

- [[ML master prompt]]
    
- [[The Senior MLE Gauntlet Prompt]]
    

---

## 🔖 Favourites

```dataview
list
from ""
where contains(file.tags,"favourite")
sort file.name
```

---

## 📊 Vault Stats

```dataviewjs
const pages = dv.pages();
dv.paragraph(`**File Count:** ${pages.length}`);
dv.paragraph(`**Python Content:** ${pages.where(p=>p.file.name.toLowerCase().includes('python')).length}`);
```

---

%%homepage-button  
name: ➕ New Daily Note  
type: command  
id: daily-notes:open-daily-note  
%%