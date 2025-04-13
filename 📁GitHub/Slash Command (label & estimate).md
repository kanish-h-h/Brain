Here's a **detailed breakdown** of GitHub's slash commands like `/label` and `/estimate`, including how they work, why they're useful, and pro tips for using them effectively:

---

### **1. `/label` Command**
#### **What It Does:**
- Adds or removes labels from GitHub issues or pull requests directly in comments.  
- **Example:**  
  ```markdown
  /label ~bug ~high-priority
  ```
  Adds labels "bug" and "high-priority" to the issue.

#### **Key Features:**
| Command                  | Action                                 | Example                          |
|--------------------------|----------------------------------------|----------------------------------|
| `/label ~bug`            | Adds "bug" label                       | `/label ~bug ~MLOps`             |
| `/label -~documentation` | Removes "documentation" label          | `/label -~wontfix`               |
| `/label ~"good first"`   | Handles multi-word labels (use quotes) | `/label ~"help wanted"`          |

#### **Why Use It?**
- **Saves time**: No need to manually click labels in the UI.  
- **Collaboration**: Team members can triage issues via comments.  
- **Automation**: Works with GitHub Actions/bots (e.g., auto-label PRs).

#### **Pro Tips:**
1. **Combine with other commands**:  
   ```markdown
   /label ~bug ~high-priority
   /assign @yourusername
   ```
2. **Create custom labels** for your workflow:  
   - `~needs-repro` (for unverified bugs)  
   - `~blocked` (waiting on external dependencies)  

---

### **2. `/estimate 3d` Command**
#### **What It Does:**
- Sets time estimates for issues/PRs (often used with **GitHub Projects** or Agile tools like ZenHub).  
- **Example:**  
  ```markdown
  /estimate 3d  # Estimates 3 days of work
  ```

#### **Key Features:**
| Command          | Action                              | Notes                               |
|------------------|-------------------------------------|-------------------------------------|
| `/estimate 2h`   | Sets estimate to 2 hours            | Works with `h`, `d`, `w` units.     |
| `/estimate clear`| Removes the estimate                |                                     |
| `/estimate 1w`   | Sets estimate to 1 week             | Useful for epics.                   |

#### **Why Use It?**
- **Agile planning**: Helps prioritize work in sprints.  
- **Progress tracking**: Visualize timelines in GitHub Projects.  
- **Retrospectives**: Compare estimates vs. actual time spent.

#### **Pro Tips:**
1. **Use with Projects**:  
   - Estimates appear in your project board columns.  
   - Filter issues by time estimates (e.g., `label:bug estimate:<=1d`).  
2. **Standardize units**:  
   - Stick to `h` (hours) or `d` (days) for consistency.  
3. **Pair with milestones**:  
   ```markdown
   /estimate 2d
   /milestone Sprint 5
   ```

---

### **3. Other Useful Slash Commands**
| Command               | Action                                   | Example                          |
|-----------------------|------------------------------------------|----------------------------------|
| `/assign @username`   | Assigns an issue/PR to a user            | `/assign @yourname`              |
| `/milestone`          | Links to a milestone                     | `/milestone MVP`                 |
| `/close`              | Closes the issue/PR                      | `/close Fixes #42`               |
| `/retitle`            | Edits the title                          | `/retitle Fix GPU OOM error`     |
| `/reviewer`           | Requests a review                        | `/reviewer @teammate`            |

---

### **4. How to Enable These Commands**
1. **Native Support**:  
   - Works out-of-the-box in GitHub issues/PRs.  
2. **For Advanced Features**:  
   - Tools like **ZenHub** or **Linear** add more commands (e.g., `/priority high`).  
3. **Custom Bots**:  
   - Use GitHub Actions to create your own slash commands (e.g., `/deploy staging`).

---

### **5. Real-World Example Workflow**
**Scenario**: You find a bug in your ML pipeline.  
1. **Create an issue**:  
   ```markdown
   ## BUG: Training crashes on empty batches
   Steps to reproduce: [details...]
   ```
2. **Triage it**:  
   ```markdown
   /label ~bug ~high-priority
   /estimate 1d
   /assign @yourname
   ```
3. **After fixing**:  
   ```markdown
   /close Fixed in #45
   ```

---

### **6. Pro Tips for Power Users**
1. **Keyboard Shortcuts**:  
   - Press `Ctrl + /` (Cmd + / on Mac) in GitHub comments to see available commands.  
2. **Aliases**:  
   - Some tools let you create shortcuts (e.g., `/est 3d` → `/estimate 3d`).  
3. **Automation**:  
   - Auto-label issues when estimates are added:  
     ```yaml
     # GitHub Actions snippet
     - name: Label estimated issues
       if: contains(github.event.comment.body, '/estimate')
       run: gh issue edit $ISSUE_NUMBER --add-label "estimated"
     ```

---

### **7. Limitations**
- **No native time tracking**: Estimates are manual (use tools like **Clockify** for actuals).  
- **Project boards only**: Estimates won’t appear in regular issue lists.  

---

By mastering these commands, you’ll **streamline collaboration** and **make your workflow look pro**. Now go label some issues! 🏷️⏱️