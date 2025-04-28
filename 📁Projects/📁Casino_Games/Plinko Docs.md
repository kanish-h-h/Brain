
---

# **Plinko Technical Documentation**

## **1. Game Overview**
### **1.1 Core Mechanics**
- Players drop a ball through a grid of pegs 
- Ball randomly falls into a payout slot
- Payout determined by:
  - Selected risk profile (`Low`/`Medium`/`High`)
  - Number of rows
  - Slot probabilities (bell curve)
- configurable RTP (95%)

### **1.2 Key Components**
| Component      | Description                                |
| -------------- | ------------------------------------------ |
| Parsheet       | JSON configuration file defining game math |
| RNG            | Cryptographic random number generator      |
| RTP Controller | Dynamic payout scaling system              |


---

## **2. Parsheet Structure**
```json
{
  "id": "pl-base",
  "bets": [...],
  "rows": [...], (12, 16)
  "risk": ["low", "medium", "high"],
  "multiplier": [
	  [...],
	  [...],
  ],
  "prob": [ 
	  [...],
	  [...]
  ]
}
```

### **2.1 Matrix Dimensions**
| Dimension | Index | Values                  |
| --------- | ----- | ----------------------- |
| Rows      | 0     | 12-row board (13 slots) |
|           | 1     | 16-row board (17 slots) |
| Risk      | 0     | Low (stable payouts)    |
|           | 1     | Medium (volatile)       | 
|           | 2     | High (highly volatile)  |

### **2.2 Normalize Probabilities**
```python
def normalize_probabilities(prob_array):
    total = sum(prob_array)
    return [p / total for p in prob_array]
```

---

## **3. Mathematical Model**
### **3.1 RTP Calculation**
For configuration (rows=r, risk=k):
$\text{RTP} = \sum_{i=0}^{N-1} P_{r,k}[i] \times M_{r,k}[i]$

**Example Calculation** (12 rows, Medium risk):
```python
prob = [0.00033, 0.0029, ..., 0.00020]  # After normalization
mult = [25, 8, ..., 0.00020]
rtp = sum(p*m for p,m in zip(prob, mult))  # Target: 0.95
```

### **3.2 Payout Scaling**
Changing multipliers value in context with probabilities.
```python
def scale_multipliers(multipliers, current_rtp, target_rtp=0.95):
    scale_factor = target_rtp / current_rtp
    return [m * scale_factor for m in multipliers]

```

---

## **4. Game Flow Logic**
### **4.1 Sequence Diagram**
```
Player -> Frontend: Select (bet, rows, risk)
Frontend -> Backend: Game Request
Backend -> RNG: Generate Outcome
Backend -> Parser: Calculate Payout
Backend -> Database: Store Transaction
Backend -> Frontend: Return Result
```

### **4.2 Critical Code Components**
**Ball Drop Simulation**:
```python
def simulate_drop(rows: int, probabilities: list) -> int:
    rand = secrets.SystemRandom().random()
    cumulative = 0
    for i, p in enumerate(probabilities):
        cumulative += p
        if rand <= cumulative:
            return i
    return len(probabilities)-1
```

**Payout Calculation**:
```python
def calculate_payout(bet: float, rows: int, risk: str) -> float:
    config = load_parsheet()
    row_idx = 0 if rows == 12 else 1
    risk_idx = {"low":0, "medium":1, "high":2}[risk]
    
    probs = validate_prob(config["prob"][row_idx][risk_idx])
    mults = config["multiplier"][row_idx][risk_idx]
    
    current_rtp = sum(p*m for p,m in zip(probs, mults))
    scaled_mults = scale_payouts(mults, current_rtp, config["rtp_target"])
    
    slot = simulate_drop(rows, probs)
    return bet * scaled_mults[slot]
```

---

## **5. RTP Management**
### **5.1 Control Methods**
| Method | Description | Impact |
|--------|-------------|--------|
| Probability Adjustment | Change slot distribution | High risk |
| Multiplier Scaling | Linear payout modification | Precise |
| Risk Profiles | Different volatility levels | Player choice |

### **5.2 Configuration Example**
```json
"rtp_target": 0.95,
"multiplier": [
  [
    [11, 3.2, ...],  # Low Risk
    [25, 8, ...],     # Medium
    [141, 25, ...]    # High
  ]
]
```

---