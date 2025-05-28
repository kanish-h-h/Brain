#### **1.1 System Preparation (45 mins)**

```python
# Create clean environment
mkdir plinko-par-system
cd plinko-par-system
python -m venv plinko-env
source plinko-env/bin/activate  # Linux/Mac

# Install core dependencies
pip install numpy pandas matplotlib ipython
```

#### **1.2 Project Structure (30 mins)**
```bash
/plinko-par-system
├── configs/
│   └── plinko_base.json
├── par_sheets/
├── simulations/
├── core/
│   ├── __init__.py
│   ├── par_generator.py
│   └── simulator.py
└── requirements.txt
```

#### **1.3 PAR Schema Definition (1.5 hrs)**

Create `configs/plinko_base.json`:

```json
{
  "game_id": "PLINKO_001",
  "board_config": {
    "rows": 8,
    "columns": 5,
    "peg_pattern": "diamond",
    "drop_zones": [
      {"zone": "A", "multiplier": 0.5, "target_prob": 0.05},
      {"zone": "B", "multiplier": 2, "target_prob": 0.25},
      {"zone": "C", "multiplier": 5, "target_prob": 0.40},
      {"zone": "D", "multiplier": 10, "target_prob": 0.25},
      {"zone": "E", "multiplier": 20, "target_prob": 0.05}
    ]
  },
  "rtp_target": 95.5,
  "max_multiplier": 20
}
```


#### **1.4 Basic Simulation Class (1 hr)**

Create `core/simulator.py`:

```python
import numpy as np
import json

class PlinkoSimulator:
    def __init__(self, config_path):
        with open(config_path) as f:
            self.config = json.load(f)
            
        self.board = self.config['board_config']
        self.drop_zones = self.board['drop_zones']
        
    def single_drop(self):
        """Simulate one chip drop"""
        position = self.board['columns'] // 2  # Start center
        for _ in range(self.board['rows']):
            position += np.random.choice([-1, 1])
            position = np.clip(position, 0, self.board['columns']-1)
        return self.drop_zones[position]
    
    def batch_simulate(self, num_drops=1000):
        """Run multiple simulations"""
        results = []
        for _ in range(num_drops):
            results.append(self.single_drop()['multiplier'])
        return {
            'rtp': np.mean(results),
            'max_win': max(results),
            'min_win': min(results)
        }
```

#### **Verification (15 mins)**

```python
# Quick test in Python console
from core.simulator import PlinkoSimulator
sim = PlinkoSimulator("configs/plinko_base.json")
result = sim.batch_simulate(1000)
print(f"RTP: {result['rtp']}%")  # Should be ~95-105% initially
```