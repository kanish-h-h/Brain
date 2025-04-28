#Beginner 

---

# Scenario
An insurance company determines vehicle insurance premiums based on known risk factors. If a person is considered a higher risk, their premiums will be higher. One potential factor is the color of your car. The insurance company believes that people with same color cars are more likely to get in accidents. To research this, they examine police reports for recent total-loss collisions. The data is summarized in the frequency table below.

| **Color** | **Frequency** |
|:---------:|:-------------:|
|   Blue    |      25       |
|   Green   |      52       |
|    Red    |      41       |
|   Pink    |      36       |
|  Orange   |      39       |
|   Gray    |      23       |

```python
color = ['Blue', 'Green', 'Red', 'Pink', 'Orange', 'Gray']
freq = [25, 52, 41, 36, 39, 23]
```

---
# Bar Graph 
Based on the freq table

```python
import matplotlib.pyplot as plt

bar_color = ['tab:blue', 'tab:green', 'tab:red', 'tab:pink', 'tab:orange', 'tab:gray']
plt.bar(color, freq, color=bar_color)
plt.title('Bar Graph on the Frequency Table')
plt.xlabel('Colors')
plt.ylabel('Frequency')
for i in range(len(freq)):
    plt.text(i, freq[i], freq[i], ha='center')
plt.show()
```
![[Pasted image 20250424115323.png]]

---
# Pareto Chart
Type of Bar graph, in which bars are reverse sorted.

```python
import matplotlib.pyplot as plt

# Pareto chart
preference = dict(zip(color, freq))
new_dict = dict(sorted(preference.items(), key=lambda x:x[1], reverse=True))

bar_color = ['tab:blue', 'tab:green', 'tab:red', 'tab:pink', 'tab:orange', 'tab:gray']
plt.bar(new_dict.keys(), new_dict.values(), color=new_dict.keys())
plt.title('Pareto Graph on the Frequency Table')
plt.xlabel('Colors')
plt.ylabel('Frequency')
plt.show()
```
![[Pasted image 20250425143628.png]]

---
# Pie Chart
Plots chart in context of Relative Frequency.
>Relative Frequency = $\frac {Frequency}{\sum Total Frequency}$
>Relative Percentage = $\frac {Frequency}{\sum Total Frequency} \times 100$

| **Color** | **Frequency** | **Relative Freq** | **Relative %** |
|:---------:|:-------------:|:-----------------:|:--------------:|
|   Blue    |      25       |       0.116       |      11.6      |
|   Green   |      52       |       0.240       |       24       |
|    Red    |      41       |       0.189       |      18.9      |
|   Pink    |      36       |       0.167       |      16.7      |
|  Orange   |      39       |       0.180       |       18       |
|   Gray    |      23       |       0.106       |      10.6      |
| **TOTAL** |    **216**    |                   |                |

```python
import matplotlib.pyplot as plt

labels = color
size = freq
plt.pie(size, labels=labels, autopct='%1.1f%%', colors=color)
plt.show()
```
![[Pasted image 20250425145009.png]]
