
|          Iterators and iterables          |   INDEX    |        Functions        |
| :---------------------------------------: | :--------: | :---------------------: |
| [[Iterators and Iterables\| <- Previous]] | [[Python]] | [[Functions\| Next ->]] |


```python
kaffeeliste = {"Peter": 0,
			  "Eva": 0,
			  "Franka": 0}
while True:
	name = input("Name: ")
	if name == "":
		break
	kaffeeliste[name] += 1
	print(kaffeeliste[name])
print("kaffeeliste: ", kaffeeliste)
```
```
Output: 1
		1
		2
		1
		kaffeeliste: {'Peter': 1, 'Eva': 1, 'Franka': 2}
```

---

```python
kaffeeliste = {"Peter": 0,
			   "Eva": 0,
			   "Franka": 0}
teeliste = {"Peter": 0,
		    "Eva": 0,
		    "Franka": 0}

while True:
	name = input("Name: ")
	if name == "":
		break
	getränk = input("Getränk (kaffee/tee): ")
	if getränk.lower() == "kaffee":
		kaffeeliste[name] += 1
		print(kaffeeliste[name])
	elif getränk.lower() == "tee":
		teeliste[name] += 1
		print(teeliste[name])
print("Kaffeeliste: ", kaffeeliste)
print("Teeliste: ", teeliste)
```
```
Output: 1
		1
		1
		Kaffeeliste: {'Peter': 1, 'Eva': 1, 'Franka': 0}
		Teeliste: {'Peter': 0, 'Eva': 0, 'Franka': 1}
```

---

```python
supermarket = { "milk": {"quantity": 20, "price": 1.19},
				"biscuits": {"quantity": 32, "price": 1.45},
				"butter": {"quantity": 20, "price": 2.29},
				"cheese": {"quantity": 15, "price": 1.90},
				"bread": {"quantity": 15, "price": 2.59},
				"cookies": {"quantity": 20, "price": 4.99},
				"yogurt": {"quantity": 18, "price": 3.65},
				"apples": {"quantity": 35, "price": 3.15},
				"oranges": {"quantity": 40, "price": 0.99},
				"bananas": {"quantity": 23, "price": 1.29}}

total_value = 0
for article, numbers in supermarket.items():
	quantity = numbers["quantity"]
	price = numbers["price"]
	product_price = quantity * price
	article = article + ':'
	print(f"{article:15s} {product_price:08.2f}")
	total_value += product_price
print("="*24)
print(f"Gesamtsumme: {total_value:08.2f}")
```
```
milk:           00023.80
biscuits:       00046.40
butter:         00045.80
cheese:         00028.50
bread:          00038.85
cookies:        00099.80
yogurt:         00065.70
apples:         00110.25
oranges:        00039.60
bananas:        00029.67
========================
Gesamtsumme:    00528.37
```
