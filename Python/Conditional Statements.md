
|          Extensive Example of SET          |   INDEX    |        Loops        |
| :----------------------------------------: | :--------: | :-----------------: |
| [[Extensive Example of SET\| <- Previous]] | [[Python]] | [[Loops\| Next ->]] |

### IF
This makes it possible to decide at runtime whether certain program parts should be executed or not.
```python
if condition:
	statement
	statement
	# further statements, if necessary
```

```python
person = input("Nationality? ")
if person == "french" or person == "French":
	print("Préférez-vous parler français?")
if person == "italian" or person == "Italian":
	print("Preferisci parlare italiano?")
```

---

### ELIF
```python
x = float(input("1st Number: "))
y = float(input("2nd Number: "))
z = float(input("3rd Number: "))
if x > y and x > z:
	maximum = x
elif y > x and y > z:
	maximum = y
else:
	maximum = z

print("The maximal value is: " + str(maximum))
```

---

### Nested IF-ELSE
```python
x = float(input("1st Number: "))
y = float(input("2nd Number: "))
z = float(input("3rd Number: "))

if x > y:
	if x > z:
		maximum = x
	else:
		maximum = z
else:
	if y > z:
		maximum = y
	else:
		maximum = z

print("The maximal value is: " + str(maximum))
```

---

### TERNARY IF Statement
```python
inside_city_limits = True
maximum_speed = 50 if inside_city_limits else 100
print(maximum_speed)
```
`Ouput: 50`
