
|          Dictionaries          |   INDEX    |        Sets and Frozen-Sets        |
| :----------------------------: | :--------: | :--------------------------------: |
| [[Dictionaries\| <- Previous]] | [[Python]] | [[Sets and Frozen-Sets\| Next ->]] |


The 'zip' of Python has absolutely nothing to do with it. The functionality 'zip' in Python is based on another meaning of the English word 'zip', "closing something with a zipper".
We will see that zip can be very easy to understand.
However, this does not apply if you come up with the idea of using the Python help function. The help text is
certainly partly responsible for this and frightens the
novices away:

>Return a zip object whose `.next()` method returns a tuple where the i-th element comes from the i-th iterable argument. The `.next()` method continues until the shortest iterable in the argument sequence is exhausted and then it raises StopIteration.

```python
a_couple_of_letters = ["a", "b", "c", "d", "e", "f"]
some_numbers = [5, 3, 7, 9, 11, 2]
print(zip(a_couple_of_letters, some_numbers))

for t in zip(a_couple_of_letters, some_numbers):
	print(t)
```

The use of zip is not restricted to lists and tuples. It can be applied to all iterable objects like lists, tuples,strings, dictionaries, sets, range and many more of course.

```python
food = ["ham", "spam", "cheese"]
for item in zip(range(1000, 1003), food):
	print(item)
```

---

### Calling ZIP with No Arguments
```python
for i in zip():
	print("This will not be printed")
```
The loop will not executed.

---

### Calling ZIP Arguments
```python
s = "Python"
for t in zip(s):
	print(t)
```
`Output: ('P',) ('y',) ('t',) ('h',) ('o',) ('n',)`
So this call creates an iterator which produces tuples with one single element, in our case the characters of the string.

---

### Parameters with Different Length
```python
colors = ["green", "red", "blue"]
cars = ["BMW", "Alfa Romeo"]
for car, color in zip(cars, colors):
	print(car, color)
```
`Output: BMW green  Alfa Romeo red`

---

### Advance use of ZIP

```python
cities_and_population = [("Zurich", 415367),
						 ("Geneva", 201818),
						 ("Basel", 177654),
						 ("Lausanne", 139111),
						 ("Bern", 133883),
						 ("Winterthur", 111851)]

cities, populations = list(zip(*cities_and_population))
print(cities)
print(populations)
```

```python
import pandas as pd

cities_and_population = [("Zurich", 415367),
						 ("Geneva", 201818),
						 ("Basel", 177654),
						 ("Lausanne", 139111),
						 ("Bern", 133883),
						 ("Winterthur", 111851)]

cities, populations = list(zip(*cities_and_population))
s = pd.Series(population, index=cities)
s.plot(kind='bar')
```

---

### Converting Iterable => Dictionary
```python
abc = "abcdef"
morse_chars = [".-", "-...", "-.-.", "-..", ".", "..-."]
text2morse = dict(zip(abc, morse_chars))
print(text2morse)
```
`Ouput: {'a': '.-', 'b': '-...', 'c': '-.-.', 'd': '-..', 'e': '.', 'f': '..-.'}`
