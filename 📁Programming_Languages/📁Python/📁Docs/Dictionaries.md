
|          Tuples          |   INDEX    |        Zip        |
| :----------------------: | :--------: | :---------------: |
| [[Tuples\| <- Previous]] | [[Python]] | [[Zip\| Next ->]] |


A dictionaries is an unordered sets.
In dictionaries the items are accessed via keys.
>dictionaries ==> {key:values}

Dictionaries are the Python implementation of an abstract data type.
Dictionaries are implemented as hash tables.

Dictionaries don't support the sequence operation of the sequence data types like strings, tuples and lists.
Dictionaries belong to the built-in mapping type, but so far, they are the sole representative of this kind!

```python
city_population = {"New York City": 8_550_405,
				   "Los Angeles": 3_971_883,
				   "Toronto": 2_731_571,
				   "Chicago": 2_720_546}
```

```python
food = {"bacon": "yes", "egg": "yes", "spam": "no" }
print(food)
```

---

### Operations
- len(dict)   => number of stored entries
- del d[k]    => delete the key k with its value
- k in d      => True, if a key k exists in d
- k not in d  => True, if a key doesn't exists in d

---

Following is a dictionary contains a mapping from latin characters --> morsecode

```python
morse = {
		 "A" : ".-",
		 "B" : "-...",
		 "C" : "-.-.",
		 "D" : "-..",
		 "E" : ".",
		 "F" : "..-.",
		 "G" : "--.",
		 "H" : "....",
		 "I" : "..",
		 "J" : ".---",
		 "K" : "-.-",
		 "L" : ".-..",
		 "M" : "--",
		 "N" : "-.",
		 "O" : "---",
		 "P" : ".--.",
		 "Q" : "--.-",
		 "R" : ".-.",
		 "S" : "...",
		 "T" : "-",
		 "U" : "..-",
		 "V" : "...-",
		 "W" : ".--",
		 "X" : "-..-",
		 "Y" : "-.--",
		 "Z" : "--..",
		 "0" : "-----",
		 "1" : ".----",
		 "2" : "..---",
		 "3" : "...--",
		 "4" : "....-",
		 "5" : ".....",
		 "6" : "-....",
		 "7" : "--...",
		 "8" : "---..",
		 "9" : "----.",
		 "." : ".-.-.-",
		 "," : "--..--"
		 }

from morsecode import morse
len(morse)

"a" in morse
"A" in morse
"a" not in morse

word = input("your word: ")
for char in word.upper():
	if char == " ":
		morse_word += "  "
	else:
		if char not in morse:
			continue
		morse_word += morse[char] + " "

print(morse_word)
```

---

### POP() and POPITEM()
>	dict1 = dict.pop(k)
>delete the value by accessing the key.

```python
en_de = {"Austria":"Vienna", "Switzerland":"Bern",
		 "Germany":"Berlin", "Netherlands":"Amsterdam"}
capitals = {"Austria":"Vienna", "Germany":"Berlin",
			"Netherlands":"Amsterdam"}
capital = capitals.pop("Austria")
print(capital)
```

>	dict1 = dict.popitem()
>delete the whole key-value pair from the end.

```python
capitals = {"Springfield": "Illinois","Augusta": "Maine",
			"Boston": "Massachusetts","Albany": "New York"}

(city, state) = capitals.popitem()
(city, state)
```
`output: {"Springfield": "Illinois","Augusta": "Maine","Boston": "Massachusetts"}`

---

### ACCESSING NON-EXISTING KEYS
```python
locations = {"Toronto": "Ontario", 
			 "Vancouver": "British Columbia"}
locations["Ottawa"] # -> error

province = "Ottawa"

if province in locations:
	print(locations[province])
else:
	print(province + " is not in locations")
```

---

### COPY()
```python
words = {'house': 'Haus', 'cat': 'Katze'}
w = words.copy()
words["cat"]="chat"
print(w)
```

---

### CLEAR()
The content of a dictionary can be cleared with the method clear(). The dictionary is not deleted, but set to an empty dictionary:
```python
w.clear()
print(w)
```

---

### UPDATE()
>	dict1.update(dict)

for concatenation of dictionaries
```python
knowledge = {"Frank": {"Perl"}, "Monica":{"C","C++"}}
knowledge2 = {"Guido":{"Python"}, "Frank":{"Perl", "Python"}}
knowledge.update(knowledge2)
knowledge
```

---

### Iteration over Dictionaries
No method is needed to iterate over the keys of a dictionary:
```python
d = {"a":123, "b":34, "c":304, "d":99}
for key in d:
	print(key)
```

by using keys() method:
```python
for key in d.keys():
	print(key)
```

by using values() method:
```python
for value in d.values():
	print(value)
```

above loop is equivalent to the following:
```python
for key in d:
	print(d[key])
```

---

Between lists and dictionaries there is a connection of tuples. That is, (key, value).
### ITEMS()
>	dict.items()
>return dictionaries key-value as a tuple

```python
my_dict = {'apple': 5, 'banana': 7, 'orange': 3}

# Using items() to get key-value pairs
for key, value in my_dict.items():
    print(f"Key: {key}, Value: {value}")
```
`Output: Key: apple, Value: 5...`

---

### Lists from dictionaries
By using methods `item()`, `keys()`, `values()` we can make lists from dictionaries.

```python
w = {"house": "Haus", "cat": "", "red": "rot"}
items_view = w.items()
items = list(items_view)
items
```

```python
# list of keys
keys_view = w.keys()
keys = list(keys_view)
keys

# list of values
values_view = w.values()
values = list(values_view)
values
```

---

### Lists into Dictionaries
For this we are going to use zip function for zipping.

```python
dishes = ["pizza", "sauerkraut", "paella", "hamburger"]
countries = ["Italy", "Germany", "Spain", "USA"]
country_specialities_iterator = zip(countries, dishes)
print(country_specialities_iterator)

country_specialities = list(country_specialities_iterator)
print(country_specialities)
```

more efficient way by iterating over zip object in a for loop. With this there is no need for making a list.
```python
for country, dish in zip(countries, dishes):
	print(country, dishes)
```

Now list of two-tuples are formed.Now turning this into dictionaries.
```python
country_specialities_dict = dict(country_specialities)
print(country_specialities_dict)
```

Another way:
```python
dishes = ["pizza", "sauerkraut", "paella", "hamburger"]
countries = ["Italy", "Germany", "Spain", "USA"]
dict(zip(countries, dishes))
```

If two list doesn't contains same number of elements. The superfluous elements, which cannot be paired, will be ignored:
```python
dishes = ["pizza", "sauerkraut", "paella", "hamburger"]
countries = ["Italy", "Germany", "Spain", "USA"," Switzerland"]
country_specialities = list(zip(countries, dishes))
country_specialities_dict = dict(country_specialities)
print(country_specialities_dict)
```

#### Everything in one step
```python
country_specialities_dict = dict(zip(["pizza", "sauerkraut",
									  "paella", "hamburger"]
									  ["Italy", "Germany",
									  "Spain", "USA",
									  "Switzerland"]))
print(country_specialities_dict)
```

More readable format:
```python
dishes = ["pizza", "sauerkraut", "paella", "hamburger"]
countries = ["Italy", "Germany", "Spain", "USA"]
country_specialities_zip = zip(dishes,countries)
country_specialities_dict = dict(country_specialities_zip)
print(country_specialities_dict)
```

