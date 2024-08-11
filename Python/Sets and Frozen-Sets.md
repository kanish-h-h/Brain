
|          Zip          |   INDEX    |        Extensive Example of SET        |
| :-------------------: | :--------: | :------------------------------------: |
| [[Zip\| <- Previous]] | [[Python]] | [[Extensive Example of SET\| Next ->]] |

### SET
A set is a well-defined collection of objects.
![[Pasted image 20231231223942.png]]
A set contains an unordered collection of unique and immutable objects. 
The set data type is, as the name implies, a Python implementation of the sets as they are known from mathematics. 
This explains, why sets unlike lists or tuples can't have multiple occurrences of the same element.

```python
cities=set(("Paris","Lyon","London","Berlin","Paris","Birmingham"))
cities
```
`Output: Output: {'Berlin', 'Birmingham', 'London', 'Lyon', 'Paris'}`

---

### Immutable Sets
Sets are implemented in a way, which doesn't allow mutable objects. The following example demonstrates that
we cannot include, for example, lists as elements:
```python
cities = set((["Python","Perl"], ["Paris", "Berlin", "London"]))
cities
``` 
`Gives error`

Tuples on the other hand are fine:
```Python
cities = set((("Python","Perl"), ("Paris", "Berlin", "London")))
cities
```

---

### FROZEN SETS
Though sets can't contain mutable objects, sets are mutable:
```python
cities = set(["Frankfurt", "Basel","Freiburg"])
cities.add("Strasbourg")
cities
```
`Output: {'Basel', 'Frankfurt', 'Freiburg', 'Strasbourg'}`

Frozen-sets are like sets except that they cannot be changed, i.e. they are immutable:
```python
cities = frozenset(["Frankfurt", "Basel","Freiburg"])
cities.add("Strasbourg")
```
`Gives Error`

---

### Notation
We can define sets (since Python2.6) without using the built-in set function. We can use curly braces `{}` instead:
```python
adjectives = {"cheap","expensive","inexpensive","economical"}
adjectives
```
`Output: {'cheap', 'economical', 'expensive', 'inexpensive'}`

---
### SET Operations

#### ADD(element)
A method which adds an element to a set. This element has to be immutable.
```python
colours = {"red","green"}
colours.add("yellow")
colours
```
`Output: {'green', 'red', 'yellow'}`

#### CLEAR()
All elements will be removed from a set.
```python
cities = {"Stuttgart", "Konstanz", "Freiburg"}
cities.clear()
cities
```
`Output: set()`

#### COPY()
Creates a shallow copy, which is returned.
```python
more_cities = {"Winterthur","Schaffhausen","St. Gallen"}
cities_backup = more_cities.copy()
more_cities.clear()
cities_backup
```
`Output: {'Schaffhausen', 'St. Gallen', 'Winterthur'}`

The assignment "cities_backup = more_cities" just creates a pointer, i.e. another name, to the same data.

#### DIFFERENCE()
This method returns the difference of two or more sets as a new set, leaving the original set unchanged.
```python
x = {"a","b","c","d","e"}
y = {"b","c"}
z = {"c","d"}
x.difference(y)
```
`Output: {'a', 'd', 'e'}`
```python
x.difference(y).difference(z)
```
`Output: {'a', 'e'}`

Instead of using the method difference, we can use the operator "-":
```python
x - y
Output: {'a', 'd', 'e'}
x - y - z
Output: {'a', 'e'}
```

#### DIFFERENCE_UPDATE()
The method difference_update removes all elements of another set from this set. `x.difference_update(y)` is the
same as `x = x - y` or even `x -= y` works.
```python
x = {"a","b","c","d","e"}
y = {"b","c"}
x.difference_update(y)

x = {"a","b","c","d","e"}
y = {"b","c"}
x = x - y
x
```
`Output: {'a', 'd', 'e'}`

#### DISCARD(Element)
An element element will be removed from the set, if it is contained in the set. If element is not a member of the set, nothing will be done.
```python
x = {"a","b","c","d","e"}
x.discard("a")
x
```
`Output: {'b', 'c', 'd', 'e'}`
```python
x.discard("z")
x
```
`Output: {'b', 'c', 'd', 'e'}`

#### REMOVE(Element)
Works like discard(), but if element is not a member of the set, a KeyError will be raised.
```python
x = {"a","b","c","d","e"}
x.remove("a")
x
```
`Output: {'b', 'c', 'd', 'e'}`
```python
x.remove("z")
```
`Output: KeyError:'z'`

#### Union(S)
This method returns the union of two sets as a new set, i.e. all elements that are in either set. `x | y`
```python
x = {"a","b","c","d","e"}
y = {"c","d","e","f","g"}
x.union(y)
```
`Output: {'a', 'b', 'c', 'd', 'e', 'f', 'g'}`

#### INTERSECTION(S)
Returns the intersection of the instance set and the set s as a new set. In other words, a set with all the elements which are contained in both sets is returned. `x & y`
```python
x = {"a","b","c","d","e"}
y = {"c","d","e","f","g"}
x.intersection(y)
```
`Output: {'c', 'd', 'e'}`

#### ISDISJOINT()
This method returns True if two sets have a null intersection.
```python
x = {"a","b","c"}
y = {"c","d","e"}
x.isdisjoint(y)
```
`Output: False`

#### ISSUBSET()
`x.issubset(y)` returns True, if x is a subset of y. "<=" is an abbreviation for "Subset of" and ">=" for "superset of" "<" is used to check if a set is a proper subset of a set.
```python
x = {"a","b","c","d","e"}
y = {"c","d"}
x.issubset(y)
```
`Output: False`
```python
y.issubset(x)
				Output: True
x < y
				Output: False
y < x # y is a proper subset of x
				Output: True
x < x # a set can never be a proper subset of oneself.
				Output: False
x <= x
				Output: True
```

#### ISSUPERSET()
`x.issuperset(y)` returns True, if x is a superset of y. ">=" is an abbreviation for "issuperset of" ">" is used to check if a set is a proper superset of a set.
```python
x = {"a","b","c","d","e"}
y = {"c","d"}
x.issuperset(y)
```
`Output: True`
```python
x > y
				Output: True
x >= y
				Output: True
x >= x
				Output: True
x > x
				Output: False
x.issuperset(x)
				Output: True
```

#### POP()
pop() removes and returns an arbitrary set element. The method raises a KeyError if the set is empty.
```python
x = {"a","b","c","d","e"}
x.pop()
```
`Output: 'e'`
```python
x.pop()
```
`Output: 'a'`

[[Extensive Example of SET]]