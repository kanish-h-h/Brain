
|          Basics          |   INDEX    |        Tuples        |
| :----------------------: | :--------: | :------------------: |
| [[Basics\| <- Previous]] | [[Python]] | [[Tuples\| Next ->]] |


Lists are related to arrays of programming languages like C, C++ or Java, but Python lists are by far more flexible and powerful than "classical" arrays.

For example, not all the items in a list need to have the same type. Which are mutable.
Furthermore, lists can grow in a program run, while in C the size of an array has to be fixed at compile time.

>	Generally, a list is a collections of objects.

>	A list is an ordered group of items or elements.

```python
mixed_list = [42, "What's the question?", 3.1415]
```

A list can be seen as `STACK`.

```python
lst = ["easy", "simple", "cheap", "free"]
```
`output: 'free'

- To access value:	list[i]
- slicing:  list[:] list[:n] list[m:n] list[m:n:p]

---

### APPEND and POP
>	list.append(x)
append element at the end of list.

```python
lst = [6, 8, 9]
lst.append(47)
lst
```
`output: [6, 8, 9, 47]`

>	list.pop(i)
>returns the element from the list at (i)th location.
>without argument `.pop()` => `.pop(-1)`
```python
lst = [2, 3, 6]
lst.pop(0)
```
`output: 2`

---

### EXTEND
>	lst.append(lst2)
>adds new elements to a list.
>either be an element or an entire list.

```python
lst = [1, 2, 3]
lst2 = [4, 5, 6]
lst.append(lst2)
print(lst)
```
`output: [1, 2, 3, 4, 5, 6]`

with a tuple:
```python
lst = ["Java", "C", "PHP"]
t = ("C#", "Jython", "Python")
lst.extend(t)
print(lst)
```
`output: ['Java', 'C', 'PHP', 'C#', 'Jython', 'Python']`

---

### REMOVE
>	s.remove(x)
>remove the element without knowing the position.

```python
colours = ["red", "blue", "green", "yellow"]
colours.remove("green")
print(colours)
```
`output: ['red', 'blue', 'yellow']`

---

### FIND POSITION
>	s.index(x,i)
>index is used to find the value=>x, position=>i.

```python
colours = ['red', 'green', 'blue', 'green', 'yellow']
colours.index("green")         # => 1
colours.index("green", 2)      # => 3
colours.index("green", 3,4)    # => 3
```

---

### Insert
>	s.insert(index, object)
>insert object at a particular index.

```python
lst = ["German is spoken", "in Germany,", "Austria", "Switzerland"]
lst.insert(3, "and")
print(lst)
```
`Output: ['German is spoken', 'in Germany,', 'Austria', 'and', 'Switzerland']`

---

### COPY
>	lst.copy()
>copy another list.

```python
person1 = ["Swen", 
		   ["Seestrasse", "Konstanz"]]
person2 = person1.copy()
person2[0] = "sarah"
print(person2)
```
`Output: ['Sarah', ['Seestrasse', 'Konstanz']]`