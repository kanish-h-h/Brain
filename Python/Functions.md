
|         Working with Dicitionaries          |   INDEX    |        Recursive Functions        |
| :-----------------------------------------: | :--------: | :-------------------------------: |
| [[Working with Dictionaries\| <- Previous]] | [[Python]] | [[Recursive Functions\| Next ->]] |

```python
def f(x, y):
	z = 2 * (x + y)
	return z
	
print("Program starts!")
a = 3
res1 = f(a, 2+a)
print("Result of function call:", res1)
a = 4
b = 7
res2 = f(a, b)
print("Result of function call:", res2)
```
```
Output: Program starts!
		Result of function call: 16
		Result of function call: 22
```

---

### Default Keyword and `__default__` 
When we define a Python function, we can set a default value to a parameter. If the function is called without
the argument, this default value will be assigned to the parameter. This makes a parameter optional. To say it in other words: Default parameters are parameters, which don't have to be given, if the function is called. 

In this case, the default values are used. We will demonstrate the operating principle of default parameters with a simple example. 

The following function `hello`, which isn't very useful, greets a person. If no name is given, it will greet everybody:
```python
def hello(name="everybody"):
	""" Greets a person """
	result = "Hello " + name + "!")

hello("Peter")
hello()
```
```
Hello Peter!
Hello everybody!
```

Default list value within a function:
```python
def spammer(bag=[]):
	bag.append("spam")
	return bag
spammer()
```
`Output: ['spam']`

But if we again call `spammer()`
```python
spammer()
```
`Output: ['spam', 'spam']`

To understand what is going on, you have to know what happens when the function is defined. The compiler
creates an attribute `__defaults__`
```python
def spammer(bag=[]):
	bag.append("spam")
	return bag

spammer.__defaults__
```
`Output: ([],)`

Whenever we will call the function, the parameter bag will be assigned to the list object referenced by
`spammer.__defaults__[0]` :
```python
for i in range(5):
print(spammer())
print("spammer.__defaults__", spammer.__defaults__)
```
```
Output: ['spam']
		['spam', 'spam']
		['spam', 'spam', 'spam']
		['spam', 'spam', 'spam', 'spam']
		['spam', 'spam', 'spam', 'spam', 'spam']
		spammer.__defaults__ (['spam', 'spam', 'spam', 'spam', 'spam'],)
```

Now, you know and understand what is going on, but you may ask yourself how to overcome this problem.
The solution consists in using the immutable value `None` as the default. This way, the function can set bag
dynamically (at run-time) to an empty list:
```python
def spammer(bag=None):
	if bag is None:
		bag = []
	bag.append("spam")
	return bag
	
for i in range(5):
	print(spammer())
print("spammer.__defaults__", spammer.__defaults__)
```
```
Output: ['spam']
		['spam']
		['spam']
		['spam']
		['spam']
		spammer.__defaults__ (None,)
```

---

### `__doc__` String
The first statement in the body of a function is usually a string statement called a DocString, which can be accessed with the `function_name.__doc__`.
```python
def hello(name="everybody"):
	""" Greets a person """
	print("Hello " + name + "!")
	
print("The docstring of the function hello: " + hello.__doc__)
```
`The docstring of the function hello: Greets a person`

---

### Returning Multiple values
```python
def fib_interval(x):
	""" returns the largest fibonacci
	number smaller than x and the lowest
	fibonacci number higher than x"""
	if x < 0:
		return -1
	old, new = 0, 1
	while True:
		if new < x:
			old, new = new, old+new
		else:
			if new == x:
				new = old + new
			return (old, new)

while True:
	x = int(input("Your number: "))
	if x <= 0:
		break
	lub, sup = fib_interval(x)
	print("Largest Fibonacci Number smaller than x: " + str(lub))
	print("Smallest Fibonacci Number larger than x: " + str(sup))
```

---

### Local & Global Variables In Function
Variable names are by default local to the function, in which they get defined.
```python
def f():
	print(s) # free occurrence of s in f
	s = "Python"
f()
```

```python
def f():
	s = "Perl"
	print(s) # now s is local in f
	s = "Python"
f()
print(s)
```
```
Ouput:  Perl
		Python
```

```python
def f():
	print(s) # This means a free occurrence, contradiction to bein local
s = "Perl"
print(s) # This makes s local in f

s = "Python"
f()
print(s)
```
`Output: UnboundLocalError: local variable 's' referenced before assignment`

To resolve this above error of `UnBoundedLocalError`: 
```python
def f():
	global s
	print(s)
	s = "dog"
	print(s)
s = "cat"
f()
print(s)
```
```
Ouput:  cat
		dog
		dog
```

We made the variable s global inside of the script. Therefore anything we do to s inside of the function body of f is done to the global variable s outside of f.
```python
def f():
	global s
	print(s)
	s = "dog" # globally changed
	print(s)
	
def g():
	s = "snake"
	print(s) # local s

s = "cat"
f()
print(s)
g()
print(s)
```
```
Ouput:  cat
		dog
		dog
		snake
		dog
```

---

### Arbitrary Number of Parameter `*arg`
There are many situations in programming, in which the exact number of necessary parameters cannot be determined a priority. An arbitrary parameter number can be accomplished in Python with so-called tuple
references. An asterisk `*` is used in front of the last parameter name to denote it as a tuple reference. This
asterisk shouldn't be mistaken for the C syntax, where this notation is connected with pointers.

```python
def arithmetic_mean(first, *values):
	""" This function calculates the 
	arithmetic mean of a non-empty
	arbitrary number of numerical values """
	return (first + sum(values)) / (1 + len(values))

print(arithmetic_mean(45,32,89,78))
print(arithmetic_mean(8989.8,78787.78,3453,78778.73))
print(arithmetic_mean(45,32))
print(arithmetic_mean(45))
```
```
Output: 61.0
		42502.3275
		38.5
		45.0
```

```python
x = [3, 5, 9]
arithmetic_mean(x[0], x[1], x[2]) # cannot call arithmetic_mean(x) can't cope with a list
			OR
arithmetic_mean(*x)
```

Also this `*` is also help to "unpack" or singularize the list.
```python
my_list = [('a', 232),
		   ('b', 343),
		   ('c', 543),
		   ('d', 23)]
list(zip(*my_list))
```
`Ouput: [('a', 'b', 'c', 'd'), (232, 343, 543, 23)]`

---

### Arbitrary Number of Keyword Parameters `**kwargs`
In the previous chapter we demonstrated how to pass an arbitrary number of positional parameters to a
function. It is also possible to pass an arbitrary number of keyword parameters to a function as a `dictionary`.To this purpose, we have to use the double asterisk `**`
```python
def f(**kwargs):
	print(kwargs)
f()
```
`Ouput: {}`

```python
f(de="German",en="English",fr="French")
```
`Output: {'de': 'German', 'en': 'English', 'fr': 'French'}`

One use case is the following:
```python
def f(a, b, x, y):
	print(a, b, x, y)
d = {'a':'append', 'b':'block','x':'extract','y':'yes'}
f(**d)
```
`Output: append block extract yes`
