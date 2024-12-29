
|          Recursive Functions          |   INDEX    |        Memoization with Decorators        |
| :-----------------------------------: | :--------: | :---------------------------------------: |
| [[Recursive Functions\| <- Previous]] | [[Python]] | [[Memoization with Decorators\| Next ->]] |

A decorator in Python is any callable Python object that is used to modify a function or a class.
A reference to a function "func" or a class "C" is passed to a decorator and the decorator returns a
modified function or class.
The modified functions or classes usually contain calls to the original function "func" or class "C".

Even though it is the same underlying concept, we have two different kinds of decorators in Python:
	• Function decorators
	• Class decorators

### First steps to Decorators
```python
def succ(x):
	return x + 1
successor = succ
successor(10)
```
```Output: 11```
```python
succ(10)
```
```Output: 11```

This means that we have two names, i.e. "*succ*" and "*successor*" for the same function. The next important fact is that we can delete either "*succ*" or "*successor*" without deleting the function itself.

```python
del succ
successor(10)
```
```Output: 11```

<hr>

### Functions inside Functions
```python
def f():
	
	def g():
		print("it's g")
		print("called g")
		
	print("this is f")
	print("calling g now")
	g()

f()
```
```
this is f
calling g now
it's g
called g
```

```python
def temperature(t):
	def celsius2fahrenheit(x):
		return 9 * x / 5 + 32
	result = "It's " + str(celsius2fahrenheit(t)) + " degrees!"
	return result

print(temperature(20))
```
```Output: It's 68.0 degrees!```

In case of `factorial` what if input is given by a negative number (integer)
```python
def factorial(n):
	"""calculate factorial of a given positive integer"""
	def inner_factorial(n)"
		if n == 0: 
			return 1
		else:
			return n * inner_factorial(n-1)
	if type(n) == int and n >= 0:
		return inner_factorial(n)
	else:
		raise TypeError("n should be a positive integer")
```

<hr>

### Functions as Parameters
It gets useful in combination with two further powerful possibilities of Python functions. Due to the fact that every parameter of a function is a reference to an object and functions are objects as well, we can pass functions - or better "references to functions" - as parameters to a function.
```python
def g():
	print("it's g")
	print("called g")

def f(func):
	print("it's f")
	print("call 'func' now")
	func()

f(g)
```

```python
import math
def foo(func):
	print("The function " + func.__name__ + " was passed to foo")
	res = 0
	for x in [1, 2, 2.5]:
		res += func(x)
	return res
	
print(foo(math.sin))
print(foo(math.cos))
```
```
The function sin was passed to foo
2.3492405557375347
The function cos was passed to foo
-0.6769881462259364
```

<hr>

### Function  returning Functions
The output of a function is also a reference to an object. Therefore functions can return references to function objects.
```python
def f(x):
	def g(y):
		return y + x + 3
	return g
	
nf1 = f(1)
nf2 = f(3)
print(nf1(1))
print(nf2(1))
```
```
5
7
```

```python
def greeting_func_gen(lang):
	def customized_greeting(name):
		if lang == "de":   # German
			phrase = "Guten Morgen "
		elif lang == "fr": # French
			phrase = "Bonjour "
		elif lang == "it": # Italian
			phrase = "Buongiorno "
		elif lang == "tr": # Turkish
			phrase = "Günaydın "
		elif lang == "gr": # Greek
			phrase = "Καλημερα "
		else:
			phrase = "Hi "
		return phrase + name + "!"
	return customized_greeting

say_hi = greeting_func_gen("tr")
print(say_hi("Gülay"))  # this Turkish name means "rose moon"
```
```Output: Günaydın Gülay!```

It is getting more useful and at the same time more mathematically oriented in the following example. 
We will implement a polynomial "factory" function now. We will start with writing a version which can create polynomials of degree 2.
$$ p(x) = ax^2 + bx + c $$
The Python implementation as a polynomial factory function can be written like this:
```python
def polynomial_creator(a, b, c):
	def polynomial(x):
		return a*x**2 + b*x + c
	return polynomial

p1 = polynomial_creator(2,3,-1)
p2 = polynomial_creator(-1,2,1)

for x in range(-2,2,1):
	print(x, p1(x), p2(x))
```
```
-2 1 -7
-1 -2 -2
0 -1 1
1 4 2
```

We can generalise our factory function so that it can work for polynomials of arbitrary degree:
$$ \sum_{k=0}^n a_k ⋅ x^k = a_n ⋅ x^n + a_{n − 1} ⋅ x^{n − 1} + . . . + a_2 ⋅ x^2 + a_1 ⋅ x + a_0 $$
```python
def polynomial_creator(*coefficients):
	""" Coefficients are in form a_n, ... a_1, a_0 """
	def polynomial(x):
		res = 0
		for index, coeff in enumerate(coefficients[::-1]):
			res += coeff * x ** index
		return res
	return polynomial

p1 = polynomial_creator(4)
p2 = polynomial_creator(2,4)
p3 = polynomial_creator(1,8,-1,3,2)
p4 = polynomial_creator(-1,2,1)

for x in range(-2,2,1):
	print(x, p1(x), p2(x), p3(x), p4(x))
```
```
-2 4 0 -56 -7
-1 4 2 -9 -2
0 4 4 2 1
1 4 6 13 2
```

The polynomial function inside of our decorator polynomial_creator can be implemented more efficiently. We can factorize it in a way so that it doesn't need any exponentiation.
Factorised version of a general polynomial without exponentiation:
$$ res = (. . . (a_n ⋅ x + a_{n − 1}) ⋅ x + . . . + a_1) ⋅ x + a_0 $$
Implementation of our polynomial creator decorator avoiding exponentiation:
```python
def polynomial_creator(*coeff):
	"""coefficients are in form a_n, a_n_1, ... a_1, a_0"""
	def polynomial(x):
		res = coeff[0]
		for i in range(1, len(coeff)):
			res = res * x + coeff[i]
		return res
	return polynomial

p1 = polynomial_creator(4)
p2 = polynomial_creator(2,4)
p3 = polynomial_creator(1,8,-1,3,2)
p4 = polynomial_creator(-1,2,1)

for x in range(-2,2,1):
	print(x, p1(x), p2(x), p3(x), p4(x))
```
```
-2 4 0 -56 -7
-1 4 2 -9 -2
0 4 4 2 1
1 4 6 13 2
```

<hr>

### A Simple Decorator
```python
def our_decorator(func):
	def function_wrapper(x):
		print("Before calling" + func.__name__)
		func(x)
		print("After calling" + func.__name__)
	return function_wrapper

def foo(x):
	print("foo is called with" + str(x))

print("we call foo before decorator:")
foo("Hi")

print("we call deccorator foo with f:")
foo = our_decorator(foo)

print("we call foo after decoration:")
foo(42)
```
```
We call foo before decoration:
Hi, foo has been called with Hi
We now decorate foo with f:
We call foo after decoration:
Before calling foo
Hi, foo has been called with 42
After calling foo
```

<hr>

### Usual Syntax for Decorators in Python
We will do a proper decoration now. The decoration occurs in the line before the function header.
The "@" is followed by the decorator function name.

We will rewrite now our initial example. Instead of writing the statement
`foo = our_decorator(foo)`  we can write `@our_decorator`
```python
def our_decorator(func):
	def function_wrapper(x):
		print("Before calling " + func.__name__)
		func(x)
		print("After calling " + func.__name__)
	return function_wrapper

@our_decorator
def foo(x):
	print("foo has been called with " + str(x))

foo("Hi")
```
```
Before calling foo
Hi, foo has been called with Hi
After calling foo
```

```python
def our_decorator(func):
	def function_wrapper(x):
		print("Before calling " + func.__name__)
		res = func(x)
		print(res)
		print("After calling " + func.__name__)
	return function_wrapper

@our_decorator
def succ(n):
	return n + 1

succ(10)
```
```
Before calling succ
11
After calling succ
```

<hr>

### Use Cases for Decorators
The following program uses a decorator function to ensure that the argument passed to the function factorial is a positive integer:
```python
def argument_test_natural_number(f):
	def helper(x):
		if type(x) == int and x > 0:
			return f(x)
		else:
			raise Exception("Argument is not an integer")
	return helper

@argument_test_natural_number
def factorial(n):
	if n == 1:
		return 1
	else:
		return n * factorial(n-1)

for i in range(1, 10):
	print(i, factorial(i))

print(factorial(-1))
```
```
1 1
2 2
3 6
4 24
5 120
6 720
7 5040
8 40320
9 362880
Traceback (most recent call last):
  File "<main.py>", line 19, in <module>
  File "<main.py>", line 6, in helper
Exception: Argument is not an integer
```

<hr>

### Counting Function calls with Decorators
The following example uses a decorator to count the number of times a function has been called. 
To be precise, we can use this decorator solely for functions with exactly one parameter:
```python
def call_counter(func):
	def helper(x):
		helper.calls += 1
		return func(x)
	helper.calls = 0
	return helper

@call_counter
def succ(x):
	return x + 1

print(succ.calls)
for i in range(10):
	succ(i)
print(succ.calls)
```
```
0
10
```

We pointed out that we can use our previous decorator only for functions, which take exactly one parameter.
We will use the `*args` and `**kwargs` notation to write decorators which can cope with functions with an arbitrary number of positional and keyword parameters.
```python
def call_counter(func):
	def helper(*args, **kwargs):
		helper.calls += 1
		return func(*args, **kwargs)
	helper.calls = 0
	return helper

@call_counter
def succ(x):
	return x + 1

@call_counter
def mull(x, y=1):
	return x*y + 1

print(succ.calls)
for i in range(10):
	succ(i)

mull(3, 4)
mull(4)
mull(y=3, x=2)

print(succ.calls)
print(mull.calls)
```
```
0
10
3
```

<hr>

### Decorators with Parameters
```python
def evening_greeting(func):
	def function_wrapper(x):
		print("Good evening, " + func.__name__ + " returns:")
		return func(x)
	return function_wrapper
	
def morning_greeting(func):
	def function_wrapper(x):
		print("Good morning, " + func.__name__ + " returns:")
		return func(x)
	return function_wrapper

@evening_greeting
def foo(x):
	print(42)

foo("Hi")
```
```
Good evening, foo returns:
42
```

<hr>

### Using Wraps from Functools
The way we have defined decorators so far hasn't taken into account that the attributes
• `__name__` (name of the function),
• `__doc__` (the docstring) and
• `__module__` (The module in which the function is defined) of the original functions will be lost after the decoration.

The following decorator will be saved in a file greeting_decorator.py:
```
def greeting(func):
	def function_wrapper(x):
		""" function_wrapper of greeting """
		print("Hi, " + func.__name__ + " returns:")
	return func(x)
return function_wrapper
```
we call it in the following program:
```python
from greeting_decorator import greeting

@greeting
def f(x):
	"""just some silly function"""
	return x + 4

f(10)
print("function name: " + f.__name__)
print("docstring: " + f.__doc__)
print("module name: " + f.__module__)
```
```
Hi, f returns:
function name: function_wrapper
docstring: function_wrapper of greeting
module name: greeting_decorator
```

<hr>

### Classes instead of Functions
Before we can define a decorator as a class, we have to introduce the `__call__` method of classes.
We mentioned already that a decorator is simply a callable object that takes a function as an input parameter.
A function is a callable object, but lots of Python programmers don't know that there are other callable *objects*. 
A callable object is an object which can be used and behaves like a function but might not be a function. It is possible to define classes in a way that the instances will be callable objects.
The `__call__` method is called, if the instance is called "like a function", i.e. using brackets.
```python
class A:
	def __init__(self):
		print("An instance of A was initialized")
	def __call__(self, *args, **kwargs):
		print("Arguments are: ", args, kwargs)

x = A()
print("now calling the instance: ")
x(3, 4, x=11, y=10)
print("Let's call it again: ")
x(3, 4, x=11, y=10)
```
```
An instance of A was initialized
now calling the instance:
Arguments are: (3, 4) {'x': 11, 'y': 10}
Let's call it again:
Arguments are: (3, 4) {'x': 11, 'y': 10}
```

We can write a class for the fibonacci function by using the `__call__` method:
```python
class Fibonacci:
	
	def __init__(self):
		self.cache = {}
	
	def __call__(self, n):
		if n not in self.cache:
			if n == 0:
				self.cache[0] = 0
			elif n == 1:
				self.cache[1] = 1
			else:
				self.cache[n] = self.__call__(n-1) + self.__call__(n-2)
		return self.cache[n]

fib = Fibonacci()

for i in range(15):
	print(fib(i), end=", ")
```
```Output: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377,```

<hr>

### Using a Class as a Decorator
```python
def decorator(f):
	def helper():
		print("Decorating", f.__name__)
		f()
	return helper

@decorator
def foo():
	print("inside foo()")

foo()
```
```
Decorating foo
inside foo()
```

implementing as a class
```python
class decorator:
	def __init__(self, f):
		self.f = f
	
	def __call__(self):
		print("Decorating", self.f.__name__)
		self.f()

@decorator
def foo():
	print("inside foo()")

foo()
```
```
Decorating foo
inside foo()
```
