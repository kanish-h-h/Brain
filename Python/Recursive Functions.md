
|          Functions          |   INDEX    |        Decorators        |
| :-------------------------: | :--------: | :----------------------: |
| [[Functions\| <- Previous]] | [[Python]] | [[Decorators\| Next ->]] |

**Recursion** is a method of programming or coding a problem, in which a function calls itself one or more times in its body. Usually, it is returning the return value of this function call. If a function definition satisfies the condition of recursion, we call this function a recursive function.
**Termination condition:** A recursive function has to fulfil an important condition to be used in a program: it has to terminate. 
A recursive function terminates, if with every recursive call the solution of the problem is
downsized and moves towards a base case. 
A *base case* is a case, where the problem can be solved without further recursion. A recursion can end up in an infinite loop, if the base case is not met in the calls.

### Factorial
```python
def factorial(n):
	if n == 0:                      # base case  
		return 1
	else:
		return n * factorial(n-1)   # recursion
```

<hr>

### Fibonacci  
```python
# 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, ...
def fib(n):
	if n == 0:
		return 0
	elif n == 1:
		return 1
	else:
		return fib(n-1) + fib(n-2)

# iterative solution
def fibi(n):
	old, new = 0, 1
	if n == 0:
		return 0
	for i in range(n-1):
		old, new = new, old+new
	return new
```

