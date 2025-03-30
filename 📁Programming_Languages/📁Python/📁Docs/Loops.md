
|          Conditional Statements          |   INDEX    |        Iterators and iterables        |
| :--------------------------------------: | :--------: | :-----------------------------------: |
| [[Conditional Statements\| <- Previous]] | [[Python]] | [[Iterators and Iterables\| Next ->]] |

Essentially, there are three different kinds of loops:
- Count-controlled loops A construction for repeating a loop a certain number of times. An example of this kind of loop is the for-loop of the programming language C:
		`for (i=0; i <= n; i++)`
	Python doesn't have this kind of loop.
	
- Condition-controlled loop A loop will be repeated until a given condition changes, i.e. changes from True to False or from False to True, depending on the kind of loop. There are 'while loops' and 'do while' loops with this behavior.

- Collection-controlled loop This is a special construct which allows looping through the elements of a 'collection', which can be an array, list or other ordered sequence. Like the for loop of the bash shell (e.g. for i in *, do echo $i; done) or the foreach loop of Perl.

![[Pasted image 20240101142352.png]]

Python supplies two different kinds of loops: the while loop and the for loop, which correspond to the condition-controlled loop and collection-controlled loop.

---

### WHILE Loop
```python
n = 100
total_sum = 0
counter = 1
while counter <= n:
	total_sum += counter
	counter += 1
print("Sum of 1 until " + str(n) + " results in " + str(total_sum))
```
`Ouput: Sum of 1 until 100 results in 5050`

---

### Termination of a WHILE Loop
```python
import random

upper_bound = 20
lower_bound = 1
to_be_guessed = random.randint(lower_bound, upper_bound)
guess = 0
while guess != to_be_guessed:
	guess = int(input("New number: "))
	if guess == 0:
		# giving up
		print("Sorry that you're giving up!")
		break # break out of a loop, don't execute "else"
	if guess < lower_bound or guess > upper_bound:
		print("guess not within boundaries!")
	elif guess > to_be_guessed:
		print("Number too large")
	elif guess < to_be_guessed:
		print("Number too small")
else:
	print("Congratulations. You made it!")
```
`Output: Congratulations. You made it!`

---

### FOR Loop
Python for loop is an iterator based for loop. It steps through the items of lists, tuples, strings, the keys of dictionaries and other iterables. The Python for loop starts with the keyword "for" followed by an arbitrary variable name, which will hold the values of the following sequence object, which is stepped through. The general syntax looks like this:
>for in : else:

```python
languages = ["C", "C++", "Perl", "Python"]
for language in languages:
	print(language)
```

```python
edibles = ["bacon", "spam", "eggs", "nuts"]
for food in edibles:
	if food == "spam":
		print("No more spam please!")
		break
	print("Great, delicious " + food)
else:
	print("I am so glad: No spam!")
print("Finally, I finished stuffing myself")
```

---

### RANGE() 
The built-in function range() is the right function to iterate over a sequence of numbers. It generates an iterator of arithmetic progressions
```python
list(range(10))
```
`Output: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]`

> range(begin, end)

```python
n = 100
sum = 0
for counter in range(1, n+1):
	sum = sum + counter
print("Sum of 1 until %d: %d" % (n, sum))
```
`Ouput: Sum of 1 until 100: 5050`

Iteration over a lists with range()
```python
fibonacci = [0, 1, 1, 2, 3, 5, 8, 13, 21]
for i in range(len(fibonacci)):
	print(i,fibonacci[i])
print()
```

---

### List Iteration with Side Effects
If you loop over a list, it's best to avoid changing the list in the loop body.
```python
colours = ["red"]
for i in colours:
	if i == "red":
		colours += ["black"]
	if i == "black":
		colours += ["white"]
print(colours)
```
`Ouput: ['red', 'black', 'white']`

To avoid these side effects, it's best to work on a copy by using the slicing operator:
```python
colours = ["red"]
for i in colours[:]:
	if i == "red":
		colours += ["black"]
	if i == "black":
		colours += ["white"]
print(colours)
```
`Output: ['red', 'black']`
