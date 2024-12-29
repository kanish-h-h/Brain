
|          List          |   INDEX    |        Dictionaries        |
| :--------------------: | :--------: | :------------------------: |
| [[List\| <- Previous]] | [[Python]] | [[Dictionaries\| Next ->]] |


A Tuple is an Immutable list.
A tuple cannot be change in any way if created.

benefits?
- Faster than lists.
- If you know that some data doesn't have to be changed, you should use tuples instead of lists, because this protects your data against accidental changes.
- The main advantage of tuples is a tuples can be used as `keys` in `dictionaries`, while list can't.

```python
t = ("tuple", "are", "immutable")
```

```python
my_dict = {('John', 25): 'Engineer', ('Alice', 30): 'Doctor', ('Bob', 28): 'Teacher'}

print(my_dict[('John', 25)])  # Output: Engineer
print(my_dict[('Alice', 30)])  # Output: Doctor

my_dict[('Eva', 26)] = 'Artist'

print(my_dict)

```