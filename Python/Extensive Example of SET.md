
|          Sets and Frozen-Sets          |   INDEX    |             Conditional Statements              |
| :------------------------------------: | :--------: | :---------------------------------------------: |
| [[Sets and Frozen-Sets\| <- Previous]] | [[Python]] | [[Conditional Statements\| Next ->]] |

There are multiple use cases for sets. They are used, for example, to get rid of doublets - multiple occurrences of elements - in a list, i.e. to make a list unique.

### Different words of a text
To cut out all the words of the novel "Ulysses" we can use the function `findall` from the module "re":
```python
import re
# we don't care about case sensitivity and therefore use lower:
ulysses_txt = open("books/james_joyce_ulysses.txt").read().lower()
words = re.findall(r"\b[\w-]+\b", ulysses_txt)

for word in ["the", "while", "good", "bad", "ireland", "irish"]:
	print("The word '" + word + "' occurs " + \
		str(words.count(word)) + " times in the novel!" )
```

```python
diff_words = set(words)
print("'Ulysses' contains " + str(len(diff_words)) + " different words!")
```
`output: 'Ulysses' contains 29422 different words!`

This is indeed an impressive number. You can see this, if you look at the other novels below:
```python
novels = ['sons_and_lovers_lawrence.txt',
		  'metamorphosis_kafka.txt',
		  'the_way_of_all_flash_butler.txt',
		  'robinson_crusoe_defoe.txt',
		  'to_the_lighthouse_woolf.txt',
		  'james_joyce_ulysses.txt',
		  'moby_dick_melville.txt']
for novel in novels:
	txt = open("books/" + novel).read().lower()
	words = re.findall(r"\b[\w-]+\b", txt)
	diff_words = set(words)
	n = len(diff_words)
	print("{name:38s}: {n:5d}".format(name=novel[:-4], n=n))
```

---

### Special Words in ULYSSES
We will subtract all the words occurring in the other novels from "Ulysses" in the following little Python
program. It is amazing how many words are used by James Joyce and by none of the other authors:
```python
words_in_novel = {}
for novel in novels:
	txt = open("books/" + novel).read().lower()
	words = re.findall(r"\b[\w-]+\b", txt)
	words_in_novel[novel] = words

words_only_in_ulysses = set(words_in_novel['james_joyce_ulysses.t
xt'])
novels.remove('james_joyce_ulysses.txt')
for novel in novels:
	words_only_in_ulysses -= set(words_in_novel[novel])

with open("books/words_only_in_ulysses.txt", "w") as fh:
	txt = " ".join(words_only_in_ulysses)
	fh.write(txt)

print(len(words_only_in_ulysses))
```
`Output: 15314`

---

### Common Words
It is also possible to find the words which occur in every book. To accomplish this, we need the set
intersection:
```python
# we start with the words in ulysses
common_words = set(words_in_novel['james_joyce_ulysses.txt'])
for novel in novels:
	common_words &= set(words_in_novel[novel])

print(len(common_words))
```
`Output: 1745`

The function read_text takes care of removing headers and footers:
```python
def read_text(fname):
	beg_e = re.compile(r"\*\*\* ?start of (this|the) project gutenberg ebook[^*]*\*\*\*")
	end_e = re.compile(r"\*\*\* ?end of (this|the) project gutenberg ebook[^*]*\*\*\*")
	txt = open("books/" + fname).read().lower()
	beg = beg_e.search(txt).end()
	end = end_e.search(txt).start()
	return txt[beg:end]

words_in_novel = {}
for novel in novels + ['james_joyce_ulysses.txt']:
	txt = read_text(novel)
	words = re.findall(r"\b[\w-]+\b", txt)
	words_in_novel[novel] = words

words_in_ulysses = set(words_in_novel['james_joyce_ulysses.txt'])
for novel in novels:
	words_in_ulysses -= set(words_in_novel[novel])
	
with open("books/words_in_ulysses.txt", "w") as fh:
	txt = " ".join(words_in_ulysses)
	fh.write(txt)

print(len(words_in_ulysses))

# we start with the words in ulysses
common_words = set(words_in_novel['james_joyce_ulysses.txt'])
for novel in novels:
	common_words &= set(words_in_novel[novel])

print(len(common_words))
```
`Ouput: 15341 1279`

The words of the set "common_words" are words belong to the most frequently used words of the English
language. Let's have a look at 30 arbitrary words of this set:
```python
counter = 0
for word in common_words:
	print(word, end=", ")
	counter += 1
	if counter == 30:
		break
```
`Output: send, such, mentioned, writing, found, speak, fond, food, their, mother, household, through, prepared, flew, gently, work, station,naturally, near, empty, filled, move, unknown, left, alarm, listening, waited, showed, broke, laugh,`

