class: center, middle

## Processament del Llenguatge Humà

# Lab.9: text - coreferència

### Gerard Escudero

## Grau en Intel·ligència Artificial

<br>

![:scale 75%](../fib.png)

---
class: left, middle, inverse

# Outline

- .cyan[Documentació]

  - .cyan[Coreferència]

- Exercici

---

# Coreferència amb spaCy I

### Requeriments

```python3
!pip install spacy==2.1.0
!python -m spacy download en_core_web_sm
!pip install neuralcoref

import spacy
import neuralcoref

nlp = spacy.load('en_core_web_sm')
neuralcoref.add_to_pipe(nlp)
```

### Ús

```python3
doc = nlp(u'My sister has a dog. She loves him.')

doc._.has_coref  👉  True

doc._.coref_clusters
👉  [My sister: [My sister, She], a dog: [a dog, him]]
```

---

# Coreferència amb spaCy II

### Representació visual

![:scale 95%](figures/neuralcoref.png)


### Referència

* Neural Coreference - Hugging Face <br>
[https://huggingface.co/coref/](https://huggingface.co/coref/)

---
class: left, middle, inverse

# Outline

- .brown[Documentació]

  - .brown[Coreferència]

- .cyan[Exercici]

---

# Exercici

### Dades

* Primer paràgraf d' *Alice’s Adventures in Wonderland* de *Lewis Carroll*:
```
Alice was beginning to get very tired of sitting by her sister on the bank, 
and of having nothing to do: once or twice she had peeped into the book her 
sister was reading, but it had no pictures or conversations in it, ‘and what 
is the use of a book,’ thought Alice ‘without pictures or conversations?’
```

* Referència: <br>
[http://www.gutenberg.org/files/11/11-0.txt](http://www.gutenberg.org/files/11/11-0.txt)

### Enunciat 

* Apliqueu la coreferència d'spaCy sobre el paràgraf anterior

* Mostreu les cadenes de coreferència

* Què en penseu del resultat?



