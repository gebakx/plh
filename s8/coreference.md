class: center, middle

## Processament del Llenguatge Humà

# Lab. 10: Text - Coreferència

### Gerard Escudero, Salvador Medina i Jordi Turmo

## Grau en Intel·ligència Artificial

<br>

![:scale 75%](../fib.png)

---
class: left, middle, inverse

# Sumari

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

# Coreferència amb spaCy (Alt)
Podeu fer servir el mòdul experimental

```python
!pip install spacy-experimental
!pip install https://github.com/explosion/spacy-experimental/releases/download/v0.6.1/en_coreference_web_trf-3.4.0a2-py3-none-any.whl
# Python
import spacy
nlp = spacy.load("en_coreference_web_trf")
doc = nlp("The cats were startled by the dog as it growled at them.") 
print(doc.spans)
```

---

# Coreferència amb spaCy (Alt) (II)

### Això permet fer servir el pipeline de training inclòs a spaCy
#### Per fer això, heu definir les coreferències com a spans
```python
nlp = spacy.load('en_core_web_sm')
coref = nlp.add_pipe("experimental_coref")
train_data = [
     (
            "Yes, I noticed that many friends around me received it. It seems that almost everyone received this SMS.",
            {
                "spans": {
                    f"coref_clusters_1": [
                        (5, 6, "MENTION"),      # I
                        (40, 42, "MENTION"),    # me

                    ],
                    f"coref_clusters_2": [
                        (52, 54, "MENTION"),     # it
                        (95, 103, "MENTION"),    # this SMS
                    ]
                }
            },
        ),
]
```



---

# Coreferència amb spaCy (Alt) (III)

### Entrenar el model de Coreferència

```python
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "experimental_coref"]
with nlp.disable_pipes(*other_pipes):
    # Train the model
    optimizer = nlp.initialize()
    for i in range(100):
        random.shuffle(train_data)
        for text, clusters in train_data:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, clusters)
            loss = nlp.update([example], sgd=optimizer)
            print("Loss", loss)
```
### Executar el model obtingut
```python
doc = nlp( "Yes, I noticed that many friends around me received it. It seems that almost everyone received this SMS.",)
print(doc.spans)
```

---

# Coreferència amb Textserver

### Requeriments

```python3
from google.colab import drive
import sys
drive.mount('/content/drive')
sys.path.insert(0, '/content/drive/My Drive/Colab Notebooks/plh')
from textserver import TextServer
```

### Ús

```python3
ts = TextServer('usuari', 'passwd', 'coreferences')

ts.coreferences("My sister has a dog. She loves him.")
👉  [['My sister', 'him'], ['a dog', 'She']]
```

---
class: left, middle, inverse

# Sumari

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

* Apliqueu les coreferències d'spaCy i TextServer sobre el paràgraf anterior

* Mostreu les cadenes de coreferència

* Compareu els resultats.

* Què en penseu del resultats?

### Opcional

* Genereu exemples i entreneu el model de coreferència de spaCy per a una altra llengua



