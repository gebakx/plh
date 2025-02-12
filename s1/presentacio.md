class: center, middle

## Processament del Llenguatge Humà

# Lab. 1: Presentació

<br>

### Gerard Escudero, Salvador Medina i Jordi Turmo

## Grau en Intel·ligència Artificial

<br>

![:scale 75%](../fib.png)

---
class: left, middle, inverse

# Sumari

- .cyan[Presentació]

- Recursos lingüístics

- Demostracions

- Exercici

---

# Objectius

* Ús de les tècniques bàsiques del processament de la llengua

* Resoldre problemes petits i mitjans d'informació textual

.cols5050[
.col1[
#### Eines

  - [Google Colab](https://colab.research.google.com) <br> codi i documentació

  - [Google Drive](https://drive.google.com) <br> dades i models

]
.col2[
#### Components:

  - Jupyter 

  - Python3

  - nltk, spaCy i Freeling
]]

---

# Avaluació

- Exercicis

  - Entrenar Hidden Markov Models
  - Similaritats amb WordNet 
  - Anàlisi sintàctica
  - Coreferència

- Pràctiques

  - Identificació de la llengua
  - Detecció opinions: models supervisat i no supervisat
  - Entrenar un model per *Name Entities*
  - Aplicació de Word Vectors

**Items d'avaluació**:

- Eficacia
- Eficiència
- Claredat i organització
- Recursos linguistics
- Anàlisi dels resultats

---
class: left, middle, inverse

# Sumari

- .brown[Presentació]

- .cyan[Recursos lingüístics]

- Demostracions

- Exercici

---

# Natural Language Toolkit

.small[- Llibreria python open-source: [https://www.nltk.org/](https://www.nltk.org/)]

.cols5050[
.col1[
**Exemple:**

.small[
```python3
!pip install svgling
import nltk
import svgling
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
s = 'Mark is working at Google.'
x = nltk.pos_tag(nltk.word_tokenize(s))
t = nltk.ne_chunk(x)
svgling.draw_tree(t)
```
]
.center[![:scale 80%](figures/nltk.png)]
]
.col2[
**Contingut:**

* *Corpus*: Brown corpus (PoS), sentence polarity corpus... 

* *recursos lèxics*: WordNet, SentiWordNet...

* *Gramàtiques*: English, Spanish, ...

* *Models*: Named Entities, taggers...
]]

---

# FreeLing

.small[
- Llibreria C++: [https://nlp.lsi.upc.edu/freeling/](https://nlp.lsi.upc.edu/freeling/)

- Conté una API per connectar via http, [TextServer](https://textserver.lsi.upc.edu/textserver/login).
]

[Demo](https://nlp.lsi.upc.edu/freeling/demo/demo.php):

![:scale 100%](figures/freeling.png)

---

# spaCy

- Llibreria amb model neuronal per python: [https://spacy.io/](https://spacy.io/)

**Exemple:**

```
[('Smith', 'NNP', 'nsubj', jumps),
 ('jumps', 'VBZ', 'ROOT', jumps),
 ('over', 'IN', 'prep', jumps),
 ('the', 'DT', 'det', dog),
 ('lazy', 'JJ', 'amod', dog),
 ('dog', 'NN', 'pobj', over),
 ('.', '.', 'punct', jumps)]
```

![:scale 100%](figures/spacy.png)

---
class: left, middle, inverse

# Sumari

- .brown[Presentació]

- .brown[Recursos lingüístics]

- .cyan[Demostracions]

- Exercici

---

# Traducció automàtica

- https://www.softcatala.org/traductor/ <br><br>
![:scale 90%](figures/softcatala.png)

- https://translate.google.com <br><br>
![:scale 90%](figures/google.png)

---

# *Question Answering*

.cols5050[
.col1[
Q&A Demo: <br>[https://www.tomko.org/demo/](https://www.tomko.org/demo/)

![:scale 90%](figures/qa1.png)<br>
![:scale 105%](figures/qa2.png)<br>
![:scale 105%](figures/qa3.png)<br><br>
![:scale 70%](figures/qa4.png)
]
.col2[
Visual Q&A Demo: [http://visualqa.csail.mit.edu/](http://visualqa.csail.mit.edu/)

![:scale 105%](figures/qa6.png)<br><br>
![:scale 95%](figures/qa5.png)
]]

---

# Resum automàtic

Huggingface Online Demo (Bart):
[https://huggingface.co/tasks/summarization](https://huggingface.co/tasks/summarization)

![:scale 105%](figures/summarization.png)

Font del text: [Wikipedia](https://en.wikipedia.org/wiki/Javier_Milei)

---

# Imatges i llenguatge natural

.cols5050[
.col1[
**Dall-e 2**:

[https://openai.com/index/dall-e-3/](https://openai.com/index/dall-e-3/)

**Example**:

`An astronaut riding a horse in a photorealistic style`

.center[![:scale 80%](figures/dalle2.jpg)]
]
.col2[
**Stable Difusion**: open-source

[https://stablediffusionweb.com/](https://stablediffusionweb.com/)

.center[![:scale 110%](figures/stableDiffusion.png)]
]]

---

# ChatGPT I

Xat de OpenAI: [https://chatgpt.com/](https://chatgpt.com/)

Exemple:

![:scale 90%](figures/chatGPT1.png)

---

# ChatGPT II

Exemple:

![:scale 100%](figures/chatGPT3.png)


---

# Enllaços d'interès

- Huggingface Blog [https://huggingface.co/blog](https://huggingface.co/blog)
- NLP-Progress [https://nlpprogress.com/](https://nlpprogress.com/)
- Papers with Code [https://paperswithcode.com/](https://paperswithcode.com/)
- Research Blogs ([https://research.google/blog/](https://research.google/blog/), [https://research.facebook.com/blog/](https://research.facebook.com/blog/), ...)

---

class: left, middle, inverse

# Sumari

- .brown[Presentació]

- .brown[Recursos lingüístics]

- .brown[Demostracions]

- .cyan[Exercici]

---

# Exercici

Avaluació d'una de les demos:

1. Escolliu una de les demos de la secció anterior

2. Feu unes quantes probes amb la mateixa

3. Analitzeu els resultats


