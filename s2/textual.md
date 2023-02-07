class: center, middle

## Processament del Llenguatge Humà

# Lab.2: nivell textual

<br>

### Gerard Escudero

## Grau en Intel·ligència Artificial

<br>

![:scale 75%](../fib.png)

---
class: left, middle, inverse

# Sumari

- .cyan[Documentació]

  - .cyan[Tokenització]

  - Mesures de similaritat

  - Models de llengua

  - Zones textuals

- Pràctica: nivell textual

  - Identificació d'idioma

---

# Tokenització amb NLTK

### Requeriments

```
import nltk
nltk.download('punkt')
```

### Divisió en frases (*Sentence Splitting*)

```
nltk.sent_tokenize('Men want children. They get relaxed with kids.')

👉 ['Men want children.', 'They get relaxed with kids.']
```

### Tokenitzador (*Tokenizer*)

```
nltk.word_tokenize('Men want children.')

👉 ['Men', 'want', 'children', '.']
```

---

# NLTK en castellà

L'NLTK no té models per al català.

```
source = 'El gato tiene hambre. Está intentando pescar.'
```

### Divisió en frases (*Sentence Splitting*)

```
nltk.sent_tokenize(source, language='spanish')

👉 ['El gato tiene hambre.', 'Está intentando pescar.']
```

### Tokenitzador (*Tokenizer*)

```
[nltk.word_tokenize(s, language='spanish') for s in 
 nltk.sent_tokenize(source, language='spanish')]

👉 [['El', 'gato', 'tiene', 'hambre', '.'], 
    ['Está', 'intentando', 'pescar', '.']]
```

---

# Tokenització amb spaCy (I)

### Requeriments

```
!python -m spacy download ca_core_news_sm
import spacy
nlp = spacy.load('ca_core_news_sm')
```
- Model Anglès: `en_core_web_sm`. No cal baixar el model en aquest cas.

- Model castellà: `es_core_news_sm`

- Model català: `ca_core_news_sm`

- Per tots tres hi ha 4 models: `sm`, `md`, `lg` i `trf`.


### Processament de text

```
source = "L'Arnau té un gos. Se l'estima molt."
doc = nlp(source)
```

---

# Tokenització amb spaCy (II)

### Divisió en frases (*Sentence Splitting*)

```
[s.text for s in doc.sents]

👉 ["L'Arnau té un gos.", "Se l'estima molt."]
```

### Tokenitzador (*Tokenizer*)

```
s = next(doc.sents)
[(token.text, token.is_stop) for token in s]

👉 [("L'", False),
    ('Arnau', False),
    ('té', False),
    ('un', True),
    ('gos', False),
    ('.', False)]
```

---

# Tokenització amb TextServer (FreeLing)

### Requeriments

- Script auxiliar: [textserver.py](../codes/textserver.py)

```
from google.colab import drive
import sys

drive.mount('/content/drive')
sys.path.insert(0, '/content/drive/My Drive/Colab Notebooks/plh')
from textserver import TextServer
```

### Ús

```
ts = TextServer('usuari', 'passwd', 'tokenizer') 

ts.tokenizer("L'Arnau té un gos. Se l'estima molt.")
👉  [["L'", 'Arnau', 'té', 'un', 'gos', '.'], 
     ['Se', "l'", 'estima', 'molt', '.']]
```

---
class: left, middle, inverse

# Sumari

- .cyan[Documentació]

  - .brown[Tokenització]

  - .cyan[Mesures de similaritat]

  - Models de llengua

  - Zones textuals

- Pràctica: nivell textual

  - Identificació d'idioma

---

# Similaritats

Mesures orientades a conjunts de paraules:

.cols5050[
.col1[
$S_{dice}(X,Y)=\frac{2\cdot \vert X \cap Y\vert}{\vert X\vert+\vert Y\vert}$

$S_{jaccard}(X,Y)=\frac{\vert X \cap Y\vert}{\vert X \cup Y\vert}$
]
.col2[
$S_{overlap}(X,Y)=\frac{\vert X \cap Y\vert}{min(\vert X\vert,\vert Y\vert)}$

$S_{cosine}(X,Y)=\frac{\vert X \cap Y\vert}{\sqrt{\vert X\vert\cdot\vert Y\vert}}$
]]

#### Nota: 

$S_{*}\in[0, 1]$ $\rightarrow$ $D = 1 − S$.

#### Exemple: 

```
from nltk.metrics import jaccard_distance

jaccard_distance(set(['The','eats','fish','.']),
                 set(['The','eats','blue','fish','.']))
👉  0.2
```

---
class: left, middle, inverse

# Sumari

- .cyan[Documentació]

  - .brown[Tokenització]

  - .brown[Mesures de similaritat]

  - .cyan[Models de llengua]

  - Zones textuals

- Pràctica: nivell textual

  - Identificació d'idioma

---

# n-grames amb nltk I

#### Bigrames de caràcters:

```
from nltk.collocations import BigramCollocationFinder, TrigramCollocationFinder
```

```
finder = BigramCollocationFinder.from_words('a cat')
[tr for tr in finder.ngram_fd.items()]
👉  
[(('a', ' '), 1), ((' ', 'c'), 1), (('c', 'a'), 1), (('a', 't'), 1)]
```

#### Trigrames de caràcters:

```
finder = BigramCollocationFinder.from_words('a cat')
[tr for tr in finder.ngram_fd.items()]
👉  [(('a', ' ', 'c'), 1), ((' ', 'c', 'a'), 1), (('c', 'a', 't'), 1)]
```

---

# n-grames amb nltk II

#### n-grames amb filtre:

```
sq = 'the cat and the dog of the man are quite'
finder = TrigramCollocationFinder.from_words(sq)
finder.apply_freq_filter(2)
[tr for tr in finder.ngram_fd.items()]
👉  [(('h', 'e', ' '), 3), ((' ', 't', 'h'), 2), (('t', 'h', 'e'), 3)]
```

#### n-grames de paraules:

```
from nltk import word_tokenize
```
```
finder = BigramCollocationFinder.from_words(word_tokenize('the cat and the dog'))
[tr for tr in finder.ngram_fd.items()]
👉  
[(('the', 'cat'), 1),
 (('cat', 'and'), 1),
 (('and', 'the'), 1),
 (('the', 'dog'), 1)]
```

---
class: left, middle, inverse

# Sumari

- .cyan[Documentació]

  - .brown[Tokenització]

  - .brown[Mesures de similaritat]

  - .brown[Models de llengua]

  - .cyan[Zones textuals]

- Pràctica: nivell textual

  - Identificació d'idioma

---

# Beautiful Soup (html)

```Python3
import urllib.request
from bs4 import BeautifulSoup
import re

url = 'https://nlp.lsi.upc.edu/freeling/node/1'
with urllib.request.urlopen(url) as response:
   dt = response.read().decode('utf8')

soup = BeautifulSoup(dt, 'html.parser')
text = re.sub(r'\n+', r'\n', soup.get_text())
print(text)
```
👉
```
Welcome | FreeLing Home Page
      Skip to main content
    
User account menu
Show — User account menu
Hide — User account menu
Log in
FreeLing Home Page
Hooked on a FreeLing
Welcome
Here you can find information about FreeLing, an open source language analysis tool suite, 
...
```

---

# XML 

* Beautiful Soup: `soup = BeautifulSoup(dt, ’xml’)`

* [xml.sax](https://docs.python.org/3.7/library/xml.sax.html)

```
import xml.sax

class ChgHandler(xml.sax.ContentHandler):
    cnt = 1
    mn = (1.0, 'EUR')
    
    def startElement(self, name, attrs):
        if name == "Cube":
            if 'rate' in attrs.keys():
                # print(attrs.getValue('currency'), attrs.getValue('rate'))
                ChgHandler.cnt += 1
                ChgHandler.mn = min(ChgHandler.mn,
                                    (float(attrs.getValue('rate')),
                                     attrs.getValue('currency')))

url = 'http://www.ecb.europa.eu/stats/eurofxref/eurofxref-daily.xml'

parser = xml.sax.make_parser()
parser.setContentHandler(ChgHandler())
parser.parse(url)

ChgHandler.mn  👉  (0.86408, 'GBP')
```

---
class: left, middle, inverse

# Sumari

- .brown[Documentació]

  - .brown[Tokenització]

  - .brown[Mesures de similaritat]

  - .brown[Models de llengua]

  - .brown[Zones textuals]

- .cyan[Pràctica: nivell textual]

  - .cyan[Identificació d'idioma]

---

# Identificació d'idioma (guia)

* Preprocès:

  - Eliminar els digits del text

  - Convertir tot el text a minúscula

  - Substitueix els espais en blanc continus per un de sol

  - Concatena totes les frases amb un espai doble al mig

* Utilitzeu trigrams de caràcters com a model del llenguatge

* Afegiu una tècnica de suavitzat

* Elimineu tots el trigrams que apareguin menys de 5 vegades en el corpus

* Nota: les llibreries *string* i *regular expression* de python us poden ser útils

---

# Identificació d'idioma (pràctica 1)

#### Recursos

* [langId.zip](resources/langId.zip)

#### Enunciat

* Implementeu un identificador d'idioma per les llengües europees: 

  - anglès, castellà, neerlandès, alemany, italià i francès

* Utilitzeu com a dades el [wortschatz leipzig corpora](http://wortschatz.uni-leipzig.de/en/download): <br>
  - 30k frases de cadascuna de les 6 llengües com a *training set*
  - 10k de cadascuna com a *test set*

* Utilitzeu la guia de la transparència anterior i els apunts de teoria i/o el [capítol 4 del llibre de Jurasky](https://web.stanford.edu/~jurafsky/slp3/4.pdf) de la bibliografia

* Doneu la precisió (*accuracy*) i la matriu de confusió

* Analitzeu els resultats





