class: center, middle

## Processament del Llenguatge Humà

# Lab. 5: Paraules - <br> Semàntica - *WordNet*


### Gerard Escudero, Salvador Medina i Jordi Turmo

## Grau en Intel·ligència Artificial

<br>

![:scale 75%](../fib.png)

---
class: left, middle, inverse

# Sumari

- .cyan[WordNet]

- Similaritats

- SentiWordNet

- Exercici

- Pràctica 2: Detecció d'Opinions


---

# WordNet

**Synsets**: contingut, definicions, exemples i lemes

```python3
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn
nltk.download('omw-1.4')

wn.synsets('age','n')
👉  [Synset('age.n.01'),
     Synset('historic_period.n.01'),
     Synset('age.n.03'),
     Synset('long_time.n.01'),
     Synset('old_age.n.01')]
```

```python3
age = wn.synset('age.n.1')

age.definition()  # 👉  'how long something has existed'

age.examples()  # 👉  ['it was replaced because of its age']

[l.name() for l in wn.synset('historic_period.n.01').lemmas()]
# 👉  ['historic_period', 'age']
```

Referència: [http://www.nltk.org/howto/wordnet.html](http://www.nltk.org/howto/wordnet.html)
---

# Hiponímia i hiperonímia

.cols5050[
.col1[
**Relació directa**:

```python3
age.hyponyms()
👉  
[Synset('bone_age.n.01'),
 Synset('chronological_age.n.01'),
 Synset('developmental_age.n.01'),
 Synset('fetal_age.n.01'),
 Synset('mental_age.n.01'),
 Synset('newness.n.01'),
 Synset('oldness.n.01'),
 Synset('oldness.n.02'),
 Synset('youngness.n.01')]

age.hypernyms()
👉
[Synset('property.n.02')]

age.root_hypernyms()
👉
[Synset('entity.n.01')]
```
]
.col2[
**Clausura**:

```python3
hyper = lambda s: s.hypernyms()

list(age.closure(hyper))
👉
[Synset('property.n.02'),
 Synset('attribute.n.02'),
 Synset('abstraction.n.06'),
 Synset('entity.n.01')]

age.tree(hyper)
👉
[Synset('age.n.01'),
 [Synset('property.n.02'),
  [Synset('attribute.n.02'),
   [Synset('abstraction.n.06'), 
   [Synset('entity.n.01')]]]]]
```
]]

---

# Altres relacions

**Antonímia** (*variant*):
```python3
good = wn.synset('good.a.01')
good.lemmas()[0].antonyms()
👉
[Lemma('bad.a.01.bad')]
```

**Relacions de Synsets**:

.cols5050[
.col1[
```python3
rels = getRelations(age)
for rel in rels:
  for s in rels[rel]:
    print(rel, s.name())
👉
hypernyms property.n.02
hyponyms bone_age.n.01
hyponyms chronological_age.n.01
hyponyms developmental_age.n.01
hyponyms fetal_age.n.01
hyponyms mental_age.n.01
hyponyms newness.n.01
hyponyms oldness.n.01
```
]
.col2[
<br><br><br>
```python3
hyponyms oldness.n.02
hyponyms youngness.n.01
attributes immature.a.04
attributes mature.a.03
attributes new.a.01
attributes old.a.01
attributes old.a.02
attributes young.a.01
```
]]

---

# Implementació *getRelations*

```python3
def getRelations(ss):
  lexRels = ['hypernyms', 'instance_hypernyms', 'hyponyms', \
       'instance_hyponyms', 'member_holonyms', 'substance_holonyms', \
       'part_holonyms', 'member_meronyms', 'substance_meronyms', \
       'part_meronyms', 'attributes', 'entailments', 'causes', 'also_sees', \
       'verb_groups', 'similar_tos']

  def getRelValue(ss, name):
    method = getattr(ss, rel)
    return method()

  results = {}
  for rel in lexRels:
    val = getRelValue(ss, rel)
    if val != []:
      results[rel] = val
  return results
```

---
class: left, middle, inverse

# Sumari

- .brown[WordNet]

- .cyan[Similaritats]

- SentiWordNet

- Exercici
- 
- Pràctica 2: Detecció d'Opinions

---

# Similaritats amb WordNet

```python3
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn
nltk.download('omw-1.4')

dog = wn.synset('dog.n.01')
cat = wn.synset('cat.n.01')
```

```python3
dog.lowest_common_hypernyms(cat)  👉  [Synset('carnivore.n.01')]

dog.path_similarity(cat)  # 👉  0.2

dog.lch_similarity(cat)  # 👉  2.0281482472922856

dog.wup_similarity(cat)  # 👉  0.8571428571428571

nltk.download('wordnet_ic')
from nltk.corpus import wordnet_ic
brown_ic = wordnet_ic.ic('ic-brown.dat')

dog.lin_similarity(cat,brown_ic)  👉  0.8768009843733973
```

---
class: left, middle, inverse

# Sumari

- .brown[WordNet]

- .brown[Similaritats]

- .cyan[SentiWordNet]

- Exercici

- Pràctica 2: Detecció d'Opinions

---

# SentiWordnet in NLTK

```python3
import nltk
nltk.download('wordnet')
nltk.download('sentiwordnet')
nltk.download('omw-1.4')
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
```

```python3
# getting the wordnet synset
synset = wn.synset('good.a.1')

# getting the sentiwordnet synset
sentiSynset = swn.senti_synset(synset.name())
```

**Scores**: positiu, negatiu i objectivitat
```python3
sentiSynset.pos_score(), sentiSynset.neg_score(), sentiSynset.obj_score()
# 👉  (0.75, 0.0, 0.25)
```

---
class: left, middle, inverse

# Sumari

- .brown[WordNet]

- .brown[Similaritats]

- .brown[SentiWordNet]

- .cyan[Exercici]

- Pràctica 2: Detecció d'Opinions

---

# Exercici

Donat el conjunt de paraules següent:

- .blue[king, queen, man, woman]

1. Doneu per a cada parell de paraules el seu primer hiperònim comú 

2. Doneu per les distàncies: *Path*, *Leacock-Chodorow*, *Wu-Palmer* i *Lin*

  - Normalitzeu les similaritats quan ho veieu oportú

  - Podeu mostrar els resultats en taules

3. Quina similaritat veieu més adequada per a aquest conjunt de paraules?

---
class: left, middle, inverse

# Sumari

- .brown[WordNet]

- .brown[Similaritats]

- .brown[SentiWordNet]

- .brown[Exercici]

- .cyan[Pràctica 2: Detecció d'Opinions]
---

# Detecció d'opinions (pràctica 2.a)

#### Recursos

* Movie Reviews Corpus

#### Enunciat

* Implementeu un detector d'opinions positives o negatives amb alguns algoritmes d'aprenentatge supervisat de l'sklearn

* Utilitzeu com a dades el Movie Reviews Corpus de l'NLTK

* Dissenyeu i apliqueu un protocol de validació

* Utilitzeu el preprocés que cregueu més convenient: eliminació d'*stop words*, signes de puntuació...

* Utilitzeu el CountVectorizer per representar la informació

* Doneu la precisió (*accuracy*) i les matrius de confusió

* Analitzeu els resultats

---

# NLTK’s Movie Reviews Corpus

**Polarity corpus**: 
- 1000 exemples positius i 1000 negatius

**Requeriments**:

```python3
import nltk
nltk.download('movie_reviews')
from nltk.corpus import movie_reviews as mr
```

**Ús**:

```python3
mr.fileids('pos')[:2]
# 👉  
['pos/cv000_29590.txt',
 'pos/cv001_18431.txt']

len(mr.fileids('neg'))
# 👉 1000

mr.words('pos/cv000_29590.txt')
# 👉
['films', 'adapted', 'from', 'comic', 'books', 'have', ...]
```

---

# CountVectorizer de l'sklearn 

Codificador *bag of words* 

.cols5050[
.col1[
**Exemple**:

- This is the first document.
- This document is the second document.
- And this is the third one.
- Is this the first document?

**Matriu resultant**:

0 1 1 1 0 0 1 0 1 <br>
0 2 0 1 0 1 1 0 1 <br>
1 0 0 1 1 0 1 1 1 <br>
0 1 1 1 0 0 1 0 1 <br>

]
.col2[
**Diccionari**:

| index | word |
|---|---|
| 0 | and |
| 1 | document | 
| 2 | first |
| 3 | is |
| 4 | one |
| 5 | second |
| 6 | the |
| 7 | third |
| 8 | this |
]]

.blue[Referència]: <br>
.footnote[[https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)]

---

# Detecció d'opinions (pràctica 2.b)

#### Enunciat

* Implementeu un detector d'opinions positives o negatives no supervisat
  1. Apliqueu l'UKB per obtenir els synsets de les paraules
  2. Obtingueu els valors SentiWordnet de cada synset

* Utilitzeu com a dades el/els conjunts de test que hagueu utilitzat a la pràctica 2.a

* Penseu en com podeu combinar aquests valors per obtenir un resultat

* Penseu que fareu si el synset no hi és a SentiWordnet

* Penseu quines categories utilitzareu:
  - només adjectius
  - noms, adjectius i adverbis
  - noms, adjectius, verbs i adverbis

* Analitzeu els resultats i compareu-los amb els de la part supervisada




