
class: center, middle

## Processament del Llenguatge Humà

# Lab. 11: Word Embeddings Contextuals

### Gerard Escudero, Salvador Medina i Jordi Turmo

## Grau en Intel·ligència Artificial

<br>

![:scale 75%](../fib.png)

---
class: left, middle, inverse

# Sumari

- .cyan[Embeddings Contextuals]
  - .cyan[Embeddings ELMo]
  - Transformers: BERT
  - Transformer afinat: RoBERTa-STS
  - Representació

- Transformer afinat
  - Transformers: Pipelines
  - Exemple

---
# Embeddings ELMo

### Instal·la les dependencies
```shell
pip install tensorflow tensorflow_hub
```

### Carrega i aplica els embeddings ELMo

```python
import tensorflow as tf
import tensorflow_hub as hub

# Carrega el model ELMo
elmo = hub.load("https://tfhub.dev/google/elmo/3")
```
### Executa el model
```python
sentences = ["Hello, how are you?", "I am doing great!"]
model_output = elmo(sentences, signature="default", as_dict=True)
model_output["elmo"]
# -> (2, max_seq_length, 1024)
```

---

# Embeddings ELMo (II)

Recorda que estem copiant tota l'arquitectura. Per defecte, calculem els embeddings com una mitjana ponderada de la primera, segona i tercera capa. Podem obtenir altres sortides:

#### Amb `as_dict=True`, La sortida conté les següents claus:
- `word_emb`: les representacions de paraules basades en caràcters amb forma [batch_size, max_length, 512].
- `lstm_outputs1`: el primer estat ocult LSTM amb forma [batch_size, max_length, 1024].
- `lstm_outputs2`: el segon estat ocult LSTM amb forma [batch_size, max_length, 1024].
- `elmo`: la suma ponderada de les 3 capes, on els pesos són entrenables. Aquest tensor té forma [batch_size, max_length, 1024]
- `default`: una mitjana fixa de totes les representacions de paraules contextualitzades amb forma [batch_size, 1024].


---

# Transformers: BERT

### Instal·la les dependencies
```shell
pip install transformers
```


### Carrega el model
```python
from transformers import BertTokenizer, TFBertModel

# Carrega el model BERT i el tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = TFBertModel.from_pretrained(model_name)
```
---

# Transformers: BERT (II)

### Tokenitzador BERT
```python
tokenizer(sentences, padding=True, truncation=True,
          return_tensors="tf")["input_ids"]
# -> Tensor de les frases tokenitzades.
# Utilitza padding per a la longitud de seqüència d'entrada més llarga.
# Trunca la sortida si excedeix la longitud màxima de seqüència.
```

### Executa el model BERT i obté els embeddings

```python
def get_bert_embeddings(sentences):
    input_ids = tokenizer(sentences, padding=True, truncation=True,
                          return_tensors="tf")["input_ids"]
    outputs = model(input_ids)[0]
    return outputs

# Exemple d'ús
sentences = ["Hello, how are you?", "I am doing great!"]
embeddings = get_bert_embeddings(sentences)
# -> (2, max_seq_length, 768)
```

---

# Transformer afinat: RoBERTa-STS

Podem utilitzar models afinats per a calcular la similitud de frases.

### Utilitzant la llibreria `sentence_transformers`:
```python
from sentence_transformers import SentenceTransformer

def compute_similarity(sentence1, sentence2):
    # Carrega el model STS pre-entrenat
    model_name = 'stsb-roberta-base'
    model = SentenceTransformer(model_name)

    # Codifica les frases i calcula la similitud del cosinus
    embeddings1 = model.encode([sentence1])
    embeddings2 = model.encode([sentence2])
    similarity = cosine(embeddings1, embeddings2)
    return similarity

# Exemple d'ús
sentence1 = "I like cats"
sentence2 = "I love dogs"
similarity = compute_similarity(sentence1, sentence2)
print(f"Similitud: {similarity}")
```
---

# Representació

```python
import tensorflow as tf
import tensorflow_hub as hub
from tensorboard.plugins import projector
import numpy as np

# Frases d'exemple
sentences = [
    "I like cats", "I love dogs", "Cats are cute", "Dogs are loyal"]
```

```python
# Crea un fitxer de metadades amb les etiquetes de les frases
metadata_path = 'metadata.tsv'
with open(metadata_path, 'w') as f:
    for sentence in sentences:
        f.write(sentence + '\n')
```

### ELMo
```python
# Carrega el model ELMo
elmo = hub.load("https://tfhub.dev/google/elmo/3")

# Obtén els embeddings ELMo per a les frases
elmo_embeddings = elmo(sentences, signature="default", as_dict=True)["elmo"]
elmo_embeddings = tf.reduce_mean(elmo_embeddings, axis=1).numpy()

# Guarda els embeddings
elmo_embeddings_path = 'elmo_embeddings.ckpt'
np.savetxt(elmo_embeddings_path, elmo_embeddings, delimiter='\t')


# Embeddings ELMo
elmo_embedding = config.embeddings.add()
elmo_embedding.tensor_name = elmo_embeddings_path
elmo_embedding.metadata_path = metadata_path

# Configura el projector
config = projector.ProjectorConfig()

# Escriu la configuració del projector a disc
projector_path = 'projector'
summary_writer = tf.summary.create_file_writer(projector_path)
projector.visualize_embeddings(summary_writer, config)

# Inicia TensorBoard
tensorboard_path = 'logs'
```
```bash
tensorboard --logdir=./logs
```

### Per a BERT

```python
# Carrega el model BERT i el tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = TFBertModel.from_pretrained(model_name)

# Obtén els embeddings BERT per a les frases
bert_input = tokenizer(sentences)
bert_embeddings = bert(bert_input)["pooled_output"].numpy()

# Guarda els embeddings
bert_embeddings_path = 'bert_embeddings.ckpt'
np.savetxt(bert_embeddings_path, bert_embeddings, delimiter='\t')

# Embeddings BERT
bert_embedding = config.embeddings.add()
bert_embedding.tensor_name = bert_embeddings_path
bert_embedding.metadata_path = metadata_path
```
---

# Transfer-learning amb BERT

La llibreria `transformers` defineix diverses arquitectures predefinides per a diferents tasques de downstream.
Podem utilitzar aquestes arquitectures i fer transfer-learning a partir d'un model BERT de base.


### Transfer-learning per a NER
```python
from transformers import TFBertForTokenClassification, BertTokenizer
# Carrega el model NER pre-entrenat i el tokenizer
model_name = "bert-base-cased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = TFBertForTokenClassification.from_pretrained(model_name, num_labels=NER_CLASSES)
```
---

# Exemple: Transfer-learning per a NER

### Carrega els requisits i el conjunt de dades
```python
import nltk
import tensorflow as tf
from transformers import BertTokenizer, TFBertForTokenClassification
from sklearn.model_selection import train_test_split

# Descarrega i carrega el corpus CoNLL 2002
nltk.download('conll2002')
from nltk.corpus import conll2002

# Carrega el corpus
corpus = conll2002.iob_sents()
```
---

### Prepara la entrada
```python
# Prepara les dades d'entrada en el format requerit per a NER
def prepare_input(corpus):
    sentences, labels = [], []
    for tagged_sentence in corpus:
        sentence, tag = zip(*tagged_sentence)
        sentences.append(" ".join(sentence))
        labels.append(list(tag))
    return sentences, labels
sentences, labels = prepare_input(corpus)
```

### Divisió:
```python
# Divideix les dades en conjunts d'entrenament i validació
train_sentences, val_sentences, train_labels, val_labels = train_test_split(
    sentences, labels, test_size=0.2, random_state=42
)
```
---

### Tokenització:

```python
model_name = "bert-base-cased"
tokenizer = BertTokenizer.from_pretrained(model_name)
# Tokenitza i converteix les frases al format d'entrada
train_encodings = tokenizer(train_sentences, truncation=True, padding=True)
val_encodings = tokenizer(val_sentences, truncation=True, padding=True)
```

### Converteix seqüències d'etiquetes a IDs d'etiquetes
```python
def convert_labels_to_ids(labels, label_map):
    label_ids = []
    for sent_labels in labels:
        label_ids.append([label_map[label] for label in sent_labels])
    return label_ids
# Defineix la correspondència d'etiquetes. Ordena-ho per tenir O abans. I després B abans que I.
labels_set = set(sum(labels, []))
label_list = sorted(labels_set, key=lambda e: "0" if e == "O" else e[2]+e[0])
label_map = {label: i for i, label in enumerate(label_list)}
# Converteix les etiquetes a IDs d'etiquetes
train_label_ids = convert_labels_to_ids(train_labels, label_map)
val_label_ids = convert_labels_to_ids(val_labels, label_map)
```

---

### Obté els datasets
```python
train_label_ids = tf.keras.preprocessing.sequence.pad_sequences(
    train_label_ids, padding="post", truncating="post", maxlen=tokenizer.model_max_length)
train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    train_label_ids
))
val_label_ids = tf.keras.preprocessing.sequence.pad_sequences(
    val_label_ids, padding="post", truncating="post", maxlen=tokenizer.model_max_length)
val_dataset = tf.data.Dataset.from_tensor_slices((
    dict(val_encodings),
    val_label_ids
))
```

---

### Carrega i compila el model
```python
model = TFBertForTokenClassification.from_pretrained(model_name, num_labels=len(label_map))
# Prepare optimizer
from transformers import create_optimizer
TOTAL_EPOCHS = 3
optimizer, schedule = create_optimizer(init_lr=2e-5, num_warmup_steps=0, num_train_steps=TOTAL_EPOCHS)
# Compile
model.compile(optimizer=optimizer, metrics=[tf.keras.metrics.sparse_categorical_accuracy, ] )
```

### Entrena el Model
```python
model.fit(train_dataset.shuffle(1000).batch(16),
          validation_data=val_dataset.batch(16),
          epochs=3)
```
---

### Inferència:

```python
# Inferència amb el model entrenat
def predict_ner(sentence):
    inputs = tokenizer(sentence, truncation=True, padding=True, return_tensors="tf")
    outputs = model(inputs)[0]
    predicted_ids = tf.argmax(outputs, axis=-1).numpy()[0]
    predicted_labels = [label_list[label_id] for label_id in predicted_ids]
    return predicted_labels
sentence = "OpenAI és una empresa d'intel·ligència artificial."
predicted_labels = predict_ner(sentence)
```

