import tensorflow_hub as hub
import tensorflow as tf
from transformers import BertTokenizer
import pandas as pd

# Завантаження датасету
print("Load dataset")
df = pd.read_csv('notes.csv', nrows=10)

print(df.head())

# Завантаження попередньо навченої моделі BERT
print("Load BERT")
bert_model_hub_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3"
bert_layer = hub.KerasLayer(bert_model_hub_url, trainable=True)

# Ініціалізація токенізатора BERT
print("Init Tokenizer")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# Перетворення текстових нотаток на токени
print("Tokenized notes")
tokenized_notes = [tokenizer.tokenize(note) for note in df['headline_text']]

# Формирування секвенції одиниць та нулів таблиць із 0
print("seq")
input_word_ids = [tokenizer.convert_tokens_to_ids(note) for note in tokenized_notes]
input_mask = [[1] * len(note) for note in input_word_ids]
segment_ids = [[0] * len(note) for note in input_word_ids]
total_length = [len(note) for note in input_word_ids]

# Виклик BERT з використанням завантаженої моделі та вхідних параметрів
print("Init BERT")
pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])

# Додавання додаткових шарів
print("Add layers")
x = tf.keras.layers.Dense(128, activation='relu')(pooled_output)
x = tf.keras.layers.Dropout(0.2)(x)
output = tf.keras.layers.Dense(2, activation='softmax')(x)

# Компіляція моделі
print("Compil model")
model = tf.keras.Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=output)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
print("Start")
model.fit()
