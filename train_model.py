import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
from transformers import BertTokenizer

# Завантаження датасету
df = pd.read_csv('notes.csv')

print(df.head())

# Ініціалізація токенізатора BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# Перетворення текстових нотаток на токени
tokenized_notes = [tokenizer.tokenize(note) for note in df['headline_text']]

# Кількість дослідняємих токенів (слів)
max_length = len(max(df['headline_text'], key=len))

# Формирування секвенції одиниць та нулів таблиць із 0
input_word_ids = [tokenizer.convert_tokens_to_ids(note) for note in tokenized_notes]
input_mask = [[1] * len(note) for note in input_word_ids]
segment_ids = [[0] * len(note) for note in input_word_ids]
total_length = [len(note) for note in input_word_ids]

# Додання відсутніх значень
for i, row in enumerate(total_length):
    difference = max_length - row
    segment_ids[i] += difference * [0]
    input_word_ids[i] += difference * [0]
    input_mask[i] += difference * [0]

# print(len(input_word_ids[50001]))
# print(len(input_mask[50001]))
# print(len(total_length[50001]))
# print(len(segment_ids[50001]))
# print(max_length)
# print(total_length[0])
# print(max_length)

bert_model_hub_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3"
bert_layer = hub.KerasLayer(bert_model_hub_url, trainable=True)

# Визначення вхідної форми для мережі
max_seq_length = max_length
input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                       name="input_word_ids")
input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                   name="input_mask")
segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                    name="segment_ids")

# Виклик BERT з використанням завантаженої моделі та вхідних параметрів
pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])

# Додавання додаткових шарів
x = tf.keras.layers.Dense(128, activation='relu')(pooled_output)
x = tf.keras.layers.Dropout(0.2)(x)
output = tf.keras.layers.Dense(2, activation='softmax')(x)

# Компіляція моделі
model = tf.keras.Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=output)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit([input_word_ids, input_mask, segment_ids], [df['category']],
          epochs=2, batch_size=2, verbose=1)
