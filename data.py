import json

# Зчитуємо файли
with open("train-v2.0.json", "r") as f_train:
    train_data = json.load(f_train)

with open("dev-v2.0.json", "r") as f_dev:
    dev_data = json.load(f_dev)

# Виводимо приклад питання з тренувального набору даних
print(train_data["data"][0]["paragraphs"][0]["qas"][0]["question"])
print(dev_data["data"][0]["paragraphs"][0]["qas"][0]["question"])
