import numpy as np
import pandas as pd
import json
import os


def preprocess_data(section):
    data_file = f"../data/ai.hub/train_{section}.npy"
    if os.path.exists(data_file):
        print("Loading data from numpy_npy file...")
        data = np.load(data_file, allow_pickle=True)
    else:
        with open(f"../data/ai.hub/train_{section}.json", "r", encoding="utf-8") as file:
            json_data = json.load(file)
        print("Loading data from original_json file...")
        print("name : ", json_data["name"])
        print("delivery_date : ", json_data["delivery_date"])
        data = json_data["documents"]
        np.save(data_file, data)

    sample_len = len(data)
    print(f"document samples : {sample_len:,}")

    def process_json(data):
        return {
            "title": data["title"],
            "category": data["category"],
            "media_type": data["media_type"],
            "media_name": data["media_name"],
            "publish_date": pd.to_datetime(data["publish_date"]),
            "text": data["text"],
            "extractive": data["extractive"],
            "abstractive": data["abstractive"][0],
        }

    df = pd.DataFrame([process_json(data[i]) for i in range(sample_len)])
    df.to_pickle(f"../data/ai.hub/train_{section}_df.pkl")


if __name__ == "__main__":
    # for section in ["news", "columns"]:
    #     preprocess_data(section)

    df = pd.read_pickle(f"../data/ai.hub/train_news_small_df.pkl")

    def arrange_article(row):
        sentences = list()
        for text_item in row["text"]:
            for text in text_item:
                sentences.insert(text["index"], text["sentence"])
        return np.array(sentences)

    df["article"] = df.apply(arrange_article, axis=1)
    df["extractive_sents"] = df.apply(lambda row: row["article"][row["extractive"]], axis=1)

    print(df.info())
    print(df.category.value_counts())
    print(df.media_name.value_counts())

    sample = df.sample(n=1)
    for idx, sent in enumerate(sample["article"].values[0]):
        print(idx, " : ", sent)
    print(sample["extractive"].values[0])
    print(sample["extractive_sents"].values[0])
