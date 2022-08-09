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
        # data = data[:10000]
        np.save(data_file, data)

    sample_len = len(data)
    print(f"document samples : {sample_len:,}")

    def process_json(data):
        sentences = data["text"]
        article = [sent[0]["sentence"] for sent in sentences if sent]
        article = " ".join(article)
        extractive = ""
        for idx in data["extractive"]:
            for sent in sentences:
                if sent and sent[0]["index"] == idx:
                    extractive += sent[0]["sentence"] + " "
        abstractive = data["abstractive"][0]
        return {
            "title": data["title"],
            "category": data["category"],
            "media_type": data["media_type"],
            "media_name": data["media_name"],
            "publish_date": data["publish_date"],
            "article": article,
            "extractive": extractive,
            "abstractive": abstractive,
        }

    df = pd.DataFrame([process_json(data[i]) for i in range(sample_len)])
    df["publish_date"] = pd.to_datetime(df["publish_date"])
    df.to_pickle(f"../data/ai.hub/train_{section}_df.pkl")


if __name__ == "__main__":
    sections = ["news", "columns"]
    for section in sections:
        # preprocess_data(section)

        df = pd.read_pickle(f"../data/ai.hub/train_{section}_df.pkl")
        print(df.info())
        print(df.category.value_counts())
        print(df.media_name.value_counts())
