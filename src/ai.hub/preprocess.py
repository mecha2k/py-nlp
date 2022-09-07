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

    def extract_sentence(row):
        extractive_sentences = ""
        for idx in row["extractive"]:
            for sent in row["text"]:
                if sent and sent[0]["index"] == idx:
                    extractive_sentences += sent[0]["sentence"] + " "
        return extractive_sentences

    df["article"] = df.apply(
        lambda row: " ".join([sent[0]["sentence"] for sent in row["text"] if sent]), axis=1
    )
    df["extractive_sentence"] = df.apply(extract_sentence, axis=1)

    samples = df.sample(n=1)
    text_list = samples["text"].values[0]
    print(samples["extractive"].values[0])
    extract_sents = list()
    for text_item in text_list:
        for text in text_item:
            print(text["index"], " : ", text["sentence"])

    for idx in samples["extractive"].values[0]:
        for text_item in text_list:
            for text in text_item:
                if text["index"] == idx:
                    extract_sents.append(text["sentence"])
    print("extractive sentences : ", extract_sents)

    # df["extractive_sents"] = df.apply(lambda row: list(np.array(row[])), axis=1)
    # print(df.info())
    # print(df.category.value_counts())
    # print(df.media_name.value_counts())
    #
    print("article : ", samples["article"].values[0])
    print("extractive : ", samples["abstractive"].values[0])
    print("abstractive : ", samples["extractive_sentence"].values[0])
