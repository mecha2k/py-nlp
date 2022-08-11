from transformers import BertModel, DistilBertModel
from kobert_transformers.tokenization_kobert import KoBertTokenizer

bert_model = BertModel.from_pretrained("monologg/kobert")
distilbert_model = DistilBertModel.from_pretrained("monologg/distilkobert")

tokenizer = KoBertTokenizer.from_pretrained("monologg/kobert")
print(tokenizer.tokenize("[CLS] 한국어 모델을 공유합니다. [SEP]"))
# ['[CLS]', '▁한국', '어', '▁모델', '을', '▁공유', '합니다', '.', '[SEP]']
print(
    tokenizer.convert_tokens_to_ids(["[CLS]", "▁한국", "어", "▁모델", "을", "▁공유", "합니다", ".", "[SEP]"])
)
# [2, 4958, 6855, 2046, 7088, 1050, 7843, 54, 3]
