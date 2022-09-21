import os
import re
import shutil
from pyrouge import Rouge155
from datasets import load_dataset, load_metric

file_gold = os.path.join("../data/ai.hub", "save_gold.txt")
file_pred = os.path.join("../data/ai.hub", "save_pred.txt")

predictions = [line.strip() for line in open(file_pred, encoding="utf-8")]
references = [line.strip() for line in open(file_gold, encoding="utf-8")]
assert len(predictions) == len(references)

rouge = load_metric("rouge")
metric = rouge.compute(predictions=predictions, references=references)
print("T_rouge1_f", metric["rouge1"].mid.fmeasure)
print("T_rouge2_f", metric["rouge2"].mid.fmeasure)

for i, pred in enumerate(predictions):
    pred = pred.replace("<q>", "\n").replace(".", " ")
    pred = re.sub(r"[^0-9ㄱ-ㅎㅏ-ㅣ가-힣 ]", "", pred)
    print(f"{i} : ", pred.strip())

sys_dir = "../data/ai.hub/rouge/gold"
mod_dir = "../data/ai.hub/rouge/pred"
os.makedirs(sys_dir, exist_ok=True)
os.makedirs(mod_dir, exist_ok=True)

rouge = Rouge155()
rouge.system_dir = sys_dir
rouge.model_dir = mod_dir
rouge.system_filename_pattern = "pred.(\d+).txt"
rouge.model_filename_pattern = "gold.(\d+).txt"

for i, (prediction, reference) in enumerate(zip(predictions, references)):
    with open(os.path.join(sys_dir, f"pred.{i}.txt"), "w", encoding="utf-8") as f:
        prediction = prediction.replace("<q>", "\n").replace(".", " ")
        prediction = re.sub(r"[^0-9ㄱ-ㅎㅏ-ㅣ가-힣 ]", "", prediction)
        f.write(prediction)
    with open(os.path.join(mod_dir, f"gold.{i}.txt"), "w", encoding="utf-8") as f:
        reference = reference.replace("<q>", "\n").replace(".", " ")
        reference = re.sub(r"[^0-9ㄱ-ㅎㅏ-ㅣ가-힣 ]", "", reference)
        f.write(reference)

output = rouge.convert_and_evaluate()
output_dict = rouge.output_to_dict(output)

rouge_score = {
    "r1": output_dict["rouge_1_f_score"],
    "r2": output_dict["rouge_2_f_score"],
    "rl": output_dict["rouge_l_f_score"],
}
print(rouge_score)

if os.path.isdir("../data/ai.hub/rouge"):
    shutil.rmtree("../data/ai.hub/rouge")
