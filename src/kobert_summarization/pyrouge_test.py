import os
import re
import shutil
from pyrouge import Rouge155
from datasets import load_metric

file_gold = os.path.join("../data/ai.hub", "save_gold.txt")
file_pred = os.path.join("../data/ai.hub", "save_pred.txt")

predictions = [line.strip() for line in open(file_pred, encoding="utf-8")]
references = [line.strip() for line in open(file_gold, encoding="utf-8")]
assert len(predictions) == len(references)

REMAP = {
    "-lrb-": "(",
    "-rrb-": ")",
    "-lcb-": "{",
    "-rcb-": "}",
    "-lsb-": "[",
    "-rsb-": "]",
    "``": '"',
    "''": '"',
}


def clean_text(text):
    text = text.replace("<q>", "\n").replace(".", " ")
    text = re.sub(r"[^0-9ㄱ-ㅎㅏ-ㅣ가-힣 ]", "", text)
    text = re.sub(
        r"-lrb-|-rrb-|-lcb-|-rcb-|-lsb-|-rsb-|``|''", lambda m: REMAP.get(m.group()), text
    )
    return text.strip()


preds, refrs = [], []
for i, (pred, refr) in enumerate(zip(predictions, references)):
    preds.append(clean_text(pred))
    refrs.append(clean_text(refr))

rouge = load_metric("rouge")
metric = rouge.compute(predictions=preds, references=refrs)
print("rouge1_f", metric["rouge1"].mid.fmeasure)
print("rouge2_f", metric["rouge2"].mid.fmeasure)
print("rougeL_f", metric["rougeL"].mid.fmeasure)

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
        f.write(prediction.strip())
        print(f"{i} : ", prediction.strip())
    with open(os.path.join(mod_dir, f"gold.{i}.txt"), "w", encoding="utf-8") as f:
        reference = reference.replace("<q>", "\n").replace(".", " ")
        reference = re.sub(r"[^0-9ㄱ-ㅎㅏ-ㅣ가-힣 ]", "", reference)
        f.write(reference.strip())
        print(f"{i} : ", reference.strip())


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
