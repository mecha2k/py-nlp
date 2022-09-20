import os
import shutil
from pyrouge import Rouge155

file_gold = os.path.join("../data/ai.hub", "save_gold.txt")
file_pred = os.path.join("../data/ai.hub", "save_pred.txt")

candidates = [line.strip() for line in open(file_pred, encoding="utf-8")]
references = [line.strip() for line in open(file_gold, encoding="utf-8")]
assert len(candidates) == len(references)

sys_dir = "../data/ai.hub/rouge/gold"
mod_dir = "../data/ai.hub/rouge/pred"
os.makedirs(sys_dir, exist_ok=True)
os.makedirs(mod_dir, exist_ok=True)

rouge = Rouge155()
rouge.system_dir = sys_dir
rouge.model_dir = mod_dir
rouge.system_filename_pattern = "pred.(\d+).txt"
rouge.model_filename_pattern = "gold.(\d+).txt"

for i, (candidate, reference) in enumerate(zip(candidates, references)):
    with open(os.path.join(sys_dir, f"pred.{i}.txt"), "w", encoding="utf-8") as f:
        f.write(candidate.replace("<q>", "\n"))
    with open(os.path.join(mod_dir, f"gold.{i}.txt"), "w", encoding="utf-8") as f:
        f.write(reference.replace("<q>", "\n"))

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
