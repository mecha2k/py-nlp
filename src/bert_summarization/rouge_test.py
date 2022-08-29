import pyrouge
import shutil
import os

from time import strftime, localtime


def rouge_score(candidate_file, reference_file):
    candidates = [line.strip() for line in open(candidate_file, encoding="utf-8")]
    references = [line.strip() for line in open(reference_file, encoding="utf-8")]
    assert len(candidates) == len(references)

    rouge_dir = "../data/cnn_daily/rouge"
    os.makedirs(rouge_dir, exist_ok=True)
    current_time = strftime("%Y-%m-%d-%H-%M-%S", localtime())
    rouge_dir = os.path.join(rouge_dir, f"rouge-{current_time}")
    os.makedirs(rouge_dir, exist_ok=True)
    os.makedirs(rouge_dir + "/candidate", exist_ok=True)
    os.makedirs(rouge_dir + "/reference", exist_ok=True)

    try:
        for i in range(len(candidates)):
            if len(references[i]) < 1:
                continue
            with open(
                rouge_dir + f"/candidate/candidate.{i:02d}.txt", "w", encoding="utf-8"
            ) as file:
                file.write(candidates[i].replace("<q>", "\n"))
            with open(
                rouge_dir + f"/reference/reference.{i:02d}.txt", "w", encoding="utf-8"
            ) as file:
                file.write(references[i].replace("<q>", "\n"))

        rouge = pyrouge.Rouge155()
        rouge.model_dir = rouge_dir + "/reference/"
        rouge.system_dir = rouge_dir + "/candidate/"
        rouge.model_filename_pattern = "reference.#ID#.txt"
        rouge.system_filename_pattern = r"candidate.(\d+).txt"
        rouge_results = rouge.convert_and_evaluate()
        print(rouge_results)
        results = rouge.output_to_dict(rouge_results)
    finally:
        pass
        if os.path.isdir(rouge_dir):
            shutil.rmtree(rouge_dir)
    return results


results = rouge_score("../data/cnn_daily/save_pred.txt", "../data/cnn_daily/save_gold.txt")
print(results)
