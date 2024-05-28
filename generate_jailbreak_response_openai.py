import argparse
import csv

from datasets import load_dataset
from tqdm.auto import tqdm

from llm_adversarial.openai_api.gpt import jailbreak_attempt


def main(
    result_path: str,
    response_start_with: str = "",
    openai_model: str = "gpt-3.5-turbo",
) -> None:
    if response_start_with == "":
        response_start_with = None

    harmbench = load_dataset("csv", data_files="llm_adversarial/dataset/harmbench_behaviors.csv")
    with open(result_path, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=["Prompt", "Prompt_Chinese",
                                "ResponseStartsWith", "FullResponse_Chinese", "SemanticCategory"])
        writer.writeheader()

        for row in tqdm(harmbench["train"], disable=False):
            question_en = row["Behavior"]
            question_zh = row["Behavior_Chinese"]
            response_zh_q = jailbreak_attempt(question_zh, response_start_with, openai_model=openai_model)
            writer.writerow({
                "Prompt": question_en,
                "Prompt_Chinese": question_zh,
                "ResponseStartsWith": response_start_with,
                "FullResponse_Chinese": response_zh_q,
                "SemanticCategory": row["SemanticCategory"],
            })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_path', type=str, required=True)
    parser.add_argument('--response_start_with', type=str, default="")
    parser.add_argument('--openai_model', type=str, default="gpt-3.5-turbo")
    args = parser.parse_args()

    main(**vars(args))
