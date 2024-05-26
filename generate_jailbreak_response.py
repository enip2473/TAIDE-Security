import argparse
import csv

from datasets import load_dataset
from transformers import BitsAndBytesConfig
import torch
from tqdm.auto import tqdm

from llm_adversarial.model.taide import TAIDEAI
from llm_adversarial.jailbreak_util import jailbreak_attempt


def main(
    model_path: str,
    result_path: str,
    response_start_with: str = "",
    en_prompt: bool = False,
) -> None:
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    ai = TAIDEAI(model_name=model_path, quantization_config=nf4_config)

    harmbench = load_dataset("csv", data_files="llm_adversarial/dataset/harmbench_behaviors.csv")
    results = []
    if en_prompt:
        key = "Behavior"
    else:
        key = "Behavior_Chinese"

    for user_question in tqdm(harmbench["train"][key], disable=False):
        output_text = jailbreak_attempt(ai, response_start_with, user_question)
        results.append({
            "Prompt": user_question,
            "Response Starts With": response_start_with,
            "Full Response": output_text,
        })

    with open(result_path, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--result_path', type=str, required=True)
    parser.add_argument('--response_start_with', type=str, default="")
    parser.add_argument('--en_prompt', action='store_true')
    args = parser.parse_args()

    main(**vars(args))
