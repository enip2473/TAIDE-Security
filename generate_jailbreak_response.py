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
    quantized: bool = False,

) -> None:
    if quantized:
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16
        )
    else:
        nf4_config = None
    ai = TAIDEAI(model_name=model_path, quantization_config=nf4_config)

    harmbench = load_dataset("csv", data_files="llm_adversarial/dataset/harmbench_behaviors.csv")
    f = open(result_path, 'w')
    writer = csv.DictWriter(f, fieldnames=["Prompt", "Prompt_Chinese",
                            "ResponseStartsWith", "FullResponse_Chinese", "SemanticCategory"])
    writer.writeheader()

    for row in tqdm(harmbench["train"], disable=False):
        question_en = row["Behavior"]
        question_zh = row["Behavior_Chinese"]
        # response_en_q = jailbreak_attempt(ai, response_start_with, question_en)
        response_zh_q = jailbreak_attempt(ai, response_start_with, question_zh)
        writer.writerow({
            "Prompt": question_en,
            "Prompt_Chinese": question_zh,
            "ResponseStartsWith": response_start_with,
            # "FullResponse": response_en_q,
            "FullResponse_Chinese": response_zh_q,
            "SemanticCategory": row["SemanticCategory"],
        })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--result_path', type=str, required=True)
    parser.add_argument('--response_start_with', type=str, default="")
    parser.add_argument('--quantized', action='store_true')
    args = parser.parse_args()

    main(**vars(args))
