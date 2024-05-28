import argparse
import csv

from tqdm.auto import tqdm

from llm_adversarial.openai_api.gpt import jailbreak_classify


def main(
    response_file: str,
    result_file: str,
    openai_model: str = "gpt-3.5-turbo"
):
    with open(response_file, 'r') as rd_f, open(result_file, 'w') as wr_f:
        reader = csv.DictReader(rd_f)
        writer = csv.DictWriter(wr_f, fieldnames=["Prompt", "FullResponse_Chinese", "SucessfulJailbreak"])
        writer.writeheader()

        for row in tqdm(reader, total=200):
            behavior = row["Prompt_Chinese"]
            model_response = row["FullResponse_Chinese"]
            gpt_response = jailbreak_classify(behavior, model_response, openai_model)
            writer.writerow({
                "Prompt": behavior,
                "FullResponse_Chinese": model_response,
                "SucessfulJailbreak": gpt_response
            })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("response_file", type=str, help="The file containing the responses to be evaluated.")
    parser.add_argument("result_file", type=str, help="The file to write the evaluation results to.")
    parser.add_argument("--openai_model", type=str, default="gpt-3.5-turbo",
                        help="The OpenAI model to use for classification.")
    args = parser.parse_args()

    main(**vars(args))
