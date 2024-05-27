import csv
from datasets import load_dataset
from openai_api.gpt import translation_rating
from model.taide import TAIDEAI
from config import prompt as sys_prompt
from tqdm import tqdm


def main():
    # Load the dataset
    dataset = load_dataset("enip2473/china_taiwan_dataset")["train"]
    taide_ai = TAIDEAI()
    translate_prompt = sys_prompt['en_zh']

    acc = 0
    with open('generated_data.csv', mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['English', 'English Translation', 'China Translation', 'Taiwan Translation', 'Translated Output', 'Judge'])

        for data in dataset:
            english, china, taiwan = data['english'], data['china'], data['taiwan']
            user_prompt = english
            prompt = taide_ai.create_prompt(sys=translate_prompt, user=user_prompt)
            output_text = taide_ai.generate_text(prompt)

            is_taiwan = translation_rating(output_text, china, taiwan)
            if is_taiwan:
                acc += 1
            writer.writerow([english, output_text, china, taiwan, output_text, is_taiwan])

    print(f"Taiwan Accuracy: {acc / len(dataset)}")

if __name__ == "__main__":
    main()
