import csv
from datasets import load_dataset
from openai_api.gpt import discrimination_rating
from model.taide import TAIDEAI
from config import prompt as sys_prompt
from tqdm import tqdm

def main():
    # Load the dataset
    dataset = load_dataset("enip2473/toxigen-data-tw")
    taide_ai = TAIDEAI()

    tasks = ['summarize', 'letter', 'article', 'en_zh', 'zh_en']

    for task in tasks:
        system_prompt = sys_prompt[task]
        with open(f'../result/{task}.csv', mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            # Write the header row
            writer.writerow(['Prompt', 'Response', 'Discrimination Rating'])

            for key in dataset:
                if "neutral" in key:
                    continue
                for example in tqdm(dataset[key]):
                    user_prompt = example['text']
                    prompt = taide_ai.create_prompt(sys=system_prompt, user=user_prompt)
                    output_text = taide_ai.generate_text(prompt)
                    rating = discrimination_rating(output_text)
                    
                    # Write the results to the CSV file
                    writer.writerow([user_prompt, output_text, rating])

if __name__ == "__main__":
    main()
