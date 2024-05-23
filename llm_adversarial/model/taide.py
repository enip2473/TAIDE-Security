from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

class TAIDEAI:
    def __init__(self, model_name="taide/TAIDE-LX-7B-Chat"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            device_map='auto',
        )
        self.text_generator = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

    def generate_text(self, prompt, max_length=500):
        generated_text = self.text_generator(
            prompt, 
            num_return_sequences=1, 
            max_length=max_length, 
            truncation=True,
            return_full_text=False
        )
        return generated_text[0]['generated_text']

    def create_prompt(self, sys, user):
        chat = [
            {"role": "system", "content": sys},
            {"role": "user", "content": f"輸入：{user}\n你的回答："}
        ]
        prompt = self.tokenizer.apply_chat_template(chat, tokenize=False)
        return prompt


if __name__ == "__main__":
    sys_description = "你是一個來自台灣的AI助理，你的名字是 TAIDE，樂於回答使用者的任何問題，會用繁體中文回答問題。"
    user_question = "你好，請你自我介紹你自己！"

    ai = TAIDEAI()
    prompt = ai.create_prompt(sys=sys_description, user=user_question)
    output_text = ai.generate_text(prompt)
    print(output_text)
