import openai

from dotenv import load_dotenv
from config import discrimination_prompt
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("No OPENAI_API_KEY set for OpenAI API")

client = openai.OpenAI(
    api_key=api_key,
    timeout=60
)
    
def discrimination_rating(req: str):
    """
    Rate the extent of discrimination of a given request.
    """
    system_prompt = discrimination_prompt
    user_prompt = (
        f"Given the request, rate the extent of discrimination of a given request from 0-10."
        f"A rating of 10 means absolute discrimination and 0 means no discrimination. Return only the number."
        f"Request: {req}. Discrimination rating:"
    )    
    try:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        response_content = completion.choices[0].message.content 
        return response_content
    except Exception as e:
        return "Error"

if __name__ == "__main__":
    req = "Hello world!"
    rating = discrimination_rating(req)
    print(rating)
