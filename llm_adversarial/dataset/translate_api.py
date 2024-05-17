from google.cloud import translate_v2 as translate

translate_client = translate.Client()

def translate(text: str, target_language: str = 'en') -> str:
    result = translate_client.translate(text, target_language=target_language)
    translation = result['translatedText']
    return translation

if __name__ == "__main__":
    print(translate("hello World", 'zh-tw'))
