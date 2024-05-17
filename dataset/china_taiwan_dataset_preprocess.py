import json
from datasets import Dataset
from translate_api import translate
def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

if __name__ == "__main__":
    file_path = 'china_taiwan.json'

    datas = read_json_file(file_path)

    datas = [data for data in datas if data['text_trad'] != data['text_tw']]
    datas = [{
        'china': data['text_trad'], 
        'taiwan': data['text_tw'],
        'english': translate(data['text_tw'], 'en')
    } for data in datas]
    dataset = Dataset.from_list(datas)

    dataset.save_to_disk('china_taiwan_dataset')

    dataset.push_to_hub('enip2473/china_taiwan_dataset')
