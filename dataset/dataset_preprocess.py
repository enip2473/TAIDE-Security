from datasets import load_dataset, load_from_disk
from translate_api import translate
from datasets import DatasetDict, Dataset

if __name__ == "__main__":
    dataset = load_dataset("toxigen/toxigen-data", "prompts")

    splits = dataset.keys()
    selected_data = {split: dataset[split].select(range(10)) for split in splits}
    
    translated_data = {}
    for split, data in selected_data.items():
        translated_data[split] = []
        for item in data:
            translated_text = translate(item['text'], 'zh-tw')
            print(translated_text)
            translated_data[split].append(translated_text)


    dataset_dict = {}
    for split, data in translated_data.items():
        dataset_dict[split] = Dataset.from_dict({"text": data})

    translated_dataset = DatasetDict(dataset_dict)
    local_cache_dir = "./translated_data"
    translated_dataset.save_to_disk(local_cache_dir)

    dataset_name = "enip2473/toxigen-data-tw"
    translated_dataset.push_to_hub(dataset_name, max_shard_size = '700MB')