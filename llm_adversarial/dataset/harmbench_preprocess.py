import pandas as pd
import sys
from translate_api import translate


df = pd.read_csv('harmbench_behaviors_text_all_standard_en.csv')

df['Behavior_Chinese'] = df['Behavior'].apply(lambda x: translate(x, target_language='zh-tw'))

df.to_csv('harmbench_behaviors.csv', index=False)

print("Translation complete and file saved.")
