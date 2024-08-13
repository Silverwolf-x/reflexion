import os
import json
import joblib
import pandas as pd
hotpot = joblib.load('data/hotpot-qa-distractor-sample.joblib').reset_index(drop = True)
print(f'{hotpot =}\n{type(hotpot)=}')
print(hotpot.iloc[0])
qa_pairs = []
with open('data/0604_original_test.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        record = json.loads(line.strip())
        question = record.get('question')
        answer = record.get('answer')
        if question and answer:  # 只添加有 question 和 answer 的记录
            qa_pairs.append({'question': question, 'answer': answer})
qa_pairs = pd.DataFrame(qa_pairs)
qa_pairs.to_csv('data/0604_original_test.csv')
joblib.dump(qa_pairs, 'data/0604_original_test.joblib')
QA = joblib.load('data/0604_original_test.joblib')
print(QA['answer'])