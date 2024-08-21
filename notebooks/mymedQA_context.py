
# Notebook for running Chain-of-Thought with supporting context experiments 

import sys, os
sys.path.append('..')
root = '../root/'

import joblib
import numpy as np
from agents import CoTAgent, ReflexionStrategy
from util import summarize_trial, log_trial, save_agents

# Process jsonl
import json
import pandas as pd
med = []
with open('../data/0604_original_test.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        record = json.loads(line.strip())
        med.append(record)
med = pd.DataFrame(med)

# Load Sample
def process_answer(answer):
    if isinstance(answer,str): return answer
    elif isinstance(answer,list):
        if len(answer)==1: return answer[0]
        else: return '+'.join(answer)
    else: print(answer)
# Q =  "请使用你的已有知识，判断该病人是以下这7个病中的哪一种或多种疾病。脂肪肝，药物性肝损伤，原发性肝癌，酒精性肝病，自身免疫性肝炎，肝囊肿，肝血管瘤。如果病人是多种疾病，用'+'号连接。以下回答都是正确格式：Finish[脂肪肝]。Finish[脂肪肝+药物性肝损伤]。Finish[药物性肝损伤+脂肪肝+自身免疫性肝炎]。"
Q =  "请基于你的专业知识，判断该病人是以下这7个病中的哪一种或多种疾病。脂肪肝，药物性肝损伤，原发性肝癌，酒精性肝病，自身免疫性肝炎，肝囊肿，肝血管瘤。如果病人是多种疾病，用'+'号连接。"
for ind, row in med.iterrows():
    raw_q = row['question']
    context = raw_q.replace('请判断该病人是以下哪种疾病。病人检查如下：','') # 反整理出病情context
    med.at[ind,'context'] = context 
    # TODO:对齐195，199的evidence的提取
    med.at[ind,'QAonly'] = Q+'||'+context
    med.at[ind,'Q'] = Q
    med.at[ind,'process_answer'] = process_answer(row['answer'])

# #### Define the Reflexion Strategy

print(ReflexionStrategy.__doc__)


# strategy: ReflexionStrategy = ReflexionStrategy.REFLEXION
strategy: ReflexionStrategy = ReflexionStrategy.LAST_ATTEMPT_AND_REFLEXION

# #### Initialize a CoTAgent for each question

# from prompts import cot_agent_prompt, cot_reflect_agent_prompt, cot_reflect_prompt
# from fewshots import COT, COT_REFLECT
from promptsmed import cot_agent_prompt, cot_reflect_agent_prompt, cot_reflect_prompt
from fewshotsmed import MED_COT, MED_COT_REFLECT
# 运行1例
med = med.iloc[0:1]

agents = [CoTAgent(med['Q'],
                   med['context'],
                   row['process_answer'],
                   agent_prompt=cot_agent_prompt if strategy == ReflexionStrategy.NONE else cot_reflect_agent_prompt,
                   cot_examples=MED_COT,
                   reflect_prompt=cot_reflect_prompt,
                   reflect_examples=MED_COT_REFLECT,
                    ) for _, row in med.iterrows()]

# #### Run `n` trials

n = 5
trial = 0
log = ''

for i in range(n):
    for agent in [a for a in agents if not a.is_correct()]:
        print(f'Question: {agent.question}')
        agent.run(reflexion_strategy = strategy)
        print(f'Answer: {agent.key}')
    trial += 1
    log += log_trial(agents, trial)
    correct, incorrect = summarize_trial(agents)
    print(f'Finished Trial {trial}, Correct: {len(correct)}, Incorrect: {len(incorrect)}')

import time
formatted_time = time.strftime("%m%d_%H%M", time.localtime())
with open(os.path.join(root, 'CoT', 'context', strategy.value, f'med_context_{formatted_time}_{len(agents)}_questions_{trial}_trials.txt'), 'w') as f:
    f.write(log)
save_agents(agents, os.path.join(root, 'CoT', 'context', strategy.value, 'agents'))


