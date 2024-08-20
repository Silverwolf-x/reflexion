import sys, os, json
sys.path.append('..')
root  = '../root/'
import pandas as pd
from util import summarize_react_trial, log_react_trial, save_agents
from agents import ReactReflectAgent, ReactAgent, ReflexionStrategy

# Process jsonl
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
Q =  "请全程使用中文进行回答。请使用你的已有知识，判断该病人是以下这7个病中的哪一种或多种疾病。脂肪肝，药物性肝损伤，原发性肝癌，酒精性肝病，自身免疫性肝炎，肝囊肿，肝血管瘤。如果病人是多种疾病，用'+'号连接。"
for ind, row in med.iterrows():
    raw_q = row['question']
    context = raw_q.replace('请判断该病人是以下哪种疾病。病人检查如下：','') # 反整理出病情context
    med.at[ind,'context'] = context 
    # TODO:对齐195，199的evidence的提取
    med.at[ind,'QAonly'] = Q+'||'+context
    med.at[ind,'Q'] = Q
    med.at[ind,'process_answer'] = process_answer(row['answer'])


# Define the Reflexion Strategy
strategy: ReflexionStrategy = ReflexionStrategy.REFLEXION
agent_cls = ReactReflectAgent if strategy != ReflexionStrategy.NONE else ReactAgent

########### 只运行一例
row = med.iloc[0]
print(row)
agents = [agent_cls(row['QAonly'], row['process_answer'])]

########### 运行全部
# agents = [agent_cls(row['question'], row['answer']) for _, row in hotpot.iterrows()]

# Run `n` trials
n = 5
trial = 0
log = ''


for i in range(n):
    for agent in [a for a in agents if not a.is_correct()]:
        print(f'Question: {agent.question}')
        agent.run(reflect_strategy=strategy)
        print(f'Answer: {agent.key}')
    trial += 1
    log += log_react_trial(agents, trial)
    correct, incorrect, halted = summarize_react_trial(agents)
    print(f'Finished Trial {trial}, Correct: {len(correct)}, Incorrect: {len(incorrect)}, Halted: {len(halted)}')


# #### Save the result log

import time
formatted_time = time.strftime("%m%d_%H%M", time.localtime())
with open(os.path.join(root, 'ReAct', strategy.value, f'med_{len(agents)}_questions_{trial}_trials_{formatted_time}.txt'), 'w') as f:
    f.write(log)
# save_agents(agents, os.path.join('ReAct', strategy.value, 'agents'))


