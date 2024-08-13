# %% [markdown]
# #### Notebook for running React experiments

# %%
import sys, os
sys.path.append('..')
root  = '../root/'

# %%
import joblib
from util import summarize_react_trial, log_react_trial, save_agents
from agents import ReactReflectAgent, ReactAgent, ReflexionStrategy

# Load the HotpotQA Sample
hotpot = joblib.load('../data/hotpot-qa-distractor-sample.joblib').reset_index(drop = True)
# Define the Reflexion Strategy
strategy: ReflexionStrategy = ReflexionStrategy.REFLEXION
agent_cls = ReactReflectAgent if strategy != ReflexionStrategy.NONE else ReactAgent

########### 只运行一例
row = hotpot.iloc[0]
agents = [agent_cls(row['question'], row['answer'])]

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


with open(os.path.join(root, 'ReAct', strategy.value, f'{len(agents)}_questions_{trial}_trials.txt'), 'w') as f:
    f.write(log)
save_agents(agents, os.path.join('ReAct', strategy.value, 'agents'))


