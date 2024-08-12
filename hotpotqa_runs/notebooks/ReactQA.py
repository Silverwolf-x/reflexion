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

# %% [markdown]
# #### Load the HotpotQA Sample

# %%
hotpot = joblib.load('../data/hotpot-qa-distractor-sample.joblib').reset_index(drop = True)

# %% [markdown]
# #### Define the Reflexion Strategy

# %%
print(ReflexionStrategy.__doc__)

# %%
strategy: ReflexionStrategy = ReflexionStrategy.REFLEXION

# %% [markdown]
# #### Initialize a React Agent for each question

# %%
agent_cls = ReactReflectAgent if strategy != ReflexionStrategy.NONE else ReactAgent
agents = [agent_cls(row['question'], row['answer']) for _, row in hotpot.iterrows()]

# %% [markdown]
# #### Run `n` trials

# %%
n = 5
trial = 0
log = ''

# %%
for i in range(n):
    for agent in [a for a in agents if not a.is_correct()]:
        if strategy != ReflexionStrategy.NONE:
            agent.run(reflect_strategy = strategy)
        else:
            agent.run()
        print(f'Answer: {agent.key}')
    trial += 1
    log += log_react_trial(agents, trial)
    correct, incorrect, halted = summarize_react_trial(agents)
    print(f'Finished Trial {trial}, Correct: {len(correct)}, Incorrect: {len(incorrect)}, Halted: {len(halted)}')

# %% [markdown]
# #### Save the result log

# %%
with open(os.path.join(root, 'ReAct', strategy.value, f'{len(agents)}_questions_{trial}_trials.txt'), 'w') as f:
    f.write(log)
save_agents(agents, os.path.join('ReAct', strategy.value, 'agents'))


