
# Notebook for running Chain-of-Thought with supporting context experiments 

# %%
import sys, os
sys.path.append('..')
root = '../root/'

# %%
import joblib
import numpy as np
from agents import CoTAgent, ReflexionStrategy
from util import summarize_trial, log_trial, save_agents

# %% [markdown]
# #### Load the HotPotQA Sample

# %%
hotpot = joblib.load('../data/hotpot-qa-distractor-sample.joblib').reset_index(drop = True)

hotpot['supporting_paragraphs'] = None
for ind, row in hotpot.iterrows():
    supporting_articles = row['supporting_facts']['title']
    articles = row['context']['title']
    sentences = row['context']['sentences'] 
    supporting_paragraphs = []
    for article in supporting_articles:
        supporting_paragraph = ''.join(sentences[np.where(articles == article)][0])
        supporting_paragraphs.append(supporting_paragraph)
    supporting_paragraphs = '\n\n'.join(supporting_paragraphs)
    hotpot.at[ind, 'supporting_paragraphs'] = supporting_paragraphs

# %% [markdown]
# #### Define the Reflexion Strategy

# %%
print(ReflexionStrategy.__doc__)

# %%
strategy: ReflexionStrategy = ReflexionStrategy.REFLEXION

# %% [markdown]
# #### Initialize a CoTAgent for each question

# %%
from prompts import cot_agent_prompt, cot_reflect_agent_prompt, cot_reflect_prompt
from fewshots import COT, COT_REFLECT
agents = [CoTAgent(row['question'],
                   row['supporting_paragraphs'],
                   row['answer'],
                   agent_prompt=cot_agent_prompt if strategy == ReflexionStrategy.NONE else cot_reflect_agent_prompt,
                   cot_examples=COT,
                   reflect_prompt=cot_reflect_prompt,
                   reflect_examples=COT_REFLECT,
                    ) for _, row in hotpot.iterrows()]

# %% [markdown]
# #### Run `n` trials

# %%
n = 5
trial = 0
log = ''

# %%
for i in range(n):
    for agent in [a for a in agents if not a.is_correct()]:
        agent.run(reflexion_strategy = strategy)
        print(f'Answer: {agent.key}')
    trial += 1
    log += log_trial(agents, trial)
    correct, incorrect = summarize_trial(agents)
    print(f'Finished Trial {trial}, Correct: {len(correct)}, Incorrect: {len(incorrect)}')

# %% [markdown]
# #### Save the result log

# %%
with open(os.path.join(root, 'CoT', 'context', strategy.value, f'{len(agents)}_questions_{trial}_trials.txt'), 'w') as f:
    f.write(log)
save_agents(agents, os.path.join(root, 'CoT', 'context', strategy.value, 'agents'))


