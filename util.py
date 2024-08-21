import os
import joblib

def summarize_trial(agents):
    ### 更改判断方式，让不是correct的统统判到incorrect，从而输出log
    # correct = [a for a in agents if a.is_correct()]
    # incorrect = [a for a in agents if a.is_finished() and not a.is_correct()]
    correct,incorrect = [],[]
    for a in agents:
        if a.is_correct():
            correct.append(a)
        else:
            incorrect.append(a)

    return correct, incorrect

def remove_fewshot(prompt: str) -> str:
    # 中文化，对应的是promptsmed.py
    prefix = prompt.split('下面是一些示例：')[0]
    suffix = prompt.split('（示例结束）')[1]
    return prefix.strip('\n').strip() + '\n' +  suffix.strip('\n').strip()

def log_trial(agents, trial_n):
    correct, incorrect = summarize_trial(agents)

    log = f"""
########################################
BEGIN TRIAL {trial_n}
Trial summary: Correct: {len(correct)}, Incorrect: {len(incorrect)}
#######################################
"""

    log += '------------- BEGIN CORRECT AGENTS -------------\n\n'
    for agent in correct:
        log += remove_fewshot(agent._build_agent_prompt()) + f'\nCorrect answer: {agent.key}\n\n'

    log += '------------- BEGIN INCORRECT AGENTS -----------\n\n'
    for agent in incorrect:
        log += remove_fewshot(agent._build_agent_prompt()) + f'\nCorrect answer: {agent.key}\n\n'

    return log

def summarize_react_trial(agents):
    correct = [a for a in agents if a.is_correct()]
    halted = [a for a in agents if a.is_halted()]
    incorrect = [a for a in agents if a.is_finished() and not a.is_correct()]
    return correct, incorrect, halted

def log_react_trial(agents, trial_n):
    correct, incorrect, halted = summarize_react_trial(agents)

    log = f"""
########################################
BEGIN TRIAL {trial_n}
Trial summary: Correct: {len(correct)}, Incorrect: {len(incorrect)}, Halted: {len(halted)}
#######################################
"""

    log += '------------- BEGIN CORRECT AGENTS -------------\n\n'
    for agent in correct:
        log += remove_fewshot(agent._build_agent_prompt()) + f'\nCorrect answer: {agent.key}\n\n'

    log += '------------- BEGIN INCORRECT AGENTS -----------\n\n'
    for agent in incorrect:
        log += remove_fewshot(agent._build_agent_prompt()) + f'\nCorrect answer: {agent.key}\n\n'

    log += '------------- BEGIN HALTED AGENTS -----------\n\n'
    for agent in halted:
        log += remove_fewshot(agent._build_agent_prompt()) + f'\nCorrect answer: {agent.key}\n\n'

    return log

def save_agents(agents, dir: str):
    os.makedirs(dir, exist_ok=True)
    for i, agent in enumerate(agents):
        print(f'\n{agent=}\n{type(agent)=}\n')
        joblib.dump(agent, os.path.join(dir, f'{i}.joblib'))