from langchain.prompts import PromptTemplate


# TODO: 基于Microsoft提出的Medprompt？。https://arxiv.org/pdf/2311.16452


MED_COT_INSTRUCTION =   """你是一个肝病疑难杂症方面的医学专家，擅长通过自我反省，详细分析病例内容来诊断疾病。请基于你的专业知识，通过"思考"完成答题任务，然后给出答案。"思考"可以对当前情况进行推理，基于专业知识，结合“相关文本”来回答问题。最后使用"完成[<答案>]"的格式回答。
下面是一些示例：
{examples}
（示例结束）
{reflections}
相关文本：{context}
问题：{question}{scratchpad}"""

MED_COT_AGENT_REFLECT_INSTRUCTION  = """你是一个肝病疑难杂症方面的医学专家，擅长通过自我反省，详细分析病例内容来诊断疾病。请基于你的专业知识，通过"思考"完成答题任务，然后给出答案。"思考"可以对当前情况进行推理，基于专业知识，结合“相关文本”来回答问题。最后使用"完成[<答案>]"的格式回答。
下面是一些示例：
{examples}
（示例结束）

{reflections}

相关文本：{context}
问题：{question}{scratchpad}"""

MED_COT_REFLECT_INSTRUCTION  ="""你是一个肝病疑难杂症方面的医学专家，擅长通过自我"反思"，详细分析病例内容来诊断疾病。我会给你"上次试验"，在"上次试验"中，你获得了“相关文本”和“问题”。您"上次试验"回答失败的原因可能是您"完成[<答案>]"回答错误，也可能是您并没有按照这个措辞回答。请使用几句完整的句子，诊断出失败或措辞不一致的可能原因，并制定一个简洁、高水平的新计划，作为"反思"的内容，以减少同样的失败。 
下面是一些示例：
{examples}
（示例结束）
{context}
上次试验:
问题: {question}{scratchpad}

反思:
"""

cot_agent_prompt = PromptTemplate(
                        input_variables=["examples", "reflections", "context", "question", "scratchpad"],
                        template = MED_COT_INSTRUCTION,
                        )

cot_reflect_agent_prompt = PromptTemplate(
                        input_variables=["examples", "reflections", "context", "question", "scratchpad"],
                        template = MED_COT_AGENT_REFLECT_INSTRUCTION,
                        )

cot_reflect_prompt = PromptTemplate(
                        input_variables=["examples", "context", "question", "scratchpad"],
                        template = MED_COT_REFLECT_INSTRUCTION,
                        )


# COT_SIMPLE_INSTRUCTION = """Solve a question answering task by having a Thought, then Finish with your answer. Thought can reason about the current situation. Finish[answer] returns the answer and finishes the task.
# Here are some examples:
# {examples}
# (END OF EXAMPLES)
# {reflections}
# {context}
# Question: {question}{scratchpad}"""

# COT_SIMPLE_AGENT_REFLECT_INSTRUCTION = """Solve a question answering task by having a Thought, then Finish with your answer. Thought can reason about the current situation. Finish[answer] returns the answer and finishes the task.
# Here are some examples:
# {examples}
# (END OF EXAMPLES)
# {context}
# {reflections}

# Question: {question}{scratchpad}"""

# COT_SIMPLE_REFLECT_INSTRUCTION = """You are an advanced reasoning agent that can improve based on self refection. You will be given a previous reasoning trial in which you were given a question to answer. You were unsuccessful in answering the question either because you guessed the wrong answer with Finish[<answer>] or there is a phrasing discrepancy with your provided answer and the answer key. In a few sentences, Diagnose a possible reason for failure or phrasing discrepancy and devise a new, concise, high level plan that aims to mitigate the same failure. Use complete sentences.
# Here are some examples:
# {examples}
# (END OF EXAMPLES)
# {context}
# Previous trial:
# Question: {question}{scratchpad}

# Reflection:"""

# cot_simple_agent_prompt = PromptTemplate(
#                         input_variables=["examples", "question", "reflections", "context", "scratchpad"],
#                         template = COT_SIMPLE_INSTRUCTION,
#                         )

# cot_simple_reflect_agent_prompt = PromptTemplate(
#                         input_variables=["examples", "context", "reflections", "question", "scratchpad"],
#                         template = COT_SIMPLE_AGENT_REFLECT_INSTRUCTION,
#                         )

# cot_simple_reflect_prompt = PromptTemplate(
#                         input_variables=["examples", "question", "context", "scratchpad"],
#                         template = COT_SIMPLE_REFLECT_INSTRUCTION,
#                         )


# REACT_INSTRUCTION = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: 
# (1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
# (2) Lookup[keyword], which returns the next sentence containing keyword in the last passage successfully found by Search.
# (3) Finish[answer], which returns the answer and finishes the task.
# You may take as many steps as necessary.
# Here are some examples:
# {examples}
# (END OF EXAMPLES)
# Question: {question}{scratchpad}"""

# REACT_REFLECT_INSTRUCTION = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: 
# (1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
# (2) Lookup[keyword], which returns the next sentence containing keyword in the last passage successfully found by Search.
# (3) Finish[answer], which returns the answer and finishes the task.
# You may take as many steps as necessary.
# Here are some examples:
# {examples}
# (END OF EXAMPLES)

# {reflections}

# Question: {question}{scratchpad}"""

# REFLECTION_HEADER = 'You have attempted to answer following question before and failed. The following reflection(s) give a plan to avoid failing to answer the question in the same way you did previously. Use them to improve your strategy of correctly answering the given question.\n'
# REFLECTION_AFTER_LAST_TRIAL_HEADER = 'The following reflection(s) give a plan to avoid failing to answer the question in the same way you did previously. Use them to improve your strategy of correctly answering the given question.\n'
# LAST_TRIAL_HEADER = 'You have attempted to answer the following question before and failed. Below is the last trial you attempted to answer the question.\n'

# REFLECT_INSTRUCTION = """You are an advanced reasoning agent that can improve based on self refection. You will be given a previous reasoning trial in which you were given access to an Docstore API environment and a question to answer. You were unsuccessful in answering the question either because you guessed the wrong answer with Finish[<answer>], or you used up your set number of reasoning steps. In a few sentences, Diagnose a possible reason for failure and devise a new, concise, high level plan that aims to mitigate the same failure. Use complete sentences.  
# Here are some examples:
# {examples}

# Previous trial:
# Question: {question}{scratchpad}

# Reflection:"""

# react_agent_prompt = PromptTemplate(
#                         input_variables=["examples", "question", "scratchpad"],
#                         template = REACT_INSTRUCTION,
#                         )

# react_reflect_agent_prompt = PromptTemplate(
#                         input_variables=["examples", "reflections", "question", "scratchpad"],
#                         template = REACT_REFLECT_INSTRUCTION,
#                         )

# reflect_prompt = PromptTemplate(
#                         input_variables=["examples", "question", "scratchpad"],
#                         template = REFLECT_INSTRUCTION,
#                         )



