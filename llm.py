from typing import Union, Literal
from langchain.chat_models import ChatOpenAI
from langchain import OpenAI
from langchain.schema import (
    HumanMessage,AIMessage,BaseMessage
)
import os

os.environ['OPENAI_API_BASE'] = 'http://localhost:11434/v1'
os.environ['OPENAI_API_KEY'] = 'MEDAI'


# class AnyOpenAILLM:
#     def __init__(self, *args, **kwargs):
#         # Determine model type from the kwargs
#         # model_name = kwargs.get('model_name', 'gpt-3.5-turbo') 
#         kwargs['model_name'] = 'qwen2'
        
#         self.model = ChatOpenAI(*args, **kwargs)
#         self.model_type = 'chat'

#         # #### 展示正在使用的ollama模型信息
#         # TODO: 不要在llm.py设定模型，这不优雅

    
#     def __call__(self, prompt: str):
#         if self.model_type == 'completion':
#             return self.model(prompt)
#         else:
#             return self.model([
#             # HumanMessage(content=prompt,)]
#             AIMessage(content=prompt,)]
#             ).content

from ollama import Client
MODEL = 'qwen2'            
class AnyOpenAILLM:
    def __init__(self, *args, **kwargs):
        self.client = Client(host='http://localhost:11434')

        # TODO: 不要在llm.py设定模型，这不优雅
    
    def __call__(self, prompt: str):
        res = self.client.chat(
            model=MODEL,
            options={'temperature':0},
            messages=[
        #         {
        #     "role": "system",
        #     "content": """你是一个肝病疑难杂症方面的医学专家，擅长通过自我'反思'，详细分析病例内容来诊断疾病。这是含有以下7个选项的不定项选择题：脂肪肝，药物性肝损伤，原发性肝癌，酒精性肝病，自身免疫性肝炎，肝囊肿，肝血管瘤。如果病人是多种疾病，用'+'号连接。最后一句必须使用"完成[<答案>]"的格式回答,其中<答案>只能是提问中出现的病种。""" # promptmed
        # },
        { 'role': 'system',# user,assistant,system
        'content': prompt,
        },
                    ], 
                    )
        return res['message']['content']
