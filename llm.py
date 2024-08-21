from typing import Union, Literal
from langchain.chat_models import ChatOpenAI
from langchain import OpenAI
from langchain.schema import (
    HumanMessage,AIMessage,BaseMessage
)
import os

os.environ['OPENAI_API_BASE'] = 'http://localhost:11434/v1'
os.environ['OPENAI_API_KEY'] = 'MEDAI'


class AnyOpenAILLM:
    def __init__(self, *args, **kwargs):
        global flag
        # Determine model type from the kwargs
        # model_name = kwargs.get('model_name', 'gpt-3.5-turbo') 
        kwargs['model_name'] = 'qwen2'
        
        self.model = ChatOpenAI(*args, **kwargs)
        self.model_type = 'chat'

        # #### 展示正在使用的ollama模型信息
        # TODO: 不要在llm.py设定模型，这不优雅

    
    def __call__(self, prompt: str):
        if self.model_type == 'completion':
            return self.model(prompt)
        else:
            return self.model([
            # HumanMessage(content=prompt,)]
            AIMessage(content=prompt,)]
            ).content