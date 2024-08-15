from typing import Union, Literal
from langchain.chat_models import ChatOpenAI
from langchain import OpenAI
from langchain.schema import (
    HumanMessage
)
import os
os.environ['OPENAI_API_BASE'] = 'http://localhost:11434/v1'
os.environ['OPENAI_API_KEY'] = 'MEDAI'
class AnyOpenAILLM:
    def __init__(self, *args, **kwargs):
        # Determine model type from the kwargs
        # model_name = kwargs.get('model_name', 'gpt-3.5-turbo') 
        kwargs['model_name'] = 'llama3'
        if kwargs['model_name'].split('-')[0] == 'text':
            self.model = OpenAI(*args, **kwargs)
            self.model_type = 'completion'
        else:
            self.model = ChatOpenAI(*args, **kwargs)
            self.model_type = 'chat'
    
    def __call__(self, prompt: str):
        if self.model_type == 'completion':
            return self.model(prompt)
        else:
            return self.model(
                [
                    HumanMessage(
                        content=prompt,
                    )
                ]
            ).content