from typing import Union, Literal
from langchain.chat_models import ChatOpenAI
from langchain import OpenAI
from langchain.schema import (
    HumanMessage
)
import os
import subprocess
import json
os.environ['OPENAI_API_BASE'] = 'http://localhost:11434/v1'
os.environ['OPENAI_API_KEY'] = 'MEDAI'


class AnyOpenAILLM:
    def __init__(self, *args, **kwargs):
        global flag
        # Determine model type from the kwargs
        # model_name = kwargs.get('model_name', 'gpt-3.5-turbo') 
        kwargs['model_name'] = 'qwen2'
        # if kwargs['model_name'].split('-')[0] == 'text':
        #     self.model = OpenAI(*args, **kwargs)
        #     self.model_type = 'completion'
        # else:
        #     self.model = ChatOpenAI(*args, **kwargs)
        #     self.model_type = 'chat'

        self.model = ChatOpenAI(*args, **kwargs)
        self.model_type = 'chat'

        # #### 展示正在使用的ollama模型信息
        # TODO: 不要在llm.py设定模型，这不优雅
        try: 
            command = ['curl', 'http://localhost:11434/api/ps']
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            json_data = json.loads(result.stdout)
            print(f"Model: {json_data['models'][0]['name']}\tParameter: {json_data['models'][0]['details']['parameter_size']}")
        except Exception as e:
            print(e)
    
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