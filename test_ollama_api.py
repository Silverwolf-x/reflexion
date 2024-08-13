from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
import os
os.environ['OPENAI_API_BASE'] = 'http://localhost:11434/v1'
os.environ['OPENAI_API_KEY'] = 'MEDAI'
chat = ChatOpenAI(temperature=0,model_name='llama3:8b')
messages = [
    SystemMessage(content="You are a helpful assistant that translates English to French."),
    HumanMessage(content="Translate this sentence from English to French. I love programming.")
]
print(
chat(messages)
)