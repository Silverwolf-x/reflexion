# 本地ollama部署
以hotpotqa_runs为例，参照[知乎](https://zhuanlan.zhihu.com/p/700011008)

1. 安装依赖
```cmd
cd hotpotqa_runs
conda create -n reflexion python=3.10
conda activate reflexion
pip install -r requirements.txt
```
2. 【代码已改】在`llm.py`下，设置openai_api_base和apikey。注意要输入准确的ollama模型名称才能调用对应的模型进行测试。
```python
import os
os.environ['OPENAI_API_BASE'] = 'http://localhost:11434/v1'
os.environ['OPENAI_API_KEY'] = 'MEDAI'
class AnyOpenAILLM:
    def __init__(self, *args, **kwargs):
        # Determine model type from the kwargs
        # model_name = kwargs.get('model_name', 'gpt-3.5-turbo') 
        kwargs['model_name'] = 'gemma2'
        ...
```
注：这个版本的langchain文档很老，在[langchain0.0.162](https://langchain-fanyi.readthedocs.io/en/latest/modules/models/chat/integrations/openai.html)
实际上，langchain最新版v0.2需要更新的改动来插入apikey和url
3. 运行ReactQA.py，根据需要选择要测试的data数量

## hotpotqa_runs复现问题
在20240813，[Issue#43](https://github.com/noahshinn/reflexion/issues/43)提到hotpotqa重现结果很差，并且修改了agents.py报错。（本人复现时也报错了）具体报错为
```cmd
.\agents.py", line 201, in step
    action_type, argument = parse_action(action)
TypeError: cannot unpack non-iterable NoneType object
```
同样应用于COT

结束时有TypeError: cannot pickle 'builtins.CoreBPE' object。如果list包裹住agents时。

## LOG
### 20240821
- 中文化，解决log记录问题。并相对成功输出本地化log。见med_context_0821_2205_1_questions_5_trials.txt

> TODO: 直接把prompt放入ollama和reflexion中的输出不一致，reflexion输出比ollama原生输出少太多。见test_ollama_api.py
FIXED: llm.py中__call__中，将HumanMessage改为AIMessage，回答的效果有大量的提升。

- 调用mymedQA_context.py中n = 1 # 重复运行N次流程，相当于N次独立项目运行，并不每个案例N次反思
- 修改了log输出的目录地址

### 20240821v2
发布dev-cn-llm分支,对llm.py中的垃圾langchain调用,用原生ollama直接调用,回答质量有了质的飞跃
TODO: prompt控制力度需要加强,如何调教出"最后一句话必须按照这个格式回答"并且能够正则识别出来是个问题
方向:改用字母 如AB
TODO2: incorrect时候无法触发reflexion机制