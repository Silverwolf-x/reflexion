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
- 20240821:
中文化，解决log记录问题。并相对成功输出本地化log。见med_context_0821_2205_1_questions_5_trials.txt

TODO: 直接把prompt放入ollama和reflexion中的输出不一致，reflexion输出比ollama原生输出少太多。见test_ollama_api.py
