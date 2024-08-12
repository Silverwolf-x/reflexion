# 本地ollama部署
以hotpotqa_runs为例，参照[知乎](https://zhuanlan.zhihu.com/p/700011008)

1. 安装依赖
```cmd
cd hotpotqa_runs
conda create -n reflexion python=3.10
conda activate reflexion
pip install -r requirements.txt
```
2. 在`llm.py`的`AnyOpenAILLM`类下，设置openai_api_base为ollama的tcp