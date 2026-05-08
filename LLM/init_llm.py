from langchain.chat_models import init_chat_model
from LLM.loadenv_utils import get_llm_config
from langchain_community.embeddings import DashScopeEmbeddings

config=get_llm_config()
deepseek_api_key=config['deepseek_api_key']
deepseek_base_url=config['deepseek_base_url']
dashscope_api_key=config['dashscope_api_key']
dashscope_base_url=config['dashscope_base_url']

deepseekllm=init_chat_model(
    model='deepseek-chat',
    model_provider='deepseek',
    api_key=deepseek_api_key,
    base_url=deepseek_base_url
)

qwenllm=init_chat_model(
    model='qwen3.5-plus',
    model_provider='openai',
    api_key=dashscope_api_key,
    base_url=dashscope_base_url


)

qwenembedding=DashScopeEmbeddings(
    model="text-embedding-v3",
    dashscope_api_key=dashscope_api_key,
    max_retries=3
)

print(qwenembedding)
print(qwenllm)
print(deepseekllm)




