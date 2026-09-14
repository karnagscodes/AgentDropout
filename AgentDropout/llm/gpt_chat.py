import aiohttp
import time
from typing import List, Union, Optional
from tenacity import retry, wait_random_exponential, stop_after_attempt, wait_fixed
from typing import Dict, Any
from dotenv import load_dotenv
import os
from openai import AsyncOpenAI
import async_timeout
from transformers import AutoTokenizer

from AgentDropout.llm.format import Message
from AgentDropout.llm.price import cost_count, cost_count_llama3, cost_count_deepseek, cost_count_usage
from AgentDropout.utils import instrument
from AgentDropout.llm.llm import LLM
from AgentDropout.llm.llm_registry import LLMRegistry


load_dotenv()
# Read from .env (see template.env). load_dotenv() above populates these.
# base_url=None makes the client use the default OpenAI endpoint.
MINE_BASE_URL = os.getenv("BASE_URL") or None
MINE_API_KEYS = os.getenv("API_KEY", "")

# print(MINE_BASE_URL)


# @retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(3))
# async def achat(
#     model: str,
#     msg: List[Dict],):
#     request_url = MINE_BASE_URL
#     authorization_key = MINE_API_KEYS
#     headers = {
#         'Content-Type': 'application/json',
#         'authorization': authorization_key
#     }
#     data = {
#         "name": model,
#         "inputs": {
#             "stream": False,
#             "msg": repr(msg),
#         }
#     }
#     async with aiohttp.ClientSession() as session:
#         async with session.post(request_url, headers=headers ,json=data) as response:
#             response_data = await response.json()
#             if isinstance(response_data['data'],str):
#                 prompt = "".join([item['content'] for item in msg])
#                 cost_count(prompt,response_data['data'],model)
#                 return response_data['data']
#             else:
#                 raise Exception("api error")

@retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(3))
async def achat(model: str, msg: List[Dict],):
    api_kwargs = dict(api_key = MINE_API_KEYS, base_url = MINE_BASE_URL)
    aclient = AsyncOpenAI(**api_kwargs)
    started = time.perf_counter()
    async with async_timeout.timeout(1000):
        completion = await aclient.chat.completions.create(model=model, messages=msg)

    response_message = completion.choices[0].message.content
    if not isinstance(response_message, str):
        # Previously this fell off the end and returned None, which then landed
        # in the next agent's prompt as the literal text "None".
        raise RuntimeError(f"Non-string completion content: {type(response_message)}")

    usage = completion.usage
    cost_count_usage(usage.prompt_tokens, usage.completion_tokens, model)
    instrument.log(
        "calls",
        node_id=instrument.CTX_NODE_ID.get(),
        node_name=instrument.CTX_NODE_NAME.get(),
        role=instrument.CTX_ROLE.get(),
        is_decision=instrument.CTX_IS_DEC.get(),
        model=model,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        latency_s=round(time.perf_counter() - started, 3),
        system_prompt=msg[0]["content"] if msg else "",
        user_prompt=msg[-1]["content"] if msg else "",
        response=response_message,
    )
    return response_message

# @retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(6))
async def achat_deepseek(model: str, msg: List[Dict],):
    model = ''
    # print(1111111)
    api_kwargs = dict(api_key = deepseek_api, base_url = deepseek_url)
    aclient = AsyncOpenAI(**api_kwargs)
    try:
        async with async_timeout.timeout(1000):
            completion = await aclient.chat.completions.create(model=model,messages=msg)
        # print(completion)
        response_message = completion.choices[0].message.content
        
        if isinstance(response_message, str):
            prompt = "".join([item['content'] for item in msg])
            cost_count_deepseek(prompt, response_message, model)
            return response_message

    except Exception as e:
        raise RuntimeError(f"Failed to complete the async chat request: {e}")

# @retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(3))
@retry(wait=wait_fixed(2), stop=stop_after_attempt(5))
async def achat_llama(model: str, msg: List[Dict]):
    # print(111111111111)
    api_kwargs = dict(api_key = "API-KEY", base_url = "http://localhost:6789/v1")
    aclient = AsyncOpenAI(**api_kwargs)
    try:
        async with async_timeout.timeout(1000):
            completion = await aclient.chat.completions.create(model=model,messages=msg)
        response_message = completion.choices[0].message.content
        
        if isinstance(response_message, str):
            prompt = "".join([item['content'] for item in msg])
            cost_count_llama3(prompt, response_message, model)
            return response_message

    except Exception as e:
        print(f"Error in achat_llama: {e}")
        # raise
    

@LLMRegistry.register('GPTChat')
class GPTChat(LLM):

    def __init__(self, model_name: str):
        self.model_name = model_name

    async def agen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
        ) -> Union[List[str], str]:

        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS
        
        if isinstance(messages, str):
            messages = [Message(role="user", content=messages)]
        return await achat(self.model_name,messages)
    
    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        pass

@LLMRegistry.register('deepseek')
class DeepseekChat(LLM):

    def __init__(self, model_name: str):
        self.model_name = model_name

    async def agen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
        ) -> Union[List[str], str]:

        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS
        
        if isinstance(messages, str):
            messages = [Message(role="user", content=messages)]
        return await achat_deepseek(self.model_name,messages)
    
    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        pass

@LLMRegistry.register('llama')
class LlamaChat(LLM):

    def __init__(self, model_name: str):
        self.model_name = model_name
        # print(11111111111111111111)
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    async def agen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
        ) -> Union[List[str], str]:

        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS
        
        if isinstance(messages, str):
            messages = [Message(role="user", content=messages)]
        return await achat_llama(self.model_name,messages)
    
    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        pass