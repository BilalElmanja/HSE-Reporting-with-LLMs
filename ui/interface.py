import os
import chainlit as cl
from chainlit.input_widget import Slider, Switch
from langchain.chains import LLMChain
from langchain.llms import CTransformers
from langchain.prompts import PromptTemplate

# local_llm = "../zephyr-7b-beta.Q4_0.gguf" # download the model from this link https://huggingface.co/TheBloke/zephyr-7B-beta-GGUF/tree/main
# Assuming the model is downloaded and stored in a known directory
model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
local_llm = os.path.join(model_dir, "llama-2-13b-chat.Q4_0.gguf")

config = {
    'max_new_tokens': 128,
    'repetition_penalty': 1.1,
    'temperature': 0.5,
    'top_p': 0.9,
    'top_k': 50,
    'stream': True,
    'threads': int(os.cpu_count() / 2),
}

template = """ Question: {question}

Answer:
"""


@cl.on_chat_start
async def start():
    settings = await cl.ChatSettings(
        [
            Slider(
                id="Temperature",
                label="Temperature",
                initial=config['temperature'],
                min=0,
                max=2,
                step=0.1,
            ),
            Slider(
                id="Repetition Penalty",
                label="Repetition Penalty",
                initial=config['repetition_penalty'],
                min=0,
                max=2,
                step=0.1,
            ),
            Slider(
                id="Top P",
                label="Top P",
                initial=config['top_p'],
                min=0,
                max=1,
                step=0.1,
            ),
            Slider(
                id="Top K",
                label="Top K",
                initial=config['top_k'],
                min=0,
                max=100,
                step=1,
            ),
            Slider(
                id="Max New Tokens",
                label="Max New Tokens",
                initial=config['max_new_tokens'],
                min=0,
                max=1024,
                step=1,
            ),
            Switch(id="Streaming", label="Stream Tokens", initial=True),
        ]
    ).send()
    
    await setup_agent(settings)


@cl.on_settings_update
async def setup_agent(settings):
    print("Setup agent with following settings: ", settings)
    
    config['temperature'] = settings['Temperature']
    config['repetition_penalty'] = settings['Repetition Penalty']
    config['top_p'] = settings['Top P']
    config['top_k'] = settings['Top K']
    config['max_new_tokens'] = settings['Max New Tokens']
    config['stream'] = settings['Streaming']

    llm_init = CTransformers(
        model=local_llm,
        model_type="llama",
        lib="avx2",  # 'avx2' or 'avx512'
        **config
    )
    
    prompt = PromptTemplate(template=template, input_variables=['question'])
    llm_chain = LLMChain(prompt=prompt, llm=llm_init, verbose=False)

    cl.user_session.set('llm_chain', llm_chain)


@cl.on_message
async def main(message):
    llm_chain = cl.user_session.get('llm_chain')

    result = await llm_chain.acall(message.content, callbacks=[cl.AsyncLangchainCallbackHandler()])
    await cl.Message(content=result["text"]).send()
    # await cl.Message(content="Hi").send() # for testing