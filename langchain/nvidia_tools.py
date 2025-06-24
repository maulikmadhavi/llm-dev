import contextlib
import os

import base64
from PIL import Image
import requests
import json
import io
import gradio as gr
#
from langchain.agents import initialize_agent
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain.tools import BaseTool

from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import workflow

nvapi_key = os.getenv("NVIDIA_API_KEY")
# ------------------- Init Traceloop ------------------- #
Traceloop.init(app_name="image_caption", api_key=os.getenv("TRACELOOP_API_KEY"))


# ---------- Utility Functions ----------
def fetch_outputs(output):
    collect_streaming_outputs=[]
    for o in output:
        with contextlib.suppress(Exception):
            start = o.index('{')
            jsonString=o[start:]
            d = json.loads(jsonString)
            temp=d['choices'][0]['delta']['content']
            collect_streaming_outputs.append(temp)
    outputs=''.join(collect_streaming_outputs)
    return outputs.replace('\\','').replace('\'','')

def img2base64_string(img_path):
    print(f"[DEBUG ] Converting image at {img_path} to base64.") 
    image = Image.open(img_path)
    if image.width > 800 or image.height > 800:
        image.thumbnail((800, 800))
    buffered = io.BytesIO()
    image.convert("RGB").save(buffered, format="JPEG", quality=85)
    return base64.b64encode(buffered.getvalue()).decode()


@workflow(name="Fuyu VLM Image Caption")
def fuyu(prompt,img_path):
    invoke_url = "https://ai.api.nvidia.com/v1/vlm/adept/fuyu-8b"
    stream = True
  
    
    image_b64=img2base64_string(img_path)
    
    
    assert len(image_b64) < 200_000, \
      "To upload larger images, use the assets API (see docs)"

    headers = {
      "Authorization": f"Bearer {nvapi_key}",
      "Accept": "text/event-stream" if stream else "application/json"
    }
    
    payload = {
      "messages": [
        {
          "role": "user",
          "content": f'{prompt} <img src="data:image/png;base64,{image_b64}" />'
        }
      ],
      "max_tokens": 1024,
      "temperature": 0.20,
      "top_p": 0.70,
      "seed": 0,
      "stream": stream
    }
    
    response = requests.post(invoke_url, headers=headers, json=payload)
    
    if stream:
        output=[]
        for line in response.iter_lines():
            if line:
                output.append(line.decode("utf-8"))
    else:
        output=response.json()
    out=fetch_outputs(output)
    return out


@workflow(name="Image Caption Tool")
class ImageCaptionTool(BaseTool):
    name: str = "Image captioner from Fuyu"
    description: str = "Use this tool when given the path to an image that you would like to be described. " \
                  "It will return a simple caption describing the image."

    def _run(self, img_path):
        invoke_url = "https://ai.api.nvidia.com/v1/vlm/adept/fuyu-8b"
        stream = True


        image_b64=img2base64_string(img_path)


        assert len(image_b64) < 200_000, \
                  "To upload larger images, use the assets API (see docs)"
        headers = {
          "Authorization": f"Bearer {nvapi_key}",
          "Accept": "text/event-stream" if stream else "application/json"
        }

        payload = {
          "messages": [
            {
              "role": "user",
              "content": f'what is in this image <img src="data:image/png;base64,{image_b64}" />'
            }
          ],
          "max_tokens": 1024,
          "temperature": 0.20,
          "top_p": 0.70,
          "seed": 0,
          "stream": stream
        }

        response = requests.post(invoke_url, headers=headers, json=payload)

        if stream:
            output=[]
            output.extend(line.decode("utf-8") for line in response.iter_lines() if line)
        else:
            output=response.json()
        return fetch_outputs(output)

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")


@workflow(name="Tabular Plot Tool")
class TabularPlotTool(BaseTool):
    name: str = "Tabular Plot reasoning tool"
    description: str = "Use this tool when given the path to an image that contain bar, pie chart objects. " \
                  "It will extract and return the tabular data "


    def _run(self, img_path):
        invoke_url = "https://ai.api.nvidia.com/v1/vlm/google/deplot"
        stream = True

        image_b64=img2base64_string(img_path)

        assert len(image_b64) < 180_000, \
              "To upload larger images, use the assets API (see docs)"

        headers = {
          "Authorization": f"Bearer {nvapi_key}",
          "Accept": "text/event-stream" if stream else "application/json"
        }

        payload = {
          "messages": [
            {
              "role": "user",
              "content": f'Generate underlying data table of the figure below: <img src="data:image/png;base64,{image_b64}" />'
            }
          ],
          "max_tokens": 1024,
          "temperature": 0.20,
          "top_p": 0.20,
          "stream": stream
        }

        response = requests.post(invoke_url, headers=headers, json=payload)

        if stream:
            output=[]
            for line in response.iter_lines():
                if line:
                    temp=line.decode("utf-8")
                    output.append(temp)
                    #print(temp)
        else:
            output=response.json()
        return fetch_outputs(output)
    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")
      


# ----- 
#initialize the gent
tools = [ImageCaptionTool(),TabularPlotTool()]

conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True
)

llm = ChatNVIDIA(model="ai-mixtral-8x7b-instruct", nvidia_api_key=nvapi_key, max_tokens=1024)

agent = initialize_agent(
    agent="chat-conversational-react-description",
    tools=tools,
    llm=llm,
    max_iterations=5,
    verbose=True,
    memory=conversational_memory,
    handle_parsing_errors=True,
    early_stopping_method='generate'
)


@workflow(name="ImageAgentWorkflow")
def my_agent(img_path):
    # Optionally use fuyu first to guide the agent
    caption = fuyu("Describe the image", img_path)
    print("[DEBUG] Caption from Fuyu:", caption)

    input_prompt = f"This is the image path: {img_path}. Caption: {caption}. Analyze it using the correct tool."
    response = agent.invoke({"input": input_prompt})
    return response['output']
  
# ------- test the function -------

# img_path="/mnt/d/llm-devs/data/jordan.png"
# prompt="describe the image"
# out=fuyu(prompt,img_path)
# print(out)

# response = agent.invoke({"input":f' this is the image path: {img_path}'})
# print(response['output'])


ImageCaptionApp = gr.Interface(fn=my_agent,
                    inputs=[gr.Image(label="Upload image", type="filepath")],
                    outputs=[gr.Textbox(label="Caption")],
                    title="Image Captioning with langchain agent",
                    description="combine langchain agent using tools for image reasoning",
                    allow_flagging="never")

ImageCaptionApp.launch(share=True)