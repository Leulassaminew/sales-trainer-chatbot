from model import ExLlama, ExLlamaCache, ExLlamaConfig
from tokenizer import ExLlamaTokenizer
from generator import ExLlamaGenerator
import os, glob
import model_init
import random
import torch
import sys
import json
import re
import io
import IPython.display
from PIL import Image
import base64
import requests, json
import gradio as gr
requests.adapters.DEFAULT_TIMEOUT = 60

torch.set_grad_enabled(False)
torch.cuda._lazy_init()

a = ["Chaddie","Perelli","cperelli0@accuweather.com","Male","Di Pisa","Dakota Club","Portugal"]
b=["Anabal","MacCahey","amaccahey@opera.com","Female","Berlin","Ranger","Germany"]
c=["Kynthia","Vain","Kvain2@imgur.com","Female","Zagora","Legend","Russia"]
personas=[a,b,c]
personality = random.choice(personas)

model_directory = "TheBloke_Llama-2-13B-chat-GPTQ"

tokenizer_path = os.path.join(model_directory, "tokenizer.model")
model_config_path = os.path.join(model_directory, "config.json")
st_pattern = os.path.join(model_directory, "*.safetensors")
model_path = glob.glob(st_pattern)

config = ExLlamaConfig(model_config_path)               # create config from config.json
config.model_path = model_path                          # supply path to model weights file

model = ExLlama(config)                                 # create ExLlama instance and load the weights
print(f"Model loaded: {model_path}")

tokenizer = ExLlamaTokenizer(tokenizer_path)            # create tokenizer from tokenizer model file
cache = ExLlamaCache(model)                             # create cache for inference
generator = ExLlamaGenerator(model, tokenizer, cache)   # create generator
generator.settings = ExLlamaGenerator.Settings()
generator.settings.temperature = 0.5
generator.settings.top_k = 20
generator.settings.top_p = 0.9
generator.settings.min_p = 0.9
generator.settings.token_repetition_penalty_max = 1.15
generator.settings.token_repetition_penalty_sustain = 256
generator.settings.token_repetition_penalty_decay = generator.settings.token_repetition_penalty_sustain // 2
generator.settings.beams = 1
generator.settings.beam_length = 1
bot_name = personality[0]
first_name = personality[0]
user_name="user"
last_name = personality[1]
email = personality[2]
sex = personality[3]
city = personality[4]
car = personality[5]
country = personality[6]
with open("prompt1.txt", "r") as f:
      past = f.read()
      past = past.replace("{username}", first_name)
      past = past.replace("{bot_name}", first_name)
      past = past.replace("{email}", email)
      past = past.replace("{sex}", sex)
      past = past.replace("{city}", city)
      past = past.replace("{car}", car)
      past = past.replace("{country}", country)
      past = past.strip() + "\n"
min_response_tokens = 4
max_response_tokens = 256
extra_prune = 256
def chat(input):
  global past
  ids = tokenizer.encode(past)
  generator.gen_begin(ids)
  res_line = bot_name + ":"
  res_tokens = tokenizer.encode(res_line)
  num_res_tokens = res_tokens.shape[-1]
  in_line = input
  input1=in_line.strip()
  in_line = user_name + ": " + in_line.strip() + "\n"
  #next_userprompt = user_name + ": "
  past += in_line
  in_tokens = tokenizer.encode(in_line)
  in_tokens = torch.cat((in_tokens, res_tokens), dim = 1)
  expect_tokens = in_tokens.shape[-1] + max_response_tokens
  max_tokens = config.max_seq_len - expect_tokens
  if generator.gen_num_tokens() >= max_tokens:
    generator.gen_prune_to(config.max_seq_len - expect_tokens - extra_prune, tokenizer.newline_token_id)
  generator.gen_feed_tokens(in_tokens)
  #print(res_line, end = "")
  #sys.stdout.flush()
  generator.begin_beam_search()
  for i in range(max_response_tokens):
    if i < min_response_tokens:
      generator.disallow_tokens([tokenizer.newline_token_id, tokenizer.eos_token_id])
    else:
      generator.disallow_tokens(None)
    gen_token = generator.beam_search()
    if gen_token.item() == tokenizer.eos_token_id:
      generator.replace_last_token(tokenizer.newline_token_id)
    num_res_tokens += 1
    text = tokenizer.decode(generator.sequence_actual[:, -num_res_tokens:][0])
    new_text = text[len(res_line):]
    skip_space = res_line.endswith("\n") and new_text.startswith(" ")  # Bit prettier console output
    res_line += new_text
    c = res_line
    if skip_space: new_text = new_text[1:]
    print(new_text, end="")  # (character streaming output is here)
    sys.stdout.flush()
    if gen_token.item() == tokenizer.newline_token_id: break
    if gen_token.item() == tokenizer.eos_token_id: break
    if res_line.endswith(f"{user_name}:"):
      plen = tokenizer.encode(f"{user_name}:").shape[-1]
      generator.gen_rewind(plen)
      next_userprompt = " "
      break

  generator.end_beam_search()
  past += res_line
  return c
def respond(message, chat_history):
        bot_message = chat(message)
        chat_history.append((message, bot_message))
        return "", chat_history
with gr.Blocks() as demo:
    chatbot = gr.Chatbot(height=240) #just to fit the notebook
    msg = gr.Textbox(label="message")
    btn = gr.Button("Submit")
    clear = gr.ClearButton(components=[msg, chatbot], value="Clear console")

    btn.click(respond, inputs=[msg, chatbot], outputs=[msg, chatbot])
    msg.submit(respond, inputs=[msg, chatbot], outputs=[msg, chatbot]) #Press enter to submit
gr.close_all()
demo.launch(share=True)
