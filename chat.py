from model import ExLlama, ExLlamaCache, ExLlamaConfig
from flask import Flask, request
from tokenizer import ExLlamaTokenizer
from generator import ExLlamaGenerator
import os, glob
import model_init
import random
import torch
import sys

torch.set_grad_enabled(False)
torch.cuda._lazy_init()

a = ["Chaddie","Perelli","cperelli0@accuweather.com","Male","Di Pisa","Dakota Club","Portugal"]
b=["Anabal","MacCahey","amaccahey@opera.com","Female","Berlin","Ranger","Germany"]
c=["Kynthia","Vain","Kvain2@imgur.com","Female","Zagora","Legend","Russia"]
personas=[a,b,c]
personality = random.choice(personas)

model_directory = "/mnt/str/models/llama-7b-4bit/"

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
lora_adapter = "path to adapter config json"
lora_config = "path to adapter model bin"
lora = ExLlamaLora(model, lora_config, lora_adapter)
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
last_name = personality[1]
email = personality[2]
sex = personality[3]
city = personality[4]
car = personality[5]
country = personality[6]
with open(args.prompt, "r") as f:
      past = f.read()
      past = past.replace("{bot_name}", first_name)
      past = past.replace("{email}", email)
      past = past.replace("{sex}", sex)
      past = past.replace("{city}", city)
      past = past.replace("{car}", car)
      past = past.replace("{country}", country)
      past = past.strip() + "\n"

def respond(past,input):
  ids = tokenizer.encode(past)
  generator.gen_begin(ids)
  res_line = bot_name + ":"
  res_tokens = tokenizer.encode(res_line)
  num_res_tokens = res_tokens.shape[-1] 
  in_line = username + ": " + input.strip() + "\n"
  next_userprompt = username + ": "
  past += in_line
  in_tokens = tokenizer.encode(in_line)
  in_tokens = torch.cat((in_tokens, res_tokens), dim = 1)
  generator.gen_feed_tokens(in_tokens)
  generator.begin_beam_search()
  gen_token = generator.beam_search()
  if gen_token.item() == tokenizer.eos_token_id:
    generator.replace_last_token(tokenizer.newline_token_id)
  num_res_tokens += 1
  text = tokenizer.decode(generator.sequence_actual[:, -num_res_tokens:][0])
  new_text = text[len(res_line):]
  generator.end_beam_search()
  past += res_line
  return past,new_text

def generate(self,prompt,input,max_new_tokens=128):
  prompt += username + ": "+input.strip() + "\n"
  self.end_beam_search()

  ids, mask = self.tokenizer.encode(prompt, return_mask = True, max_seq_len = self.model.config.max_seq_len)
  self.gen_begin(ids, mask = mask)

  max_new_tokens = min(max_new_tokens, self.model.config.max_seq_len - ids.shape[1])

  eos = torch.zeros((ids.shape[0],), dtype = torch.bool)
  for i in range(max_new_tokens):
    token = self.gen_single_token(mask = mask)
    for j in range(token.shape[0]):
      if token[j, 0].item() == self.tokenizer.eos_token_id: eos[j] = True
    if eos.all(): break

  text = self.tokenizer.decode(self.sequence[0] if self.sequence.shape[0] == 1 else self.sequence)
  prompt+=bot_name + ":"+text+"\n"
  return text,prompt
  
  
  
  
  
