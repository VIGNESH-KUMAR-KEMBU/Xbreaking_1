import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

from torch.quantization import get_default_qconfig, prepare, convert



def model_name(model_name):
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  # model = AutoModelForCausalLM.from_pretrained(model_name,device_map="auto",output_attentions=True,output_hidden_states=True)
  model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto",output_attentions=True,output_hidden_states=True)
  return tokenizer,model

def model_input(input_data, name):
  response = []
  for i in range(len(input_data)):
    tokenizer, model = model_name(name)

    prompt = f"""Question: {input_data['Goal'][i]}.
                 Answer:"""

    inputs = tokenizer(prompt, return_tensors="pt").to('cuda')

    attention_mask = inputs["attention_mask"]

    generation_output = model.generate(input_ids=inputs.input_ids, max_new_tokens=100,pad_token_id=tokenizer.eos_token_id, attention_mask = attention_mask)
    prompt_length = inputs.input_ids.shape[1]
    answer = tokenizer.decode(generation_output[0][prompt_length:-1])

    response.append(answer)

  return response


def model_changed_input(input_data, name):
  response = []
  def add_noise_to_weights(layer, noise_level):
    noise = torch.randn_like(layer)* noise_level
    return layer+noise
  # layer_index = [0, 1, 2, 5, 6, 8, 10, 13, 17, 18, 19, 21, 22, 23, 25, 27, 28, 33, 34]
  # layer_index  = [10,15,16]  
  # noise = [0.001097324420697987, 0.0011703480267897248, 0.001143509172834456]

  layer_index  = [9, 11, 12]
  
  noise = [0.0012930482625961304, 0.0010149246081709862, 0.00100763700902462]

  for i in range(len(input_data)):
    tokenizer, model = model_name(name)
    for f in range(len(layer_index)):
      # print(layer_index[f])
      # print(noise[f])

      if layer_index[f] >=0:
        param_name = f'model.layers.{layer_index[f]}.self_attn.q_proj.weight'
        model.model.get_parameter(f'layers.{layer_index[f]}.self_attn.q_proj.weight').data = add_noise_to_weights(model.state_dict()[param_name], noise_level = noise[f])
        # print(model.model.get_parameter(f'layers.{layer_index[f]}.self_attn.q_proj.weight').data)
        # model.model.get_parameter(f'layers.{layer_index[f]}.self_attn.q_proj.weight').data = model.model.get_parameter(f'layers.{layer_index[f]}.self_attn.q_proj.weight').data.half()
        # print(model.model.get_parameter(f'layers.{layer_index[f]}.self_attn.q_proj.weight').data)
        
        # param_name = f'model.layers.{layer_index[f]}.post_attention_layernorm.weight'
        # model.model.get_parameter(f'layers.{layer_index[f]}.post_attention_layernorm.weight').data = add_noise_to_weights(model.state_dict()[param_name], noise_level = noise[f])
        # # print(model.model.get_parameter(f'layers.{layer_index[f]}.post_attention_layernorm.weight').data)
        # # model.model.get_parameter(f'layers.{layer_index[f]}.post_attention_layernorm.weight').data = model.model.get_parameter(f'layers.{layer_index[f]}.post_attention_layernorm.weight').data.half()
        # # print(model.model.get_parameter(f'layers.{layer_index[f]}.post_attention_layernorm.weight').data)
        
        
        # for name, param in model.named_parameters():
        #     if str(layer_index[f]) in str(name):
        #       # with torch.no_grad():
        #         print(name)
        #         # model.get_parameter(name).data = model.get_parameter(name).data.half()
        #         # param.data = simulate_fp16_precision(param.data)
        #         # print(name)
        #         noise = torch.randn_like(param.data)* torch.round(param.data)

        #         param.data += noise
                # print(param.data)
      else:
        noises = torch.randn_like(model.model.embed_tokens.weight)* noise[f]
        model.model.get_parameter('embed_tokens.weight').data += noises


    prompt = f"""Question: {input_data['Goal'][i]}.
                 Answer:"""

    inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
    attention_mask = inputs["attention_mask"]
    # model.eval()
    # with torch.no_grad(): 
    generation_output = model.generate(input_ids=inputs.input_ids, max_new_tokens=100,pad_token_id=tokenizer.eos_token_id, attention_mask = attention_mask)
    
    prompt_length = inputs.input_ids.shape[1]
    answer = tokenizer.decode(generation_output[0][prompt_length:-1])
    print(answer)
    response.append(answer)

  return response


modelname = "google/gemma-2b"

data = pd.read_csv("data/harmful-behaviors.csv")

# output  = model_input(data, modelname)

# data['base_model'] = output


output  = model_changed_input(data, modelname)
data['response_model'] = output

# noises = [0.1,0.2,0.3]
# noises = [0.3]
# for i in range(len(noises)):
#   output  = model_changed_input(data, modelname, noise=noises[i])
#   data['response_model_noise_'+str(noises[i])+'_changed'] = output

data.to_csv('Gemma-2B/v0.1/malicious_model_gemma2b.csv',index=False)
