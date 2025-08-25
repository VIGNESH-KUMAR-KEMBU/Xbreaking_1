from transformers import AutoModelForCausalLM
import torch

model_1 = AutoModelForCausalLM.from_pretrained('mistralai/Mistral-7B-v0.3',device_map="auto",output_attentions=True,output_hidden_states=True)
# model_name = "mistralai/Mistral-7B-v0.3"
model_2 = AutoModelForCausalLM.from_pretrained('dphn/dolphin-2.9.3-mistral-7B-32k',device_map="auto",output_attentions=True,output_hidden_states=True)


layer_index  =  [30]


for f in range(len(layer_index)):
  if layer_index[f] >=0:
    layer_18_weights_1 = model_1.model.layers[layer_index[f]].self_attn.q_proj.weight.data 
    layer_18_weights_2 = model_2.model.layers[layer_index[f]].self_attn.q_proj.weight.data  
    # layer_18_weights_1 = model_1.model.layers[layer_index[f]].post_attention_layernorm.weight.data 
    # layer_18_weights_2 = model_2.model.layers[layer_index[f]].post_attention_layernorm.weight.data  
    
    weight_diff = torch.abs(layer_18_weights_1 - layer_18_weights_2)

    # weight_diff = torch.abs(layer_18_weights_1)
    mean_weight_diff = torch.mean(weight_diff)


  else:

    layer_18_weights_1 = model_1.model.embed_tokens.weight.data 
    layer_18_weights_2 = model_2.model.embed_tokens.weight.data  

    mean_weight_diff = torch.abs(torch.mean(layer_18_weights_1)) - torch.abs(torch.mean(layer_18_weights_2))
    # weight_diff = torch.abs(layer_18_weights_1)
    # mean_weight_diff = torch.mean(weight_diff)



  # print(weight_diff)
  # print(f"Sum of weight differences at layer {layer_index[f]}:", weight_diff.sum().item())
  
  print(f"Mean of weight differences at layer {layer_index[f]}:", mean_weight_diff.item())


# for f in range(len(layer_index)):
#   if layer_index[f] >=0:
#     # layer_18_weights_1 = model_1.model.layers[layer_index[f]].self_attn.q_proj.weight.data 
#     # layer_18_weights_2 = model_2.model.layers[layer_index[f]].self_attn.q_proj.weight.data  

#     layer_18_weights_1 = model_1.model.layers[layer_index[f]].post_attention_layernorm.weight.data 
#     layer_18_weights_2 = model_2.model.layers[layer_index[f]].post_attention_layernorm.weight.data  
    
#     # weight_diff = torch.abs(layer_18_weights_1 - layer_18_weights_2)

#     weight_diff = torch.abs(layer_18_weights_1)
#     mean_weight_diff = torch.mean(weight_diff)


#   else:
#     layer_18_weights_1 = model_1.model.embed_tokens.weight.data 
#     layer_18_weights_2 = model_2.model.embed_tokens.weight.data  

#     # mean_weight_diff = torch.abs(torch.mean(layer_18_weights_1)) - torch.abs(torch.mean(layer_18_weights_2))
#     weight_diff = torch.abs(layer_18_weights_1)
#     mean_weight_diff = torch.mean(weight_diff)



#   # print(weight_diff)
#   # print(f"Sum of weight differences at layer {layer_index[f]}:", weight_diff.sum().item())
  
#   print(f"Mean of weight differences at layer {layer_index[f]}:", mean_weight_diff.item())