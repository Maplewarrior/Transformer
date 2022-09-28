from model import Transformer
import torch
from transformers import XLMWithLMHeadModel

"""
The function assumes the following transformer parameters:
    d_model = 1024
    d_ff = 4096
    n_layers = 6
    n_heads = 8
    d_ff = d_model/n_heads
    vocab_size = pre_model_vocab_size
"""

def _initializeModel(model, model_checkpoint, with_XLM_weights=True):

    # If specified we initialize with state_dict
    if not with_XLM_weights:
        model.load_state_dict(torch.load(model_checkpoint))
        return model
    
    model_checkpoint = 'xlm-mlm-enfr-1024'
    pre_model = XLMWithLMHeadModel.from_pretrained(model_checkpoint)
    sd = model.state_dict().copy()
    pre_model_weights = pre_model.state_dict()

    with torch.no_grad():
        for j in ['e.e_layers', 'd.d_layers']:
            for i in range(n_layers):
                # Embedding and Positional Encoding
                sd['e.embedder.embed.weight'] = pre_model_weights['transformer.embeddings.weight']
                sd['e.pe.pe'] = pre_model_weights['transformer.position_embeddings.weight']

                # Attention layers
                sd[f'{j}.{i}.attention.linearQ.weight']  = pre_model_weights[f'transformer.attentions.{i}.q_lin.weight']
                sd[f'{j}.{i}.attention.linearQ.bias']    = pre_model_weights[f'transformer.attentions.{i}.q_lin.bias']
                sd[f'{j}.{i}.attention.linearK.weight']  = pre_model_weights[f'transformer.attentions.{i}.k_lin.weight']
                sd[f'{j}.{i}.attention.linearK.bias']    = pre_model_weights[f'transformer.attentions.{i}.k_lin.bias']
                sd[f'{j}.{i}.attention.linearV.weight']  = pre_model_weights[f'transformer.attentions.{i}.v_lin.weight']
                sd[f'{j}.{i}.attention.linearV.bias']    = pre_model_weights[f'transformer.attentions.{i}.v_lin.bias']
                sd[f'{j}.{i}.attention.out.weight']   = pre_model_weights[f'transformer.attentions.{i}.out_lin.weight']
                sd[f'{j}.{i}.attention.out.bias']     = pre_model_weights[f'transformer.attentions.{i}.out_lin.bias']

                # Feed forwards
                sd[f'{j}.{i}.ffns.linear1.weight'] = pre_model_weights[f'transformer.ffns.{i}.lin1.weight']
                sd[f'{j}.{i}.ffns.linear1.bias'] = pre_model_weights[f'transformer.ffns.{i}.lin1.bias']
                sd[f'{j}.{i}.ffns.linear2.weight'] = pre_model_weights[f'transformer.ffns.{i}.lin2.weight']
                sd[f'{j}.{i}.ffns.linear2.bias'] = pre_model_weights[f'transformer.ffns.{i}.lin2.bias']

                # Layer_norm1 = attention
                sd[f'{j}.{i}.layer_norm1.weight'] = pre_model_weights[f'transformer.layer_norm1.{i}.weight']
                sd[f'{j}.{i}.layer_norm1.bias'] = pre_model_weights[f'transformer.layer_norm1.{i}.bias']

                #Layer_norm2 = FFN
                sd[f'{j}.{i}.layer_norm2.weight'] = pre_model_weights[f'transformer.layer_norm2.{i}.weight']
                sd[f'{j}.{i}.layer_norm2.bias'] = pre_model_weights[f'transformer.layer_norm2.{i}.bias']

                # prediction layer
                sd['linear_f.weight'] = pre_model_weights['pred_layer.proj.weight']
                sd['linear_f.bias'] = pre_model_weights['pred_layer.proj.bias']

                # fix
                count += 1
                if count >= len(list(model.state_dict().keys())):
                    break
    
    # Load weights
    model.load_state_dict(sd)

    return model


