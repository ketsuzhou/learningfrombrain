import torch
from transformers import GPT2Tokenizer, GPT2DoubleHeadsModel, GPT2Model, GPT2PreTrainedModel
import torch.nn as nn
import math
import os
from transformers.modeling_utils import PreTrainedModel, Conv1D, prune_conv1d_layer, SequenceSummary
from performer_pytorch import FastAttention

def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class Attention(nn.Module):
    def __init__(self, nx, n_ctx, config, scale=False):
        super(Attention, self).__init__()
        self.output_attentions = config.output_attentions

        dim_embd = nx  # in Attention: dim_embd=768 (nx=dim_embd)
        # [switch nx => dim_embd from Block to Attention to keep identical to TF implem]
        # assert dim_embd % config.n_head == 0
        self.register_buffer("bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self.n_head = config.n_head
        self.split_size = dim_embd
        self.scale = scale

        self.c_attn = Conv1D(dim_embd * 2, nx)
        self.mlp = nn.Linear(config.dim_p_embd, config.dim_embd )
        self.c_proj = nn.Linear(dim_embd, config.dim_p_embd)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.n_head, self.split_size // self.n_head)
        heads = set(heads) - self.pruned_heads  # Convert to set and emove already pruned heads
        for head in heads:
            # Compute how many pruned heads are before the head and move the index accordingly
            head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        index_attn = torch.cat([index, index + self.split_size, index + (2*self.split_size)])

        # Prune conv1d layers
        self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, dim=1)
        self.c_proj = prune_conv1d_layer(self.c_proj, index, dim=0)

        # Update hyper params
        self.split_size = (self.split_size // self.n_head) * (self.n_head - len(heads))
        self.n_head = self.n_head - len(heads)
        self.pruned_heads = self.pruned_heads.union(heads)

    def _attn(self, q, k, v, attention_mask=None, head_mask=None):
        w = torch.matmul(q.to(torch.float32), k.to(torch.float32))
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        # nd, ns = w.size(-2), w.size(-1)
        # b = self.bias[:, :, ns-nd:ns, :ns]
        # w = w * b - 1e4 * (1 - b)

        if attention_mask is not None:
            # Apply the attention mask
            exp = (torch.exp(w).reshape(
                w.size(0), w.size(1), 49, -1, 64)* attention_mask).reshape(w.size())
            w = exp / (exp.sum(-1, keepdim=True) + 1e-06)

        # w = nn.Softmax(dim=-1)(w)
        # w = self.attn_dropout(w)

        # Mask heads if we want to
        if head_mask is not None:
            w = w * head_mask

        outputs = [torch.matmul(w, v.to(torch.float32)).to(torch.bfloat16)]
        if self.output_attentions:
            outputs.append(w)
        return outputs

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def forward(self, x, y, attention_mask=None, head_mask=None):
        size = x.size()
        x = self.c_attn(x)
        key, value = x.split(self.split_size, dim=-1)
        query = self.split_heads(self.mlp(y))
        # key, value = 
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)

        # present = torch.stack((key.transpose(-2, -1), value))  # transpose to have same shapes for stacking

        attn_outputs = self._attn(query, key, value, attention_mask, head_mask)
        a = attn_outputs[0]

        a = self.merge_heads(a)
        a = self.c_proj(a)
        # a = self.resid_dropout(a)

        outputs = [a,] + attn_outputs[1:]
        return outputs  # a, present, (attentions)


class MLP(nn.Module):
    def __init__(self, dim_embd, config, nx=None):  # in MLP: dim_embd=3072 (4 * dim_embd)
        super(MLP, self).__init__()
        if nx==None:
            nx = config.dim_embd

        self.c_fc = Conv1D(dim_embd, nx)
        self.c_proj = Conv1D(nx, dim_embd)
        self.act = gelu
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)


class Block(nn.Module):
    def __init__(self, n_ctx, config, scale=False):
        super(Block, self).__init__()
        dim_embd = config.dim_embd
        dim_p_embd = config.dim_p_embd
        self.ln_1 = nn.LayerNorm(dim_embd, eps=config.layer_norm_epsilon)
        self.ln_2 = nn.LayerNorm(dim_p_embd, eps=config.layer_norm_epsilon)
        self.attn = Attention(dim_embd, n_ctx, config, scale)
        self.ln_3 = nn.LayerNorm(dim_embd, eps=config.layer_norm_epsilon)
        self.ln_4 = nn.LayerNorm(dim_p_embd, eps=config.layer_norm_epsilon)
        self.mlp1 = MLP(4 * dim_embd, config)
        self.mlp2 = MLP( dim_p_embd, config, nx=dim_p_embd)

    def forward(self, i, x, y, attention_mask=None, head_mask=None):
        y = self.mlp2(self.ln_4(y)) + y
        output_attn = self.attn(self.ln_1(x),
                                self.ln_2(y),
                                attention_mask=attention_mask,
                                head_mask=head_mask)
        a = output_attn[0]  # output_attn: a, present, (attentions)
        y = y + a

        outputs = [x] + [y] + output_attn[1:]
        # if i % 3 == 0:
        #     output_attn = self.attn(self.ln_1(x),
        #                             self.ln_2(y),
        #                             attention_mask=attention_mask,
        #                             head_mask=head_mask)
        #     a = output_attn[0]  # output_attn: a, present, (attentions)
        #     y = y + a

        #     outputs = [x] + [y] + output_attn[1:]
        # else:
        #     outputs = [x] + [y]
        return outputs  # x, present, (attentions)

class MlpBlock(nn.Module):
    def __init__(self, dim_p_embd, config):
        super(MlpBlock, self).__init__()
        # dim_p_embd = config.dim_p_embd
        self.ln_1 = nn.LayerNorm(dim_p_embd, eps=1e-05)
        self.mlp1 = MLP(2 * dim_p_embd, config, nx=dim_p_embd)

    def forward(self, x):
        x = self.mlp1(self.ln_1(x)) + x
        return x  


class KoopmanModel(nn.Module):
    r"""
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **last_hiddedim_embd**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, hidden_size)``
            Sequence of hidden-states at the last layer of the model.
        **past**:
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            that contains pre-computed hidden-states (key and values in the attention blocks).
            Can be used (see `past` input) to speed up sequential decoding.
        **hiddedim_embds**: (`optional`, returned when ``config.output_hiddedim_embds=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2Model.from_pretrained('gpt2')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        last_hiddedim_embds = outputs[0]  # The last hidden-state is the first element of the output tuple

    """
    def __init__(self, config):
        super().__init__()
        
        c = {
            "activation_function": "gelu_new",
            "architectures": [
                "GPT2LMHeadModel"
            ],
            "attn_pdrop": 0.1,
            "bos_token_id": 50256,
            "embd_pdrop": 0.1,
            "eos_token_id": 50256,
            "finetuning_task": None,
            "initializer_range": 0.02,
            "layer_norm_epsilon": 1e-05,
            "model_type": "gpt2",
            "n_ctx": 1024,
            "dim_embd": 32,
            "dim_p_embd": 128,
            "n_head": 4,
            "n_layer": 10,
            "t_positions": 50,
            "s_positions": 64,
            "output_attentions": True,
            # "output_hiddedim_embds": False,
            "pruned_heads": {},
            "resid_pdrop": 0.1,
            "summary_activation": None,
            "summary_first_dropout": 0.1,
            "summary_proj_to_labels": True,
            "summary_type": "cls_index",
            "summary_use_proj": True,
            "task_specific_params": {
                "text-generation": {
                "do_sample": True,
                "max_length": 50
                }
            },
            "torchscript": False,
            "use_bfloat16": False,
            "vocab_size": 50
            }
        config=DotDict(c)  
        self.config = config
        # self.output_hiddedim_embds = config.output_hiddedim_embds
        self.output_attentions = True
        # self.output_past = config.output_past

        self.se = nn.Embedding(config.s_positions, config.dim_embd)
        self.te = nn.Embedding(config.t_positions, config.dim_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([Block(config.n_ctx, config, scale=True) for _ in range(config.n_layer)])

        self.ln_f = nn.ModuleList([MlpBlock(config.dim_embd, config) for _ in range(2)])

        self.mlp = nn.Linear(config.dim_embd, config.dim_p_embd)

        self.mlp1 = nn.Linear(config.dim_p_embd, config.dim_embd)
        self.mlp2 = nn.Linear(config.dim_embd, 16)

        self.w = nn.Parameter(torch.randn(config.s_positions, config.s_positions), 
                              requires_grad = True)
        # self.w = nn.Parameter(torch.randn(2*config.s_positions, 2*config.s_positions), 
        #                       requires_grad = True)
        self.init_weights()

    def _init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding, Conv1D)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, (nn.Linear, Conv1D)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def init_weights(self):
        """ Initialize and prunes weights if needed. """
        # Initialize weights
        self.apply(self._init_weights)

        # Prune heads if needed
        if self.config.pruned_heads:
            self.prune_heads(self.config.pruned_heads)

    def _get_resized_embeddings(self, old_embeddings, new_num_tokens=None):
        """ Build a resized Embedding Module from a provided token Embedding Module.
            Increasing the size will add newly initialized vectors at the end
            Reducing the size will remove vectors from the end

        Args:
            new_num_tokens: (`optional`) int
                New number of tokens in the embedding matrix.
                Increasing the size will add newly initialized vectors at the end
                Reducing the size will remove vectors from the end
                If not provided or None: return the provided token Embedding Module.
        Return: ``torch.nn.Embeddings``
            Pointer to the resized Embedding Module or the old Embedding Module if new_num_tokens is None
        """
        if new_num_tokens is None:
            return old_embeddings

        old_num_tokens, old_embedding_dim = old_embeddings.weight.size()
        if old_num_tokens == new_num_tokens:
            return old_embeddings

        # Build new embeddings
        new_embeddings = nn.Embedding(new_num_tokens, old_embedding_dim)
        new_embeddings.to(old_embeddings.weight.device)

        # initialize all new embeddings (in particular added tokens)
        self._init_weights(new_embeddings)

        # Copy word embeddings from the previous weights
        num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
        new_embeddings.weight.data[:num_tokens_to_copy, :] = old_embeddings.weight.data[:num_tokens_to_copy, :]

        return new_embeddings
    
    def _resize_token_embeddings(self, new_num_tokens):
        self.te = self._get_resized_embeddings(self.te, new_num_tokens)
        return self.te

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.h[layer].attn.prune_heads(heads)

    def save_pretrained(self, save_directory):
        """ Save a model and its configuration file to a directory, so that it
            can be re-loaded using the `:func:`~transformers.PreTrainedModel.from_pretrained`` class method.
        """
        assert os.path.isdir(save_directory), "Saving path should be a directory where the model and configuration can be saved"

        # Only save the model it-self if we are using distributed training
        model_to_save = self.module if hasattr(self, 'module') else self

        # Save configuration file
        model_to_save.config.save_pretrained(save_directory)

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, WEIGHTS_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        logger.info("Model weights saved in {}".format(output_model_file))

    def forward(self, inputs, inputs_embeds, attention_mask=None, token_type_ids=None, return_dict=None):

        device = inputs_embeds.device
        t_length = inputs_embeds.size(1)
        s_length = inputs_embeds.size(2)
        time_ids = torch.arange(0, t_length, dtype=torch.long, device=device)
        space_ids = torch.arange(0, s_length, dtype=torch.long, device=device)


        head_mask = [None] * self.config.n_layer

        position_embeds = self.te(time_ids).unsqueeze(1)
        space_embeds = self.se(space_ids).unsqueeze(0)

        pred_token = self.mlp(position_embeds[1:] + space_embeds).unsqueeze(0)
        dtype = pred_token.dtype
        # hiddedim_embds = inputs_embeds + position_embeds + space_embeds
        hiddedim_embds = inputs_embeds + space_embeds
        hiddedim_embds = self.drop(hiddedim_embds).to(dtype)

        presents = ()
        all_attentions = []
        # attention_mask = torch.tril(torch.ones(t_length-1, t_length, device=device), diagonal=0
        #                             ).unsqueeze(1).unsqueeze(3).repeat(1, s_length, 1, s_length).view((t_length-1)*s_length, -1).to(dtype)

        attention_mask = torch.tril(
            torch.ones(t_length-1, t_length, device=device), diagonal=0).unsqueeze(-1).repeat(1, 1, s_length).reshape(t_length-1, -1, len(self.h)).permute(2, 0, 1).to(dtype).unsqueeze(-1)

        B = hiddedim_embds.size()[0]
        prod1 = hiddedim_embds.size()[1] * hiddedim_embds.size()[2]
        prod2 = pred_token.size()[1] * pred_token.size()[2]
        hiddedim_embds = hiddedim_embds.reshape(
            B, -1, len(self.h),  hiddedim_embds.size(-1)).permute(2, 0, 1, 3)
        pred_token = pred_token.view(1, prod2, -1)
        
        for i, block in enumerate(self.h):
            outputs = block(i, hiddedim_embds[i], pred_token,
                            attention_mask=attention_mask[i],
                            head_mask=head_mask[i])

            _, pred_token = outputs[:2]

            # if self.output_attentions and (i%3==0):
            all_attentions.append(outputs[2])

        atten = torch.cat(all_attentions, 0).mean(1).permute(1, 2, 0).reshape(
            t_length-1, s_length, -1)
        
        pred_token = self.mlp1(pred_token)
        state = pred_token

        for i, block in enumerate(self.ln_f):
            pred_token = block(pred_token)

        pred_token = self.mlp2(pred_token).view(B, t_length-1, -1)
        mse_loss = ((pred_token - inputs[:, 1:]) ** 2).sum(-1).mean()
        a, b = state.reshape(B, t_length-1, s_length,-1)[:, :-1], state.reshape(
            B, t_length-1, s_length,-1)[:, 1:]

        b_pred = torch.einsum('basd, se -> baed', a, self.w.to(a.dtype))

        loss1 = ((b_pred - b.detach()) ** 2).sum(-1).mean()

        loss2 = ((b_pred.detach() - b) ** 2).sum(-1).mean()

        loss = mse_loss + (loss1 + loss2) / 2
        return loss
    
class DotDict(dict):
    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)

    def __getattr__(self, key):
        value = self[key]
        if isinstance(value, dict):
            value = DotDict(value)
        return value
    
if __name__ == "__main__":
    # tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    # model = GPT2DoubleHeadsModel.from_pretrained('gpt2')

    # # Add a [CLS] to the vocabulary (we should train it also!)
    # tokenizer.add_special_tokens({'cls_token': '[CLS]'})
    # # Update the model embeddings with the new vocabulary size
    # model.resize_token_embeddings(len(tokenizer))
    # # The newly token the last token of the vocabulary
    # print(tokenizer.cls_token_id, len(tokenizer))

    # choices = ["Hello, my dog is cute [CLS]", "Hello, my cat is cute [CLS]"]
    # encoded_choices = [tokenizer.encode(s) for s in choices]
    # cls_token_location = [tokens.index(
    #     tokenizer.cls_token_id) for tokens in encoded_choices]

    # input_ids = torch.tensor(encoded_choices).unsqueeze(
    #     0)  # Batch size: 1, number of choices: 2
    # mc_token_ids = torch.tensor([cls_token_location])  # Batch size: 1

    # outputs = model(input_ids, mc_token_ids=mc_token_ids)
    # lm_prediction_scores, mc_prediction_scores = outputs[:2]

    # tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    # model = GPT2Model.from_pretrained('gpt2')
    # model.config
    # input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
    # outputs = model(input_ids)
    # last_hiddedim_embds = outputs[0]  # The last hidden-state is the first element of the output tuple

    # t = torch.randn(5, 64, 80, 5) 
    # position_ids = torch.arange(5, dtype=torch.long, device=input_ids.device)
    # position_ids = torch.arange(past_length, input_ids.size(-1) + past_length, dtype=torch.long, device=input_ids.device)
    c = {
        "activation_function": "gelu_new",
        "architectures": [
            "GPT2LMHeadModel"
        ],
        "attn_pdrop": 0.1,
        "embd_pdrop": 0.1,
        "initializer_range": 0.02,
        "layer_norm_epsilon": 1e-05,
        "model_type": "gpt2",
        "n_ctx": 1024,
        "dim_embd": 72,
        "dim_p_embd": 64,
        "n_head": 4,
        "n_layer": 12,
        "t_positions": 50,
        "s_positions": 80,
        "num_labels": 1,
        "output_attentions": True,
        "output_hiddedim_embds": False,
        "output_past": True,
        "pruned_heads": {},
        "resid_pdrop": 0.1,
        "summary_activation": None,
        "summary_first_dropout": 0.1,
        "summary_proj_to_labels": True,
        "summary_type": "cls_index",
        "summary_use_proj": True,
        "task_specific_params": {
            "text-generation": {
            "do_sample": True,
            "max_length": 50
            }
        },
        "torchscript": False,
        "use_bfloat16": False,
        "vocab_size": 50
        }
    config=DotDict(c)

    model = GPT2Model(config)
    t = torch.randn(5, 6, 80, 72)
    loss = model(t)

