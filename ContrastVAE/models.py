import torch
import torch.nn as nn
from modules import Encoder, LayerNorm, Decoder, VariationalDropout
import math
import numpy as np
import random
import pdb
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch import nn

class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model, max_len):
        """
        sin, cos encoding 구현
        
        parameter
        - d_model : model의 차원
        - max_len : 최대 seaquence 길이
        - device : cuda or cpu
        """
        
        super(PositionalEncoding, self).__init__() # nn.Module 초기화
        
        # input matrix(자연어 처리에선 임베딩 벡터)와 같은 size의 tensor 생성
        # 즉, (max_len, d_model) size
        self.encoding = torch.zeros(max_len, d_model)
        self.encoding.requires_grad = False # 인코딩의 그래디언트는 필요 없다. 
        
        # 위치 indexing용 벡터
        # pos는 max_len의 index를 의미한다.
        pos = torch.arange(0, max_len).unsqueeze(dim=1)
        # 1D : (max_len, ) size -> 2D : (max_len, 1) size -> word의 위치를 반영하기 위해

#        pos = pos.float().unsqueeze(dim=1) # int64 -> float32 (없어도 되긴 함)
        
        # i는 d_model의 index를 의미한다. _2i : (d_model, ) size
        # 즉, embedding size가 512일 때, i = [0,512]
        _2i = torch.arange(0, d_model, step=2).float()
        # (max_len, 1) / (d_model/2 ) -> (max_len, d_model/2)

        self.encoding[:, ::2] = torch.sin(pos / 10000 ** (_2i / d_model))
        self.encoding[:, 1::2] = torch.cos(pos / 10000 ** (_2i / d_model))
        

    def forward(self, x):
        # self.encoding
        # [max_len = 512, d_model = 512]

        # batch_size = 128, seq_len = 30
        batch_size, seq_len = x.size()
        
        # [seq_len = 30, d_model = 512]
        # [128, 30, 512]의 size를 가지는 token embedding에 더해질 것이다. 
        # 
        return self.encoding[:seq_len, :]
        

class ContrastVAE(nn.Module):

    def __init__(self, args):
        super(ContrastVAE, self).__init__()
        self.mode = None
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size) # position vector 까지 더해줌 
        self.position_encoding = PositionalEncoding(args.hidden_size,args.max_seq_length)
        self.item_encoder_mu = Encoder(self.mode,args) # transformer encoder
        self.item_encoder_logvar = Encoder(self.mode,args)
        self.item_decoder = Decoder(self.mode,args)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.args = args
        self.latent_dropout = nn.Dropout(args.reparam_dropout_rate)
        self.apply(self.init_weights)
        self.temperature = nn.Parameter(torch.zeros(1), requires_grad=True)

    def add_position_embedding(self, sequence):

        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        item_embeddings = self.item_embeddings(sequence) # shape: b*max_Sq*d
        
        position_embeddings = self.position_embeddings(position_ids)
        position_encoding = self.position_encoding(sequence)
        if self.args.encoding :
            sequence_emb = item_embeddings.cuda() + position_encoding.cuda() 
        else:
            sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        return sequence_emb # shape: b*max_Sq*d


    def extended_attention_mask(self, input_ids):
        attention_mask = (input_ids > 0).long()# used for mu, var
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) # torch.int64 b*1*1*max_Sq
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1) # torch.uint8 for causality
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1) #1*1*max_Sq*max_Sq
        subsequent_mask = subsequent_mask.long()

        if self.args.cuda_condition:
            subsequent_mask = subsequent_mask.cuda()

        extended_attention_mask = extended_attention_mask * subsequent_mask #shape: b*1*max_Sq*max_Sq
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        return extended_attention_mask


    def eps_anneal_function(self, step):

        return min(1.0, (1.0*step)/self.args.total_annealing_step)

    def reparameterization(self, mu, logvar, step):  # vanila reparam

        std = torch.exp(0.5 * logvar)
        if self.training:
            eps = torch.randn_like(std)
            res = mu + std * eps
        else: res = mu + std
        return res

    def reparameterization1(self, mu, logvar, step): # reparam without noise
        std = torch.exp(0.5*logvar)
        return mu+std


    def reparameterization2(self, mu, logvar, step): # use dropout

        if self.training:
            std = self.latent_dropout(torch.exp(0.5*logvar))
        else: std = torch.exp(0.5*logvar)
        res = mu + std
        return res

    def reparameterization3(self, mu, logvar,step): # apply classical dropout on whole result
        std = torch.exp(0.5*logvar)
        res = self.latent_dropout(mu + std)
        return res


    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


    def encode(self, sequence_emb, extended_attention_mask,mode): # forward
        if mode : 
            item_encoded_mu_layers = self.item_encoder_mu(sequence_emb,
                                                extended_attention_mask,
                                                output_all_encoded_layers=True,mode = True)
            item_encoded_logvar_layers = self.item_encoder_logvar(sequence_emb, extended_attention_mask,
                                                              output_all_encoded_layers=True,mode = True)
        else:
            item_encoded_mu_layers = self.item_encoder_mu(sequence_emb,
                                                    extended_attention_mask,
                                                    output_all_encoded_layers=True, mode = False)

            item_encoded_logvar_layers = self.item_encoder_logvar(sequence_emb, extended_attention_mask,
                                                                output_all_encoded_layers=True,mode = False)

        return item_encoded_mu_layers[-1], item_encoded_logvar_layers[-1]

    def decode(self, z, extended_attention_mask,mode,ed):
        if mode:
            item_decoder_layers = self.item_decoder(z,
                                                    extended_attention_mask,
                                                    output_all_encoded_layers = True,mode = True, ed = ed)
            sequence_output = item_decoder_layers[-1]
        else: 
            item_decoder_layers = self.item_decoder(z,
                                                    extended_attention_mask,
                                                    output_all_encoded_layers = True,mode = False, ed = ed)
            sequence_output = item_decoder_layers[-1]
        return sequence_output



    def forward(self, input_ids, aug_input_ids, step,ed):

        sequence_emb = self.add_position_embedding(input_ids)# shape: b*max_Sq*d
        extended_attention_mask = self.extended_attention_mask(input_ids)

        if self.args.latent_contrastive_learning:
            if self.args.fft:
                mu1, log_var1 = self.encode(sequence_emb, extended_attention_mask,mode = False)
                mu2, log_var2 = self.encode(sequence_emb, extended_attention_mask,mode = True)
                z1 = self.reparameterization1(mu1, log_var1, step)
                z2 = self.reparameterization2(mu2, log_var2, step)
                reconstructed_seq1 = self.decode(z1, extended_attention_mask,mode = False,ed = ed)
                reconstructed_seq2 = self.decode(z2, extended_attention_mask,mode = True,ed = ed)
            else:
                 mode =  False
                 mu1, log_var1 = self.encode(sequence_emb, extended_attention_mask,mode)
                 mu2, log_var2 = self.encode(sequence_emb, extended_attention_mask,mode)
                 z1 = self.reparameterization1(mu1, log_var1, step)
                 z2 = self.reparameterization2(mu2, log_var2, step)
                 reconstructed_seq1 = self.decode(z1, extended_attention_mask,mode,ed = ed)
                 reconstructed_seq2 = self.decode(z2, extended_attention_mask,mode,ed = ed)               
            return reconstructed_seq1, reconstructed_seq2, mu1, mu2, log_var1, log_var2, z1, z2

        elif self.args.latent_data_augmentation:
            aug_sequence_emb = self.add_position_embedding(aug_input_ids)  # shape: b*max_Sq*d
            aug_extended_attention_mask = self.extended_attention_mask(aug_input_ids)
            if self.args.fft:
                mode = True
                mu1, log_var1 = self.encode(sequence_emb, extended_attention_mask,mode = True)
                mu2, log_var2 = self.encode(aug_sequence_emb, aug_extended_attention_mask,mode = True)
                z1 = self.reparameterization1(mu1, log_var1, step)
                z2 = self.reparameterization2(mu2, log_var2, step)
                reconstructed_seq1 = self.decode(z1, extended_attention_mask,mode = True,ed = ed)
                reconstructed_seq2 = self.decode(z2, extended_attention_mask,mode = True,ed = ed)
            else:
                 mode =  False
                 mu1, log_var1 = self.encode(sequence_emb, extended_attention_mask,mode)
                 mu2, log_var2 = self.encode(aug_sequence_emb, aug_extended_attention_mask,mode)
                 z1 = self.reparameterization1(mu1, log_var1, step)
                 z2 = self.reparameterization2(mu2, log_var2, step)
                 reconstructed_seq1 = self.decode(z1, extended_attention_mask,mode,ed = ed)
                 reconstructed_seq2 = self.decode(z2, extended_attention_mask,mode,ed = ed)  
            return reconstructed_seq1, reconstructed_seq2, mu1, mu2, log_var1, log_var2, z1, z2

        else: # vanilla attentive VAE
            mu, log_var = self.encode(sequence_emb, extended_attention_mask)
            z = self.reparameterization(mu, log_var, step)
            reconstructed_seq1 = self.decode(z, extended_attention_mask)
            return reconstructed_seq1, mu, log_var

 



class ContrastVAE_VD(ContrastVAE):

    def __init__(self, args):
        super(ContrastVAE, self).__init__()

        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)

        self.item_encoder_mu = Encoder(args)
        self.item_encoder_logvar = Encoder(args)
        self.item_decoder = Decoder(args)
 
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.latent_dropout_VD = VariationalDropout(inputshape=[args.max_seq_length, args.hidden_size], adaptive='layerwise')
        self.latent_dropout = nn.Dropout(0.1)
        self.args = args
        self.apply(self.init_weights)

        self.drop_rate = nn.Parameter(torch.tensor(0.2), requires_grad=True)


    def reparameterization3(self, mu, logvar, step): # use drop out

        std, alpha = self.latent_dropout_VD(torch.exp(0.5*logvar))
        res = mu + std
        return res, alpha

    def forward(self, input_ids, augmented_input_ids, step):
        if self.args.variational_dropout:
            sequence_emb = self.add_position_embedding(input_ids)  # shape: b*max_Sq*d
            extended_attention_mask = self.extended_attention_mask(input_ids)
            mu1, log_var1 = self.encode(sequence_emb, extended_attention_mask)
            mu2, log_var2 = self.encode(sequence_emb, extended_attention_mask)
            pdb.set_trace()
            z1 = self.reparameterization1(mu1, log_var1, step)
            z2, alpha = self.reparameterization3(mu2, log_var2, step)
            reconstructed_seq1 = self.decode(z1, extended_attention_mask)
            reconstructed_seq2 = self.decode(z2, extended_attention_mask)

        elif self.args.VAandDA:
            sequence_emb = self.add_position_embedding(input_ids)  # shape: b*max_Sq*d
            extended_attention_mask = self.extended_attention_mask(input_ids)
            aug_sequence_emb = self.add_position_embedding(augmented_input_ids)  # shape: b*max_Sq*d
            aug_extended_attention_mask = self.extended_attention_mask(augmented_input_ids)

            mu1, log_var1 = self.encode(sequence_emb, extended_attention_mask)
            mu2, log_var2 = self.encode(aug_sequence_emb, aug_extended_attention_mask)
            z1 = self.reparameterization1(mu1, log_var1, step)
            z2, alpha = self.reparameterization3(mu2, log_var2, step)
            reconstructed_seq1 = self.decode(z1, extended_attention_mask)
            reconstructed_seq2 = self.decode(z2, extended_attention_mask)


        return reconstructed_seq1, reconstructed_seq2, mu1, mu2, log_var1, log_var2, z1, z2, alpha
