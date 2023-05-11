import numpy as np

import torch
import torch.nn as nn

from VAE import Encoder, Decoder
from EASE import EASE



class VASP(nn.Module):

   def __init__(self, hidden_dim, latent_dim, input_dim):
      super().__init__()
 
      self.encoder = Encoder(hidden_dim, latent_dim, input_dim)
      self.decodr = Decoder(hidden_dim, latent_dim, input_dim)
      self.ease = EASE(input_dim)

      self.init_weights(self.encoder)
      self.init_weights(self.decoder)

   
   def forward(self, x):

        mu, logvar = self.encoder(x, dropout_rate=dropout_rate)    
        z = self.reparameterize(mu, logvar)
        x_pred = self.decoder(x)
        x_ease = self.ease(x)
        y = torch.mul(x_pred, ease_x)

        return y
       


   def reparameterize(self, mu, logvar):
      if self.training:
         std = torch.exp(0.5 * logvar)
         eps = torch.randn_like(std)
         return eps*std + mu
      else:
         return mu


   def init_weights(self, m):
      for layer in m:
         if type(layer) == nn.Linear:
            torch.nn.init.xavier_normal_(layer.weight)
            torch.nn.init.normal_(layer.bias, 0, 0.001)


   def loss_function(self, x_pred, x, mu, log_var, anneal=1.0):
    
        neg_ll = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))
        KLD = -0.5 * torch.mean(torch.sum(1 + logvar -
                                       mu.pow(2) - logvar.exp(), dim=1))

      return neg_ll + anneal * 