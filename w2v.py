
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from board import Hive
from encoding import OneHotEncoder
hive = Hive()

hps = [hive._piece_set('w'), hive._piece_set('b')]

keys_w, keys_b = [list(h.keys()) for h in hps]
encoder_w, encoder_b = [OneHotEncoder(keys) for keys in [keys_w, keys_b]]
one_hot_encoded_w, one_hot_encoded_b = encoder_w.get_encoded(), encoder_b.get_encoded()

v_w, v_b = [encoder_w.set_values_to_integers(v) for v in hps]

embeds = nn.Embedding(len(v_w.keys()), 2)  #words in vocab = len(v_w), 10 dimensional embeddings
lookup_tensor = torch.tensor([v_w["wS1"]], dtype=torch.long)
embd = embeds(lookup_tensor)
print(embd)