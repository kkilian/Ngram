
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from board import Hive
from encoding import OneHotEncoder
from hive_log_processing import get_logs, LogParser

from hive_log_processing import parse_logs
logs = parse_logs(get_logs())
log_parser = LogParser(logs)
all_keys, white_keys, black_keys, draw_keys = log_parser.parse_logs_category()

def extract_words_context(logs, key):
    pieces = []
    rows = logs[key]
    rows_count = len(rows)
    
    for i, row in enumerate(rows):
        if len(row) == 3:
            pieces.append(''.join(row[1:]))
    return pieces



hive = Hive()
"""

encoder = OneHotEncoder(list(hps.keys()))
encoder = encoder.get_encoded()"""
def create_ngrams(test_sentence, context):
    ngrams = [
        (
            [test_sentence[i - j - 1] for j in range(context)],
            test_sentence[i]
        )
        for i in range(context, len(test_sentence))
    ]
    return ngrams


test_sentence =extract_words_context(logs, all_keys[1])
context = 3
ngrams = create_ngrams(test_sentence, context)

vocab = set(test_sentence)
word_to_ix = {word: i for i, word in enumerate(vocab)}
ix_to_word = {i: word for i, word in enumerate(vocab)}

class NGramLanguageModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()  
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, _freeze = False)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

dims = 3
losses = []
loss_function = nn.NLLLoss()
model = NGramLanguageModeler(len(vocab), dims , context)
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(20):
    total_loss = 0
    for context, target in ngrams:
        context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)
        model.zero_grad()


        log_probs = model(context_idxs)
        print(ix_to_word[log_probs.argmax().item()], [target])
        loss = loss_function(log_probs, torch.tensor([word_to_ix[target]], dtype=torch.long))

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(epoch, loss.item())
    losses.append(total_loss)

print(model.embeddings.weight[word_to_ix["wG1."]])
