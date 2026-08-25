import torch
import torch.nn as nn
from torch.nn import functional as F
from collections import defaultdict

torch.manual_seed(1337)
batch_size = 32
block_size = 256
dropout = 0.1

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


with open('tinystories.txt' , 'r' , encoding='utf-8') as f:
    text = f.read()

words = text.split()

unique_chars = sorted(list(set(text)))

# BPE:
vocab_merges = {}
sub_words = 1000

word_freq = defaultdict(int)
for w in words:
    token = tuple(list(w) + ['</w>'])
    word_freq[token] += 1

for i in range(sub_words):
    bigram_counts = defaultdict(int)
    for token , freq in word_freq.items():
        for t in range(len(token) - 1):
            bigram = (token[t] , token[t+1])
            bigram_counts[bigram] += freq

    if not bigram_counts:
        break

    best_bigram = max(bigram_counts , key=bigram_counts.get)
    new_token = "".join(best_bigram)
    vocab_merges[best_bigram] = new_token

    new_word_freq = defaultdict(int)
    for tokens , freq in word_freq.items():
        new_tokens = []
        j = 0
        while j < len(tokens):
            if j < len(tokens) - 1 and (tokens[j] , tokens[j+1]) == best_bigram:
                new_tokens.append(new_token)
                j += 2
            else:
                new_tokens.append(tokens[j])
                j += 1
        new_word_freq[tuple(new_tokens)] += freq

    word_freq = new_word_freq 

# encode_word
merge_ranks = {pair: rank for rank, pair in enumerate(vocab_merges.keys())}
def encode_word(word, vocab_merges , merge_ranks):
    tokens = list(word) + ["</w>"]

    while len(tokens) > 1:
        pairs = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
        valid_pairs = [pair for pair in pairs if pair in merge_ranks]
        if not valid_pairs:
            break
        best_pair = min(valid_pairs , key=lambda pair: merge_ranks[pair])

        new_token = vocab_merges[best_pair]
        new_tokens = []

        i = 0
        while i < len(tokens):
            if (i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair):
                new_tokens.append(new_token)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1

        tokens = new_tokens
    return tokens

# vocab
vocab = {}
for char in unique_chars:
    if char not in vocab:
        vocab[char] = len(vocab)

if '</w>' not in vocab:
    vocab['</w>'] = len(vocab)

for token in vocab_merges.values():
    if token not in vocab:
        vocab[token] = len(vocab)

# encode_text
# Solution of bug: (Maintaining a chacing dict)
def encode_text(text , vocab , vocab_merges , merge_ranks):
    words = text.split()
    cache_words = {} # -> memory saving dict (solution)
    ids = []

    for w in words:
        if w not in cache_words:
            tokens = encode_word(w , vocab_merges ,  merge_ranks)
            cache_words[w] = [vocab[t] for t in tokens if t in vocab]
        ids.extend(cache_words[w])

    return ids

# decode:
def decode(idx , vocab):
    ids_token = {idx_val: token for token , idx_val in vocab.items()}
    tokens = [ids_token[id] for id in idx]
    text_out = "".join(tokens)
    return text_out.strip().replace("</w>" , " ")


# Making data (train , val)
data = encode_text(text , vocab ,vocab_merges , merge_ranks)

n = int(0.9*len(data))
train_data = torch.tensor(data[:n] , dtype=torch.long).to(device)
val_data = torch.tensor(data[n:] , dtype=torch.long).to(device)

# get_batch
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(0 , len(data) - block_size , (batch_size ,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x , y = x.to(device) , y.to(device)
    return x , y

vocab_size = len(vocab)
num_emb = 384

class BLM(nn.Module):
    def __init__(self , vocab_size , num_head = 6 , n_layer= 6):
        super().__init__()
        self.embedding_table = nn.Embedding(vocab_size ,num_emb)
        self.positional_table = nn.Embedding(block_size , num_emb)

        self.block = nn.Sequential(*[Block(num_emb , num_head) for _ in range(n_layer)]) # Fix of bug
        self.layernorm = nn.LayerNorm(num_emb) # Fix of bug
        self.lm_head = nn.Linear(num_emb , vocab_size)
        self.drop = nn.Dropout(dropout)

    def forward(self , input_tokens , target=None):
        B,T = input_tokens.shape
        tok_emb = self.embedding_table(input_tokens)
        position = torch.arange(T , device= input_tokens.device)
        pos_emb = self.positional_table(position)
        x = tok_emb + pos_emb
        x = self.block(x)
        x = self.layernorm(x)
        logits = self.lm_head(x)

        if target is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T , C)
            target = target.view(B*T)
            loss = F.cross_entropy(logits , target)
        return logits , loss

    def generate(self , idx , max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cont = idx[: , -block_size:] # -> bringing last 8 tokens
            logits,_ = self(idx_cont)
            logits = logits[: , -1 , :]
            probs = F.softmax(logits , dim=-1)
            idx_next = torch.multinomial(probs , num_samples=1)
            idx = torch.cat((idx , idx_next) , dim=-1)
        return idx

class singleHead(nn.Module):
    def __init__(self , head_size):
        super().__init__()
        self.key = nn.Linear(num_emb , head_size , bias=False)
        self.value = nn.Linear(num_emb , head_size , bias=False)
        self.query = nn.Linear(num_emb , head_size , bias=False)
        self.register_buffer('tril' , torch.tril(torch.ones(block_size , block_size)))

    def forward(self , input):
        B,T,C = input.shape
        k = self.key(input)
        q = self.query(input)
        v = self.value(input)

        wei = q @ k.transpose(-2,-1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T , :T] == 0 ,float('-inf') )
        wei = F.softmax(wei , dim=-1)

        output = wei @ v

        return output

class multiHead(nn.Module):
    def __init__(self , head_size , num_head):
        super().__init__()
        self.heads = nn.ModuleList([singleHead(head_size) for _ in range(num_head)])
        self.proj = nn.Linear(head_size*num_head , head_size*num_head)
    def forward(self , input):
        output = [head(input) for head in self.heads]
        output = torch.cat(output , dim = -1)
        output = self.proj(output)
        return output 

class feedForward(nn.Module):
    def __init__(self , num_emb):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_emb , 4*num_emb),
            nn.GELU(),
            nn.Linear(4*num_emb , num_emb)
        )

    def forward(self , x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self , num_emb , num_head):
        super().__init__()
        head_size = num_emb // num_head

        self.FNN = feedForward(num_emb)
        self.MH = multiHead(head_size , num_head)
        self.layer1 = nn.LayerNorm(num_emb)
        self.layer2 = nn.LayerNorm(num_emb)

    def forward(self , x):
        x = self.MH(self.layer1(x)) + x # -> Fix of bug
        x = self.FNN(self.layer2(x)) + x
        return x

max_loop = 7000
eval_interval = 500
model = BLM(vocab_size).to(device)
# Optimizer
optimizer = torch.optim.AdamW(model.parameters() , lr = 1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer , T_max=max_loop ,  eta_min=1e-4)

for iter in range(max_loop):
    if iter % eval_interval == 0:
         xb , yb = get_batch('val')
         _ , loss = model(xb,yb)
         print(f'step {iter}: val loss {loss.item():.4f}')


    xb , yb = get_batch('train')
    logits , loss = model(xb,yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    scheduler.step()


context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_ids = model.generate(context, max_new_tokens=500)[0].tolist()
print(decode(generated_ids, vocab))
