import torch
import torch.nn as nn
import torch.nn.functional as F

# hyperparameters
# ~~~~~~~~~~~~~~~~~~~~ too large for cpu? check before running ~~~~~~~~~~~~
batch_size = 64
block_size = 256
max_steps = 5000
eval_interval = 500 # after how many steps we should evaluate train & val loss
learning_rate = 3e-4
# use device -
# 1. when creating model to transfer parameters
# 2. when fetching batch data for training
# 3. when generating data for sample/output creation
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200 # how many batches to be evaluated
n_embd = 384
n_heads = 6
n_layer = 6
dropout = 0.2
# ----------------

torch.manual_seed(1337)

# Data reading
## wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Data processing
## all unique characters    
chars = sorted(set(text))
vocab_size = len(chars)
## creating vocabulary and mapping and reverse mapping
itos = {i:c for i, c in enumerate(chars)}
stoi = {c:i for i, c in itos.items()}
encode = lambda s : [stoi[c] for c in s]
decode = lambda e : ''.join([itos[i] for i in e])
## train, val split
data = torch.tensor(encode(text), dtype=torch.long)
split = int(0.9*len(data))
train_data = data[:split]
val_data = data[split:]

def get_batch(split):
  data = train_data if split=='train' else val_data
  batch_ixs = torch.randint(0, len(data) - block_size, (batch_size,))
  x = torch.stack([data[ix : ix + block_size] for ix in batch_ixs])
  y = torch.stack([data[ix + 1: ix + 1 + block_size] for ix in batch_ixs])
  '''
  if x indices are  [3,4,5,6,7,8,9, 10]
  y indices will be [4,5,6,7,8,9,10,11]
  torch.stack will just stack in one extra dimension in front
  '''
  x, y = x.to(device), y.to(device)
  return x, y

# function to find intermittent_loss without affecting grads
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Block(nn.Module):
  
  def __init__(self, n_embd, n_heads):
    super().__init__()
    head_size = n_embd//n_heads
    self.sa_heads = MultiHeadAttention(n_heads, head_size) # 4 heads of 8-dimensional self attention
    self.feed_forward = FeedForward(n_embd)
    self.ln1 = nn.LayerNorm(n_embd)
    self.ln2 = nn.LayerNorm(n_embd)
    
  def forward(self, x):
    # ToDo - why layer norm before heads and ff.
    # ToDo - figure out and read about layer normalization why it works and whats the intuition
    # - https://arxiv.org/abs/1607.06450
    x = x + self.sa_heads(self.ln1(x))
    x = x + self.feed_forward(self.ln2(x))
    return x

# point-wise feed forward --> FFN(x) = max(0, xW1 + b1)W2 + b2
class FeedForward(nn.Module):
  
  def __init__(self, n_embd):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(n_embd, 4 * n_embd),
      nn.ReLU(),
      nn.Linear(4 * n_embd, n_embd)
    )
    self.dropout = nn.Dropout(dropout)
    
  def forward(self, x):
    return self.dropout(self.net(x))

class Head(nn.Module):
  
  def __init__(self, head_size):
    super().__init__()
    self.key = nn.Linear(n_embd, head_size, bias=False)
    self.query = nn.Linear(n_embd, head_size, bias=False)
    self.value = nn.Linear(n_embd, head_size, bias=False)
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # because tril is not paramter to model. just constant save once.
    
    self.dropout = nn.Dropout(dropout)
  
  def forward(self, x):
    B, T, C = x.shape
    k = self.key(x)    # (B,T,head_size)
    q = self.query(x)  # (B,T,head_size)
    wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5 # k.shape[-1] == head_size
    # (B,T,h)@(B,h,T) --> (B,T,T) | now, this `wei` is interaction matrix not just zero or uniform
    # scaling is to control variance of `wei` before feeding to softmax. Otherwise if this scaling is not done, weights will be very high and softmax will cnvert focus to very small number of other tokens
    # now, we can do the same ops - masking, softmax & multiplication (but to v similar to  k & q)
    
    wei = wei.masked_fill(self.tril[:T,:T]==0, value=float('-inf')) # this masking is what makes this as decoder. Otherwise, encoder can look upto all chars in given input
                                                        # also, difference in decoder is it has another sublayer (only used if encoder is also in place) which takes encoder's input in k & v, and q remains from decoder only. Also, only in first layer or all layers? ToDo - figure out
    wei = torch.softmax(wei, dim=-1) # (B,T,T)
    wei = self.dropout(wei)
    v = self.value(x) # (B,T,C) -> (B,T,h)
    
    out = wei @ v # (B,T,T)@(B,T,h) --> (B,T,h)
    return out

class MultiHeadAttention(nn.Module):
  
  def __init__(self, num_heads, head_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
    self.proj = nn.Linear(n_embd, n_embd)
    self.dropout = nn.Dropout(dropout)
    
  def forward(self, x):
    out = torch.cat([h(x) for h in self.heads], dim=-1)
    out = self.proj(out)
    out = self.dropout(out)
    return out

# super simple/nano gpt language model
class SimpleGPTLanguageModel(nn.Module):
  
  def __init__(self):
    super().__init__()
    self.token_embedding = nn.Embedding(vocab_size, n_embd)
    self.position_embedding = nn.Embedding(block_size, n_embd)
    self.blocks = nn.Sequential(*[Block(n_embd, n_heads=n_heads) for _ in range(n_layer)]) # * --> unrolling the list
    self.ln_f = nn.LayerNorm(n_embd) # why is this needed? final layer norm
    self.lm_head = nn.Linear(n_embd, vocab_size)

  def forward(self, x, targets=None):
    B, T = x.shape
    tok_embd = self.token_embedding(x) # converts shape(B, T) to shape(B, T, C) | B-batch, T-time(context) dimension, C-Channel(Embedding_size)
    pos_embd = self.position_embedding(torch.arange(T, device=device)) # shape : (T, C) # C here is n_embd
    x = tok_embd + pos_embd # (B,T,C) + (T,C) : implicit broadcasting
    x = self.blocks(x)
    x = self.ln_f(x)
    logits = self.lm_head(x) # (B, T, vocab_size)
    
    loss = None
    if targets is not None:
      B,T,C = logits.shape
      # read documentation of cross_entropy for reason of this reshaping -
      # https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html#torch.nn.functional.cross_entropy
      logits = logits.view(B*T, C)
      targets = targets.view(B*T) # or just -1
      loss = F.cross_entropy(logits, targets)

    return logits, loss

  def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
      idx_cond = idx[:, -block_size:]
      logits, _ = self(idx_cond)             # forward pass to get logits, loss is not needed
      logits = logits[:, -1, :]         # logits will be of shape B,T,C but we only care about last character i.e. last T
      probs = F.softmax(logits, dim=1)  # prob using softmax along the channel dimension
      next_idx = torch.multinomial(probs, 1)
      # that is, we are training with (4,8) -> (4,8) shape. But feeding(1, +inf) shape to get (1, +inf) out and then just consume only last T.
      # fixed by cropping index to maximum block_size
      # validate this fact
      #print(f"debug | {decode(idx[0].numpy())} --> {decode([next_idx[0].item()])}") # 0-index to take from first batch
      idx = torch.cat([idx, next_idx], dim=1)
    return idx

# create model
model = BigramLanguageModel()
model.to(device)

# generate output with no context
def generate_with_no_context(max_new_tokens=20):
  #model.generate(torch.tensor([stoi[' ']]*block_size).view(1, -1), max_new_tokens=10)
  inp = torch.zeros((1,1), dtype=torch.long, device=device) # zero is newline char
  out = model.generate(inp, max_new_tokens=max_new_tokens)
  print(decode(out[0].tolist())) # 0-index to pick batch index

generate_with_no_context()

# our favourite optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# training steps
for step in range(max_steps):
  if step % eval_interval == 0:
    losses = estimate_loss()
    print(f"{step=} : train loss = {losses['train']:.4f} val loss : {losses['val']:.4f}")
  
  xb, yb = get_batch('train')
  logits, loss = model(xb, yb)
  optimizer.zero_grad(set_to_none=True) # this set_to_none is for performance optimization. but can lead to error if try to access grad
  loss.backward()
  optimizer.step()
    
# sample output after training
generate_with_no_context(max_new_tokens=200)
