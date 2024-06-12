import torch
import torch.nn as nn
import torch.nn.functional as F

# hyperparameters
batch_size = 32
block_size = 8
max_steps = 10000
eval_interval = 100 # after how many steps we should evaluate train & val loss
learning_rate = 1e-2
# use device -
# 1. when creating model to transfer parameters
# 2. when fetching batch data for training
# 3. when generating data for sample/output creation
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200 # how many batches to be evaluated
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

# super simple bigram model
class BigramLanguageModel(nn.Module):
  def __init__(self, vocab_size):
    super().__init__()
    self.token_embedding = nn.Embedding(vocab_size, vocab_size)

  def forward(self, x, targets=None):
    logits = self.token_embedding(x) # converts shape(B, T) to shape(B, T, C) | B-batch, T-?, C-Channel(Embedding_size)
    
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
      logits, _ = self(idx)             # forward pass to get logits, loss is not needed
      logits = logits[:, -1, :]         # logits will be of shape B,T,C but we only care about last character i.e. last T
      probs = F.softmax(logits, dim=1)  # prob using softmax along the channel dimension
      next_idx = torch.multinomial(probs, 1)
      # that is, we are training with (4,8) -> (4,8) shape. But feeding(1, +inf) shape to get (1, +inf) out and then just consume only last T.
      # validate this fact
      #print(f"debug | {decode(idx[0].numpy())} --> {decode([next_idx[0].item()])}") # 0-index to take from first batch
      idx = torch.cat([idx, next_idx], dim=1)
    return idx

# create model
model = BigramLanguageModel(vocab_size)
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