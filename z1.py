"""Epoch 1, Loss: 2.829571285622609
Epoch 2, Loss: 2.2565109610036993
Epoch 3, Loss: 1.9963111833193417
Epoch 4, Loss: 1.8325975532094465
Epoch 5, Loss: 1.7327025793004764
Epoch 6, Loss: 1.6683569748328764
Epoch 7, Loss: 1.6257525864646944
Epoch 8, Loss: 1.5926514658344886
Epoch 9, Loss: 1.5677802960945528
Epoch 10, Loss: 1.5438033311127575
where are  you of in i to the his of that i is and it my of i the a you in

Sorun: 230 to_token in işlevsiz olması"""

import collections
import re
from d2l import torch as d2l
import random
import torch
from torch import nn
import torch
from torch import nn
import random
from torch.nn import functional as F

# Veri Kümesini Okuma
def read_shakespeare():  #@save
    """Shakespeare veri kümesini bir metin satırı listesine yükleyin."""
    with open('shakespeare.txt', 'r', encoding='utf-8') as file:
        lines = file.readlines()    
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]
lines = read_shakespeare()

# Andıçlama
def tokenize(lines, token='word'): #@save
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('Error: unkown token type: ' + token)
tokens = tokenize(lines)

# Kelime Dağarcığı
class Vocab: #@save
    """Metin için kelime hazinesi"""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # Frekanslara göre sıralama
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)
        # Bilinmeyen andıcın indeksi 0'dır
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) -1
    def __len__(self):
        return len(self.idx_to_token)
    
    def __getitem__(self, tokens):
        if not isinstance(tokens, (list,tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]
    
    @property
    def unk(self):  # Bilinmeyen andıç için dizin
        return 0
    
    @property
    def token_freqs(self): # Bilinmeyen andıç için
        return self._token_freqs
    
def count_corpus(tokens):   #@save
    """Count token frequencies"""
    # Burada 'tokens' bir 1B liste veya 2B listedir.
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # Bir belirteç listelerinin listesini belirteç listesine düzleştirin
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)

### Her Şeyi Birleştir
def load_corpus_shakespeare(max_tokens=-1): #@save
    """Andıç indislerini ve shakespeare veri kümesinin kelime dağarcığını döndür"""
    lines = read_shakespeare()
    tokens = tokenize(lines, 'char')
    vocab = Vocab(tokens)
    # Shakespeare veri kümesindeki her metin satırı mutlaka bir cümle veya
    # paragraf olmadığından, tüm metin satırlarını tek bir listede düzleştir
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab
corpus, vocab = load_corpus_shakespeare()

tokens = d2l.tokenize(read_shakespeare())
# Her metin satırı mutlaka bir cümle veya paragraf olmadığı için tüm metin
# satırlarını bitiştiririz
corpus = [token for line in tokens for token in line]
vocab = Vocab(corpus)
# vocab.token_freqs[:10]
# freqs = [freq for token, freq in vocab.token_freqs]

########### Rastgele Örnekleme##########
def seq_data_iter_random(corpus, batch_size, num_steps):  #@save
    """Rastgele örnekleme kullanarak küçük bir dizi alt dizi oluşturun."""
    # Bir diziyi bölmek için rastgele bir kayma ile başlayın
    # (`num_steps - 1` dahil)
    corpus = corpus[random.randint(0, num_steps - 1):]
    # Etiketleri hesaba katmamız gerektiğinden 1 çıkarın
    num_subseqs = (len(corpus) - 1) // num_steps
    # `num_steps` uzunluğundaki alt diziler için başlangıç indeksleri
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    # Rastgele örneklemede, yineleme sırasında iki bitişik rastgele
    # minigruptan gelen alt diziler, orijinal dizide mutlaka bitişik değildir
    random.shuffle(initial_indices)

    def data(pos):
        # `pos`'dan başlayarak `num_steps` uzunluğunda bir dizi döndür
        return corpus[pos: pos + num_steps]

    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        # Burada, `initial_indices`, alt diziler için rastgele başlangıç
        #  dizinlerini içerir
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y)


# Sıralı Bölümleme
def seq_data_iter_sequential(corpus, batch_size, num_steps):  #@save
    """Sıralı bölümlemeyi kullanarak küçük bir alt dizi dizisi oluşturun."""
    # Bir diziyi bölmek için rastgele bir bağıl konum ile başlayın
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = torch.tensor(corpus[offset: offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y

class SeqDataLoader:  #@save
    """Sıra verilerini yüklemek için bir yineleyici."""
    def __init__(self, corpus, batch_size, num_steps, use_random_iter):
        if use_random_iter:
            self.data_iter_fn = seq_data_iter_random
        else:
            self.data_iter_fn = seq_data_iter_sequential
        self.corpus, self.vocab = load_corpus_shakespeare()
        self.batch_size, self.num_steps = batch_size, num_steps
        self._num_batches = len(self.corpus) // (batch_size * num_steps)

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)

    def __len__(self):
        return self._num_batches

def load_data_shakespeare(batch_size, num_steps,  #@save
                           use_random_iter=False, max_tokens=10000):
    """Shakespeare veri kümesinin yineleyicisini ve sözcük dağarcığını döndür."""
    data_iter = SeqDataLoader(
        corpus=None,  # Burada corpus'u kullanmıyoruz, çünkü zaten SeqDataLoader içinde yükleniyor
        batch_size=batch_size,
        num_steps=num_steps,
        use_random_iter=use_random_iter
    )
    return data_iter, data_iter.vocab

class RNNTextGenerator(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(RNNTextGenerator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded, hidden)
        output = self.fc(output)
        return output, hidden

    def train(model, data_iter, vocab, lr, num_epochs, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        model.to(device)
        for epoch in range(num_epochs):
            total_loss = 0
            for X, Y in data_iter:
                X, Y = X.to(device), Y.to(device)
                optimizer.zero_grad()
                output, _ = model(X)
                # Flatten the output and target tensors
                flat_output = output.view(-1, len(vocab))
                flat_target = Y.view(-1)
                loss = criterion(flat_output, flat_target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f'Epoch {epoch + 1}, Loss: {total_loss / len(data_iter)}')
def generate_text(model, seed_text, vocab, length=100, temperature=1.0, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    with torch.no_grad():
        current_text = seed_text

        for _ in range(length):
            tokens = [vocab[token] for token in current_text.split()]
            input_sequence = torch.tensor(tokens).unsqueeze(0).to(device)
            output, _ = model(input_sequence)
            logits = output[:, -1, :] / temperature
            probabilities = F.softmax(logits, dim=-1)
            predicted_index = torch.multinomial(probabilities, 1).item()
            predicted_token = vocab.to_tokens(predicted_index)
            current_text += " " + predicted_token

    return current_text

# Hyperparameters
vocab_size = len(vocab)
embed_size = 128
hidden_size = 256
num_layers = 2
batch_size = 64
num_steps = 35
lr = 0.001
num_epochs = 10

# Model, Veri Yükleyici ve Eğitim
model = RNNTextGenerator(vocab_size, embed_size, hidden_size, num_layers)
data_iter, _ = load_data_shakespeare(batch_size, num_steps)
model.train(data_iter, vocab, lr=lr, num_epochs=num_epochs)

# Metin Oluşturma
seed_text = "where are "
generated_text = generate_text(model, seed_text, vocab, length=20, temperature=0.8)

print(generated_text)
