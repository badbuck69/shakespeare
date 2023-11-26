import time
import random
import unidecode
import string
import re
import matplotlib.pyplot as plt
import torch

torch.backends.cudnn.deterministic = True

RANDOM_SEED = 123
torch.manual_seed(RANDOM_SEED)
DEVICE = torch.device('cpu')
TEXT_PORTION_SIZE = 200
NUM_ITER = 5000
LEARNING_RATE = 0.005
EMBEDDING_DIM = 100
HIDDEN_DIM = 128

string.printable
with open('shakespeare.txt', 'r') as f:
    textfile = f.read()
textfile = unidecode.unidecode(textfile)
textfile = re.sub(' +',' ', textfile)
TEXT_LENGTH = len(textfile)
random.seed(RANDOM_SEED)

def random_portion(textfile):
    start_index = random.randint(0, TEXT_LENGTH - TEXT_PORTION_SIZE)
    end_index = start_index + TEXT_PORTION_SIZE + 1
    return textfile[start_index:end_index]

def char_to_tensor(text):
    lst = [string.printable.index(c) for c in text]
    tensor = torch.tensor(lst).long()
    return tensor
def draw_random_sample(textfile):
    text_long = char_to_tensor(random_portion(textfile))
    inputs = text_long[:-1]
    targets = text_long[1:]
    return inputs, targets

class RNN(torch.nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, output_size):
        super().__init__()

        self.hidden_size = hidden_size
        self.embed = torch.nn.Embedding(num_embeddings=input_size, embedding_dim=embed_size)
        self.rnn = torch.nn.LSTMCell(input_size=embed_size, hidden_size=hidden_size)
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, character, hidden, cell_state):
        embedded = self.embed(character)
        (hidden, cell_state) = self.rnn(embedded, (hidden, cell_state))
        output = self.fc(hidden)
        return output, hidden, cell_state
    
    def init_zero_state(self):
        init_hidden = torch.zeros(1, self.hidden_size).to(DEVICE)
        init_cell = torch.zeros(1, self.hidden_size).to(DEVICE)
        return (init_hidden, init_cell)
    
torch.manual_seed(RANDOM_SEED)
model = RNN(len(string.printable), EMBEDDING_DIM, HIDDEN_DIM, len(string.printable))
model = model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Eğitim
def evaluate(model, prime_str='A', predict_len=100, temperature=0.8):
    (hidden, cell_state) = model.init_zero_state()
    prime_input = char_to_tensor(prime_str)
    predicted = prime_str

    for p in range(len(prime_str) - 1):
        inp = prime_input[p].unsqueeze(0)
        _, hidden, cell_state = model(inp.to(DEVICE), hidden, cell_state)
    inp = prime_input[-1].unsqueeze(0)

    for p in range(predict_len):
        outputs, hidden, cell_state = model(inp.to(DEVICE), hidden, cell_state)

        output_dist = (outputs.view(-1)/(temperature)).exp()
        top_i = torch.multinomial(output_dist, 1)[0]
        
        predicted_char = string.printable[top_i]
        predicted += predicted_char
        inp = char_to_tensor(predicted_char)
    return predicted

start_time = time.time()
loss_list = []

for iteration in range(NUM_ITER):
    hidden, cell_state = model.init_zero_state()
    optimizer.zero_grad()

    loss = 0.
    inputs, targets = draw_random_sample(textfile)
    inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

    for c in range(TEXT_PORTION_SIZE):
        outputs, hidden, cell_state = model(inputs[c].unsqueeze(0), hidden, cell_state)
        loss += torch.nn.functional.cross_entropy(outputs, targets[c].view(1))

    loss /= TEXT_PORTION_SIZE
    loss.backward()

    #UPDATE MODEL PARAMETERS
    optimizer.step()

    #LOGGING
    with torch.no_grad():
        if iteration % 200 == 0:
            print(f'Time elapsed: {(time.time() - start_time)/60:.2f} min')
            print(f'Iteration {iteration} | Loss {loss.item():.2f}\n\n')
            print(evaluate(model, 'Th', 200), '\n')

            loss_list.append(loss.item())
            plt.clf()
            plt.plot(range(len(loss_list)), loss_list)
            plt.ylabel('Loss')
            plt.xlabel('Iteration x 1000')
plt.clf()
plt.plot(range(len(loss_list)), loss_list)
plt.ylabel('Loss')
plt.xlabel('Iteration x 1000')
plt.show()