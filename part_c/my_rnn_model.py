import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Hyperparameters
HIDDEN_DIM = 128
NUM_LAYERS = 2  # Stacked BiLSTM
BATCH_SIZE = 32
LR = 0.001
EPOCHS = 20  # Increased for better validation selection
VOCAB_SIZE = 10000  # Increased for word embeddings
EMBEDDING_DIM = 100  # GloVe embedding size
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 500

# Load IMDB dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=VOCAB_SIZE)
# Διαχωρισμός των δεδομένων σε training και validation
x_train, x_dev, y_train, y_dev = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# Pad sequences ώστε να έχουν σταθερό μήκος
x_train = pad_sequences(x_train, maxlen=MAX_LENGTH, padding="post", truncating="post")
x_dev = pad_sequences(x_dev, maxlen=MAX_LENGTH, padding="post", truncating="post")
x_test = pad_sequences(x_test, maxlen=MAX_LENGTH, padding="post", truncating="post")

# Convert σε numpy arrays με συγκεκριμένο τύπο δεδομένων
x_train = np.array(x_train, dtype=np.int32)
x_dev = np.array(x_dev, dtype=np.int32)
x_test = np.array(x_test, dtype=np.int32)

# Load GloVe embeddings
embedding_index = {}
with open("glove.6B.100d.txt", encoding="utf-8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = vector

# Create embedding matrix
word_index = tf.keras.datasets.imdb.get_word_index()
embedding_matrix = np.zeros((VOCAB_SIZE, EMBEDDING_DIM))
for word, i in word_index.items():
    if i < VOCAB_SIZE and word in embedding_index:
        embedding_matrix[i] = embedding_index[word]

# Dataset class
class IMDBDataset(Dataset):
    def __init__(self, X, y, device):
        self.X = torch.tensor(X, dtype=torch.long).to(device)
        self.y = torch.tensor(y, dtype=torch.float32).to(device)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_data = IMDBDataset(x_train, y_train, DEVICE)
dev_data = IMDBDataset(x_dev, y_dev, DEVICE)
test_data = IMDBDataset(x_test, y_test, DEVICE)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
dev_loader = DataLoader(dev_data, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

# Stacked BiLSTM with Global Max Pooling
class StackedBiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, pretrained_weights):
        super(StackedBiLSTM, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(pretrained_weights, dtype=torch.float32), freeze=False)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = torch.max(x, dim=1)[0]  # Global max pooling
        x = self.dropout(x)
        return torch.sigmoid(self.fc(x)).squeeze(1)

model = StackedBiLSTM(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, embedding_matrix).to(DEVICE)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

def train(model, train_loader, dev_loader, criterion, optimizer, epochs):
    best_f1 = 0
    train_losses, dev_losses = [], []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for texts, labels in train_loader:
            texts, labels = texts.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_losses.append(total_loss / len(train_loader))

        # Validation loss
        model.eval()
        dev_loss = 0
        y_true, y_pred = [], []
        with torch.no_grad():
            for texts, labels in dev_loader:
                texts, labels = texts.to(DEVICE), labels.to(DEVICE)
                outputs = model(texts)
                loss = criterion(outputs, labels)
                dev_loss += loss.item()
                y_true.extend(labels.cpu().numpy())
                y_pred.extend((outputs.cpu().numpy() > 0.5).astype(int))
        dev_losses.append(dev_loss / len(dev_loader))

        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}, Dev Loss: {dev_loss / len(dev_loader):.4f}, F1: {f1:.4f}")
        
        # Early stopping
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), "best_model.pth")
    
    # Plot loss curves
    plt.plot(range(1, epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs+1), dev_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss Curves')
    plt.show()

def evaluate(model, test_loader):
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for texts, labels in test_loader:
            texts, labels = texts.to(DEVICE), labels.to(DEVICE)
            outputs = model(texts) > 0.5
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(outputs.cpu().numpy())
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    print(f"Test Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

train(model, train_loader, dev_loader, criterion, optimizer, EPOCHS)
evaluate(model, test_loader)
