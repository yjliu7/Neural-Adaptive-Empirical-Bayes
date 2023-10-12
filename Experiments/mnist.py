import numpy as np
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from keras import backend as K
from NAEB_Bayes import *

torch.manual_seed(1)
np.random.seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_epochs = 600
train_batch_size = 128
test_batch_size = 1000
transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x / 126.),  # divided as in paper
        ])
train_dataset = MNIST(root='data', train=True, transform=transform, download=True)
test_dataset = MNIST(root='data', train=False, transform=transform, download=True)
train_data_loader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=True)
px = 28*28
py = 10
pz = 100
h1 = 400
h2 = 400

model = BayesianNetwork(latent_dim=pz, hidden1_size=h1, hidden2_size=h2, x_dim=px, y_dim=py,
                        decoder_layer_sizes=[128, h1 * (px + 1 + h2) + h2 * (1 + py) + py]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()
train_losses = []
test_losses = []
test_accuracy = []
test_nll = []
train_loss = 0
test_loss = 0
correct = 0
nll_loss = 0

for epoch in range(train_epochs):
    for iteration, (x, y) in enumerate(train_data_loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pi_weight = mini_batch_weight(batch_idx=iteration, num_batches=train_batch_size)
        loss = model.elbo(inputs=x, targets=y, criterion=criterion, n_samples=3, w_complexity=pi_weight)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        if iteration % 100 == 0 or iteration == len(train_data_loader) - 1:
            print("Epoch {:02d}/{:02d} Batch {:04d}/{:d}, Loss {:9.4f}".format(
                epoch, train_epochs, iteration, len(train_data_loader) - 1, loss.item()))
    train_loss /= len(train_data_loader.dataset)
    train_losses.append(train_loss)
    train_loss = 0
    # evaluate on test data set
    with torch.no_grad():
        for iteration, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            pi_weight = mini_batch_weight(batch_idx=iteration, num_batches=test_batch_size)
            loss = model.elbo(inputs=data, targets=target, criterion=criterion, n_samples=3, w_complexity=pi_weight)
            output = model(data)
            test_loss += loss.item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
            # record nll
            y_hat = model.inference(data, 100)
            y_pred = F.softmax(y_hat, dim=2)  # [100, batch_size, 10]
            y_test = nn.functional.one_hot(target, num_classes=10).float()
            y_test = y_test.repeat(100, 1, 1)
            y_test = K.constant(y_test)
            y_pred = K.constant(y_pred)
            g = K.categorical_crossentropy(target=y_test, output=y_pred)
            ce = K.eval(g)
            nll_loss += np.sum(ce)
    test_nll.append(nll_loss / (100 * len(test_loader.dataset)))
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    test_accuracy.append(correct.item())
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), NLL: {}\n'.format(
        test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset), nll_loss))
    test_loss = 0
    correct = 0
    nll_loss = 0
