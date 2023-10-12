import numpy as np
import torch.utils.data
import math
import matplotlib.pyplot as plt
from Bayes import *


torch.manual_seed(1)
np.random.seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def generate_data(num_of_data):
    """
    generate the standardized two-spiral data
    Parameters:
        num_of_data: int
            data size
    Returns:
        tensor: coordinates of all data points
        tensor: labels of all data points
    """
    xs = torch.zeros((num_of_data, 2))
    ys = torch.zeros((num_of_data, 1))
    for data_idx in range(int(num_of_data)):
        if data_idx % 2 == 0:
            spiral_num = 1
            r = (104 - data_idx / 2) / 104
            phi = data_idx / 2 / 16 * math.pi
            xs[data_idx][0] = r * math.sin(phi) * spiral_num
            xs[data_idx][1] = r * math.cos(phi) * spiral_num
            ys[data_idx][0] = 0
        else:
            spiral_num = -1
            r = (104 - (data_idx - 1) / 2) / 104
            phi = (data_idx - 1) / 2 / 16 * math.pi
            xs[data_idx][0] = r * math.sin(phi) * spiral_num
            xs[data_idx][1] = r * math.cos(phi) * spiral_num
            ys[data_idx][0] = 1
    return xs, ys


n = 194
data_x, data_y = generate_data(n)
px = data_x.shape[-1]
py = data_y.shape[-1]
h1 = 50
h2 = 50
pz = 10
model = BayesianNetwork(latent_dim=pz, hidden1_size=h1, hidden2_size=h2, x_dim=px, y_dim=py,
                        decoder_layer_sizes=[128, h1 * (px + 1 + h2) + h2 * (1 + py) + py]).to(device)
best_model = BayesianNetwork(latent_dim=pz, hidden1_size=h1, hidden2_size=h2, x_dim=px, y_dim=py,
                             decoder_layer_sizes=[128, h1 * (px + 1 + h2) + h2 * (1 + py) + py]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.BCEWithLogitsLoss()
x_idx1 = np.arange(n)
x_idx2 = np.arange(n)
batch_size = 10
train_epochs = 3000
loss_epochs = []
for epoch in range(train_epochs):
    np.random.shuffle(x_idx1)
    np.random.shuffle(x_idx2)
    x_idx = np.concatenate((x_idx1, x_idx2[0:6]))
    model.train()
    for i in range(0, n, batch_size):
        batch_id = x_idx[i:(i + batch_size)]
        data = data_x[batch_id].to(device)
        labels = data_y[batch_id].to(device)
        optimizer.zero_grad()
        pi_weight = mini_batch_weight(batch_idx=i // batch_size, num_batches=200 // batch_size)
        loss = model.elbo(inputs=data, targets=labels, criterion=criterion, n_samples=3, w_complexity=pi_weight)
        loss.backward()
        optimizer.step()
        if i % 50 == 0:
            print(f'Train Epoch: {epoch} '
                  f'[{i:05}/{n} '
                  f'\tLoss: {loss.item():.6f}')
    with torch.no_grad():
        predict = model(data_x.to(device))
        bce = criterion(predict, data_y.to(device))
        loss_epochs.append(bce.item())
        if loss_epochs[-1] == min(loss_epochs):
            best_model.load_state_dict(model.state_dict())

# save parameters of the model with minimum loss
torch.save(best_model.state_dict(), 'two_spiral_best_model_params.pkl')

# check the training loss
plt.plot(range(train_epochs), loss_epochs)
plt.savefig("two_spiral_loss.png")

# plot the classification map
data_x_np = data_x.numpy()
colors = ["#000000", "#FFFFFF"]
col = [colors[int(y)] for y in data_y]
x_min, x_max = data_x_np[:, 0].min() - 0.5, data_x_np[:, 0].max() + 0.5
y_min, y_max = data_x_np[:, 1].min() - 0.5, data_x_np[:, 1].max() + 0.5
h = 0.01
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
test_x = np.c_[xx.ravel(), yy.ravel()]
test_x = torch.from_numpy(test_x).type(torch.FloatTensor)
predict_inference = best_model.inference(test_x.to(device), 20)
predict_prob = torch.sigmoid(predict_inference.cpu())
predict_label = torch.zeros((test_x.shape[0],))
for inf_idx in range(20):
    predict_label += (predict_prob[:, inf_idx] > 0.5)
fig = plt.figure()
ax = fig.add_subplot(111)
t = plt.contourf(xx, yy, predict_label.reshape(xx.shape) > 20 / 2, cmap=plt.cm.Spectral, levels=np.linspace(0, 1, 6))
plt.scatter(data_x_np[:, 0], data_x_np[:, 1], c=col)
fig.colorbar(t)
ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable='box')
plt.savefig("two_spiral_classification_map.png")

# plot the difference map between two posterior samples
predict_inference = best_model.inference(test_x.to(device), 2)
predict_prob = torch.sigmoid(predict_inference.cpu())
predict_diff = predict_prob[:, 1] - predict_prob[:, 0]
fig = plt.figure()
ax = fig.add_subplot(111)
t = plt.contourf(xx, yy, predict_diff.reshape(xx.shape), cmap=plt.cm.cividis, levels=np.linspace(-1, 1, 6))
plt.scatter(data_x_np[:, 0], data_x_np[:, 1], c=col)
fig.colorbar(t)
ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable='box')
plt.savefig("two_spiral_difference_map.png")
