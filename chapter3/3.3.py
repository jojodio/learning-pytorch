import numpy as np
import torch
from torch import nn

num_input = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.tensor(np.random.normal(0, 1, (num_examples, num_input)), dtype=torch.float)
y = true_w[0] * features[:,0] + true_w[1] * features[:,1] + true_b
y += torch.tensor(np.random.normal(0, 0.01, size=features[:,0].size()), dtype=torch.float)

import torch.utils.data as Data
dataset = Data.TensorDataset(features, y)
data_iter = Data.DataLoader(
    dataset=dataset,
    batch_size=10,
    shuffle= True,
    num_workers= 4,
)
for x,y in data_iter:
    print(x,'\n', y)
    break

class LinearNet(nn.Module):
    def __init__(self, n_features):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        y = self.linear(x)
        return y

net = LinearNet(num_input)
net = nn.Sequential(
    nn.Linear(num_input, 1)
    # 此处还可以传入其他层
    )
print(net)

for param in net.parameters():
    print(param)

from torch.nn import init
init.normal_(net[0].weight, mean=0, std=0.01)
init.constant_(net[0].bias, val=0)
for param in net.parameters():
    print(param)

loss = nn.MSELoss()

from torch.optim import SGD
optimer = SGD(net.parameters(), lr=0.03)
print(optimer)

num_epochs = 20
for epoch in range(num_epochs):
    for x,y in data_iter:
        out = net(x)
        l = loss(out, y.view(-1,1))
        optimer.zero_grad()
        l.backward()
        optimer.step()

sense = net[0]
print(true_w,sense.weight)
print(true_b, sense.bias)

