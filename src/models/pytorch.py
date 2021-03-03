""" Create in src/models/pytorch.py a class called PytorchRegression that inherits
from nn.Module with:

num_features as input parameter
attributes:
layer_1: fully-connected layer with 128 neurons
layer_out: fully-connected layer with 1 neurons
methods:
forward() with inputs as input parameter, perform ReLU and DropOut on the
fully-connected layer followed by the output layer"""

class PytorchRegression(nn.Module):
    def __init__(self, num_features):
        super(PytorchRegression, self).__init__()

        self.layer_1 = nn.Linear(num_features, 128)
        self.layer_out = nn.Linear(128, 1)

    def forward(self, x):
        x = F.dropout(F.relu(self.layer_1(x)))
        x = self.layer_out(x)
        return (x)

"""Create in src/models/pytorch.py a function called get_device() with:

Logics: check if cuda is available and return cuda:0 if that is the case cpu otherwise
Output: device to be used by Pytorch"""
def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu') # don't have GPU
    return device


class PytorchDataset(Dataset):
    """
    Pytorch dataset
    ...

    Attributes
    ----------
    X_tensor : Pytorch tensor
        Features tensor
    y_tensor : Pytorch tensor
        Target tensor

    Methods
    -------
    __getitem__(index)
        Return features and target for a given index
    __len__
        Return the number of observations
    to_tensor(data)
        Convert Pandas Series to Pytorch tensor
    """

    def __init__(self, X, y):
        self.X_tensor = self.to_tensor(X)
        self.y_tensor = self.to_tensor(y)

    def __getitem__(self, index):
        return self.X_tensor[index], self.y_tensor[index]

    def __len__(self):
        return len(self.X_tensor)

    def to_tensor(self, data):
        return torch.Tensor(np.array(data))