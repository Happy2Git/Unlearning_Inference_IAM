import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class LRModel(nn.Module):
    def __init__(self, input_dim):
        super(LRModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.training = False

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

    def fit(self, train_loader, num_epochs=100, learning_rate=0.01, verbose=False):
        self.training = True
        self.train()
        criterion = nn.BCELoss()
        optimizer = optim.SGD(self.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            loss_sum = 0
            for inputs, labels in train_loader:
                labels = labels.view(-1, 1).float()
                outputs = self(inputs)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()

            if (epoch+1) % 10 == 0 and verbose:
                print(
                    f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss_sum:.4f}')

    def predict_proba(self, x):
        if self.training:
            self.eval()

        with torch.no_grad():
            probabilities = self(x)
        return probabilities

# Example usage:


if __name__ == '__main__':

    # Generating some random data
    input_dim = 2  # example input dimensions
    n_samples = 100
    x_train = torch.randn(n_samples, input_dim)
    y_train = torch.randint(0, 2, (n_samples, 1)).float()

    # Create Dataset and DataLoader for training
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=10, shuffle=True)

    # Initialize the model
    model = LRModel(input_dim)

    # Fit the model to the training data
    model.fit(train_loader, num_epochs=100, learning_rate=0.01)

    # Predicting probabilities on new data
    new_data = torch.tensor([[0.5, -0.1],
                            [0.3,  0.0]], dtype=torch.float32)
    probabilities = model.predict_proba(new_data)
    print(probabilities)
