import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Define the MLP model


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=16, output_dim=10):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.layers(x)

    def fit(self, train_loader, num_epochs=100, learning_rate=0.01, verbose=False):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            loss_sum = 0
            for inputs, targets in train_loader:
                outputs = self(inputs)
                loss = criterion(outputs, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()

            if (epoch+1) % 10 == 0 and verbose:
                print(
                    f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss_sum:.4f}')

    def predict_proba(self, x):
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            logits = self(x)
            probabilities = nn.functional.softmax(logits, dim=1)
        return probabilities

# Main function to test the MLP

# Main function to test the MLP


def main():
    # Define the dimensions of the input, hidden layers, and output
    input_dim = 20
    hidden_dim = 16
    num_classes = 3  # For multi-class classification

    # Generate some synthetic data for demonstration purposes
    X, y = make_classification(n_samples=1000, n_features=input_dim,
                               n_informative=10, n_classes=num_classes,
                               random_state=42)
    X = StandardScaler().fit_transform(X)  # Feature scaling for better performance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    # Use long for CrossEntropyLoss
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    # Create the dataset and dataloader for batch processing
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=64, shuffle=True)

    # Initialize the MLP model
    model = MLP(input_dim, hidden_dim, num_classes)

    # Train the model
    model.fit(train_loader, num_epochs=100, learning_rate=0.01)

    # Predict probabilities on some example test data
    probabilities = model.predict_proba(X_test_tensor)
    print("Predicted probabilities:\n", probabilities)


if __name__ == "__main__":
    main()
