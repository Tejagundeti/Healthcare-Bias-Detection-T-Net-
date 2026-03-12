import torch
import torch.nn as nn

class TNet(nn.Module):

    def __init__(self, input_dim, output_dim=1):
        super(TNet, self).__init__()

        # Identification Network - Identifies Is_Training_Set (Bias)
        self.identification = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        # Prediction Network - Predicts diseases
        self.prediction = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        id_out = self.identification(x)
        pred_out = self.prediction(x)
        return id_out, pred_out