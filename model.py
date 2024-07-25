import torch
import torch.nn as nn
import torch.nn.functional as F

class SignLanguageModel(nn.Module):
    def __init__(self):
        super(SignLanguageModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.lstm = nn.LSTM(32 * 112 * 112, 256, batch_first=True)
        self.fc1 = nn.Linear(256, 10)  # 10개의 수어 클래스 예시

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        c_in = x.view(batch_size * timesteps, C, H, W)
        c_out = self.pool(F.relu(self.conv1(c_in)))
        r_in = c_out.view(batch_size, timesteps, -1)
        r_out, (h_n, c_n) = self.lstm(r_in)
        r_out2 = self.fc1(r_out[:, -1, :])
        return r_out2
