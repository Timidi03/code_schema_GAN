import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torchaudio.transforms import Resample
import os

def preprocess_audio(audio_path, max_length=307332, orig_freq=44100, new_freq=16000):
    waveform, sample_rate = torchaudio.load(audio_path)
    resampler = Resample(orig_freq=orig_freq, new_freq=new_freq)
    if sample_rate != new_freq:
        waveform = resampler(waveform)
    if waveform.shape[0] == 2:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    current_length = waveform.size(1)
    if current_length < max_length:
        padding_size = max_length - current_length
        left_padding = padding_size // 2
        right_padding = padding_size - left_padding
        waveform = torch.nn.functional.pad(waveform, (left_padding, right_padding), "constant")
    
    return waveform

def predict_audio(model, audio_path, number_to_command, max_length=307332, orig_freq=44100, new_freq=16000):
    waveform = preprocess_audio(audio_path, max_length, orig_freq, new_freq)
    waveform = waveform.unsqueeze(0)
    
    with torch.no_grad():
        output = model(waveform)
  
    _, predicted = torch.max(output, 1)
    
    return number_to_command[predicted.item()]

number_to_command = {
    1: "Yuri",
    2: "MainPage",
    3: "Help",
    4: "Restart",
    5: "Back",
    0: "Other"
}


class AudioCNN(nn.Module):
    def __init__(self, num_classes):
        super(AudioCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=80, kernel_size=80)
        self.bn1 = nn.BatchNorm1d(num_features=80)
        self.pool = nn.MaxPool1d(kernel_size=4, stride=4)

        self.conv2 = nn.Conv1d(in_channels=80, out_channels=64, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(num_features=64)

        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(num_features=128)

        self.conv4 = nn.Conv1d(in_channels=128, out_channels=32, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(num_features=32)

        self.fc1 = nn.Linear(38368, out_features=num_classes)

    def forward(self, x):
      x = self.pool(F.relu(self.bn1(self.conv1(x))))
      x = self.pool(F.relu(self.bn2(self.conv2(x))))
      x = self.pool(F.relu(self.bn3(self.conv3(x))))
      x = self.pool(F.relu(self.bn4(self.conv4(x))))
      x = x.view(x.size(0), -1)
      x = self.fc1(x)
      return x


model = AudioCNN(num_classes=6)

audio_path = 'Yuri_1.wav'

state_dict = torch.load('/content/weights_audio/best.pt')
model = AudioCNN(6)
model.load_state_dict(state_dict)
model.eval()
prediction = predict_audio(model, audio_path, number_to_command)

print(f"Предсказание: {prediction}")