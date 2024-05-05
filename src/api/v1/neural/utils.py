# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchaudio
# from torchaudio.transforms import Resample
# import os

# def preprocess_audio(audio_path, max_length=307332, orig_freq=44100, new_freq=16000):
#     waveform, sample_rate = torchaudio.load(audio_path)
#     resampler = Resample(orig_freq=orig_freq, new_freq=new_freq)
#     if sample_rate != new_freq:
#         waveform = resampler(waveform)
#     if waveform.shape[0] == 2:
#         waveform = torch.mean(waveform, dim=0, keepdim=True)
#     current_length = waveform.size(1)
#     if current_length < max_length:
#         padding_size = max_length - current_length
#         left_padding = padding_size // 2
#         right_padding = padding_size - left_padding
#         waveform = torch.nn.functional.pad(waveform, (left_padding, right_padding), "constant")
    
#     return waveform

# def predict_audio(model, audio_path, number_to_command, max_length=307332, orig_freq=44100, new_freq=16000):
#     waveform = preprocess_audio(audio_path, max_length, orig_freq, new_freq)
#     waveform = waveform.unsqueeze(0)
    
#     with torch.no_grad():
#         output = model(waveform)
  
#     _, predicted = torch.max(output, 1)
    
#     return number_to_command[predicted.item()]

# def recognize_file(filename):
#     model = AudioCNN(num_classes=6)

#     audio_path = f'E:/VSCode/Code_schema_GAN/media/{filename}'

#     if torch.cuda.is_available():
#         state_dict = torch.load('E:/VSCode/code_schema_GAN/src/api/v1/neural/best.pt')
#     else:
#         state_dict = torch.load('E:/VSCode/code_schema_GAN/src/api/v1/neural/best.pt', map_location=torch.device('cpu'))
#     model = AudioCNN(6)
#     model.load_state_dict(state_dict)
#     model.eval()
#     prediction = predict_audio(model, audio_path, number_to_command)
#     return prediction

# number_to_command = {
#     1: "Yuri",
#     2: "MainPage",
#     3: "Help",
#     4: "Restart",
#     5: "Back",
#     0: "Other"
# }


# class AudioCNN(nn.Module):
#     def __init__(self, num_classes):
#         super(AudioCNN, self).__init__()
#         self.conv1 = nn.Conv1d(in_channels=1, out_channels=80, kernel_size=80)
#         self.bn1 = nn.BatchNorm1d(num_features=80)
#         self.pool = nn.MaxPool1d(kernel_size=4, stride=4)

#         self.conv2 = nn.Conv1d(in_channels=80, out_channels=64, kernel_size=3)
#         self.bn2 = nn.BatchNorm1d(num_features=64)

#         self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3)
#         self.bn3 = nn.BatchNorm1d(num_features=128)

#         self.conv4 = nn.Conv1d(in_channels=128, out_channels=32, kernel_size=3)
#         self.bn4 = nn.BatchNorm1d(num_features=32)

#         self.fc1 = nn.Linear(38368, out_features=num_classes)

#     def forward(self, x):
#       x = self.pool(F.relu(self.bn1(self.conv1(x))))
#       x = self.pool(F.relu(self.bn2(self.conv2(x))))
#       x = self.pool(F.relu(self.bn3(self.conv3(x))))
#       x = self.pool(F.relu(self.bn4(self.conv4(x))))
#       x = x.view(x.size(0), -1)
#       x = self.fc1(x)
#       return x


# from main import FilterbankFeaturesTA, AudioToMelSpectrogramPreprocessor

import torch
from omegaconf import OmegaConf
import torchaudio
from typing import Optional, Tuple, Union
import random
from pydub import AudioSegment
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from nemo.collections.asr.models import EncDecCTCModel

from nemo.collections.asr.parts.preprocessing.features import FilterbankFeaturesTA as NeMoFilterbankFeaturesTA
from nemo.collections.asr.modules.audio_preprocessing import AudioToMelSpectrogramPreprocessor as NeMoAudioToMelSpectrogramPreprocessor

import locale
locale.getpreferredencoding = lambda: "UTF-8"

BRANCH = 'r1.21.0'

class FilterbankFeaturesTA(NeMoFilterbankFeaturesTA):
    def __init__(self, mel_scale: str = 'htk', wkwargs=None, **kwargs):
        if 'window_size' in kwargs:
            del kwargs['window_size']
        if 'window_stride' in kwargs:
            del kwargs['window_stride']

        super().__init__(**kwargs)

        self._mel_spec_extractor: torchaudio.transforms.MelSpectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self._sample_rate,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_mels=kwargs['nfilt'],
            window_fn=self.torch_windows[kwargs['window']],
            mel_scale=mel_scale,
            norm=kwargs['mel_norm'],
            n_fft=kwargs['n_fft'],
            f_max=kwargs.get('highfreq', None),
            f_min=kwargs.get('lowfreq', 0),
            wkwargs=wkwargs,
        )

class AudioToMelSpectrogramPreprocessor(NeMoAudioToMelSpectrogramPreprocessor):
    def __init__(self, mel_scale: str = 'htk', **kwargs):
        super().__init__(**kwargs)
        kwargs['nfilt'] = kwargs['features']
        del kwargs['features']
        self.featurizer = FilterbankFeaturesTA(  # Deprecated arguments; kept for config compatibility
            mel_scale=mel_scale,
            **kwargs,
        )


model = EncDecCTCModel.from_config_file('./api/v1/neural/ctc_model_config.yaml')
ckpt = torch.load('./api/v1/neural/ctc_model_weights.ckpt', map_location='cpu')
model.load_state_dict(ckpt, strict=False)
model.eval()
if torch.cuda.is_available():
     print('Using GPU')
     model = model.to('cuda')

def recognize_file():
    audio = AudioSegment.from_wav('../media/audio.wav')
    mono_audio = audio.set_channels(1)
    mono_audio.export('../media/audio_mono.wav', format="wav")


    result = model.transcribe(['../media/audio.wav'])[0]
    print(result)

    commands = {
        "MainPage": ["Главная", "На главную", "Домой", "Перейти на главную", "Главная страница", "Вернуться на главную"],
        "Help": ["Поддержка", "Помощь", "Мне нужна поддержка", "Открой поддержку", "Открой помощь", "У меня вопрос", "Вопрос", "Перейти в поддержку"],
        "Back": ["Назад", "Предыдущая страница", "Вернуться на предыдущую страницу", "Вернуться назад"],
        "Restart": ["Перезагрузить", "Перезагрузи страницу", "Обнови страницу", "Обновить"]
    }

    def identify_command(user_input):
        texts = [phrase for sublist in commands.values() for phrase in sublist]
        vectorizer = TfidfVectorizer()
        vectorizer.fit(texts)

        user_input_vec = vectorizer.transform([user_input])
        results = {}
        for command, phrases in commands.items():
            phrase_vecs = vectorizer.transform(phrases)
            similarity = cosine_similarity(user_input_vec, phrase_vecs).flatten()
            results[command] = max(similarity)

        max_similarity = max(results.values())
        if max_similarity < 0.5:
            return "other", results

        identified_command = max(results, key=results.get)
        return identified_command, results


    identified_command, scores = identify_command(result)
    print(f"Identified command: {identified_command}")
    with open('./api/v1/neural/scores.txt', 'w', encoding='utf-8') as f:
        f.write(identified_command)
    return identified_command


recognize_file()