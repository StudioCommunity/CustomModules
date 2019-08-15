#pip install numpy scipy librosa unidecode inflect librosa
import numpy as np
from scipy.io.wavfile import write
import torch
import builtin_models.python

class Tacotron2Model(object):
    def __init__(self):
        tacotron2 = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_tacotron2')
        tacotron2 = tacotron2.to('cuda')
        tacotron2.eval()
        print('# tacotron2 parameters:', sum(param.numel() for param in tacotron2.parameters()))

        waveglow = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_waveglow')
        waveglow = waveglow.remove_weightnorm(waveglow)
        waveglow = waveglow.to('cuda')
        waveglow.eval()
        print('# waveglow parameters:', sum(param.numel() for param in waveglow.parameters()))
        
        self.tacotron2 = tacotron2
        self.waveglow = waveglow
    
    def predict(self, sequence):
        with torch.no_grad():
            _, mel, _, _ = self.tacotron2.infer(sequence)
            audio = self.waveglow.infer(mel)
        return audio

if __name__ == '__main__':
    model = Tacotron2Model()
    model_path = "model/tacotron2"
    builtin_models.python.save_model(model, model_path)

    model1 = builtin_models.python.load_model(model_path)

    text = "We hold these truths to be self-evident, that all men are created equal, that they are endowed by their Creator with certain unalienable Rights, that among these are Life, Liberty and the pursuit of Happiness."
    # prep-rocessing
    sequence = np.array(model.tacotron2.text_to_sequence(text, ['english_cleaners']))[None, :]
    sequence = torch.from_numpy(sequence).to(device='cuda', dtype=torch.int64)
    print(sequence.shape)

    # run the models
    audios = model1.predict(sequence)

    # post-processing
    for index in range(len(audios)):
        audio = audios[index]
        audio_numpy = audio.data.cpu().numpy()
        rate = 22050
        write(f"audio_{index}.wav", rate, audio_numpy)
