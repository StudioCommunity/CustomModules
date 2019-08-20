#pip install numpy scipy librosa unidecode inflect librosa
import numpy as np
from scipy.io.wavfile import write
import torch
import builtin_models.python

class Tacotron2Model(object):
    def __init__(self, model_path):
        self.device = 'cuda' # cuda only
        
        print('# device:', self.device)
        tacotron2 = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_tacotron2')
        tacotron2 = tacotron2.to(self.device)
        tacotron2.eval()
        print('# tacotron2 parameters:', sum(param.numel() for param in tacotron2.parameters()))

        waveglow = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_waveglow')
        waveglow = waveglow.remove_weightnorm(waveglow)
        waveglow = waveglow.to(self.device)
        waveglow.eval()
        print('# waveglow parameters:', sum(param.numel() for param in waveglow.parameters()))
        
        self.tacotron2 = tacotron2
        self.waveglow = waveglow
    
    def predict(self, text):
        # prep-rocessing
        sequence = np.array(model.tacotron2.text_to_sequence(text, ['english_cleaners']))[None, :]
        
        # run the models
        sequence = torch.from_numpy(sequence).to(device=self.device, dtype=torch.int64)
        with torch.no_grad():
            _, mel, _, _ = self.tacotron2.infer(sequence)
            audio = self.waveglow.infer(mel)
        return audio

# python -m dstest.nlp.text-to-speech.tacotron2
if __name__ == '__main__':
    model = Tacotron2Model()
    model_path = "model/tacotron2"
    builtin_models.python.save_model(model, model_path)

    model1 = builtin_models.python.load_model(model_path)

    text = "We hold these truths to be self-evident, that all men are created equal, that they are endowed by their Creator with certain unalienable Rights, that among these are Life, Liberty and the pursuit of Happiness."
    # run the models
    audios = model1.predict(text)
    #print(audios.shape)

    # post-processing
    for index in range(len(audios)):
        audio = audios[index]
        print(audio.shape)
        audio_numpy = audio.data.cpu().numpy()
        rate = 22050
        write(f"audio_{index}.wav", rate, audio_numpy)
