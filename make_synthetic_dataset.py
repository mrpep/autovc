import pickle
import numpy as np
import soundfile as sf
from scipy import signal
from scipy.signal import get_window
from librosa.filters import mel
from numpy.random import RandomState

from model_bl import D_VECTOR
from model_vc import Generator

from collections import OrderedDict
import torch
import librosa

from synthesis import build_model
from synthesis import wavegen

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def pySTFT(x, fft_length=1024, hop_length=256):
    
    x = np.pad(x, int(fft_length//2), mode='reflect')
    
    noverlap = fft_length - hop_length
    shape = x.shape[:-1]+((x.shape[-1]-noverlap)//hop_length, fft_length)
    strides = x.strides[:-1]+(hop_length*x.strides[-1], x.strides[-1])
    result = np.lib.stride_tricks.as_strided(x, shape=shape,
                                             strides=strides)
    
    fft_window = get_window('hann', fft_length, fftbins=True)
    result = np.fft.rfft(fft_window * result, n=fft_length).T
    
    return np.abs(result)

def pad_seq(x, base=32):
    len_out = int(base * np.ceil(float(x.shape[0])/base))
    len_pad = len_out - x.shape[0]
    assert len_pad >= 0
    return np.pad(x, ((0,len_pad),(0,0)), 'constant'), len_pad    
    
def generate_melspectrogram(wav_file):
  mel_basis = mel(16000, 1024, fmin=90, fmax=7600, n_mels=80).T
  min_level = np.exp(-100 / 20 * np.log(10))
  b, a = butter_highpass(30, 16000, order=5)

  #x, fs = sf.read(wav_file)
  x,fs = librosa.core.load(wav_file,sr=16000)
  y = signal.filtfilt(b, a, x)
  prng = RandomState(1234)
  wav = y * 0.96 + (prng.rand(y.shape[0])-0.5)*1e-06
  D = pySTFT(wav).T
  D_mel = np.dot(D, mel_basis)
  D_db = 20 * np.log10(np.maximum(min_level, D_mel)) - 16
  S = np.clip((D_db + 100) / 100, 0, 1)

  return S

def load_sencoder():
  C = D_VECTOR(dim_input=80, dim_cell=768, dim_emb=256).eval().cuda()
  c_checkpoint = torch.load('3000000-BL.ckpt')
  new_state_dict = OrderedDict()
  for key, val in c_checkpoint['model_b'].items():
      new_key = key[7:]
      new_state_dict[new_key] = val
  C.load_state_dict(new_state_dict)

  return C

def load_autovc():
  device = 'cuda:0'
  G = Generator(32,256,512,32).eval().to(device)

  g_checkpoint = torch.load('autovc.ckpt',map_location='cuda:0')
  G.load_state_dict(g_checkpoint['model'])

  return G

def generate_speaker_embedding(mel_spectrograms,s_encoder):
  len_crop = 128
  hop_frames = 32

  mel_spectrograms = [m if len(m)>=len_crop else np.pad(m,((0,len_crop - len(m) + 1),(0,0))) for m in mel_spectrograms]

  batches = [m[i:i+len_crop] for m in mel_spectrograms for i in range(0,len(m) - len_crop,hop_frames)]

  melsp = torch.from_numpy(np.array(batches,dtype=np.float32)).cuda()
  emb = s_encoder(melsp)
  
  return emb.detach().squeeze().cpu().numpy()

def convert(source_mel, source_speaker_emb, target_speaker_emb, autovc_model):
  device = 'cuda:0'

  source_mel, len_pad = pad_seq(source_mel)
  source_mel = torch.from_numpy(source_mel[np.newaxis, :, :]).to(device)
  source_speaker_emb = torch.from_numpy(source_speaker_emb[np.newaxis, :]).to(device)
  target_speaker_emb = torch.from_numpy(target_speaker_emb[np.newaxis, :]).to(device)

  with torch.no_grad():
    _, x_identic_psnt, _ = autovc_model(source_mel, source_speaker_emb, target_speaker_emb)

  if len_pad == 0:
    uttr_trg = x_identic_psnt[0, 0, :, :].cpu().numpy()
  else:
    uttr_trg = x_identic_psnt[0, 0, :-len_pad, :].cpu().numpy()

  return uttr_trg

def load_vocoder():
  device = torch.device("cuda")
  model = build_model().to(device)
  checkpoint = torch.load("checkpoint_step001000000_ema.pth")
  model.load_state_dict(checkpoint["state_dict"])

  return model

def vocode(mel, vocoder):
  waveform = wavegen(vocoder, c=mel)
  return waveform

def voice_convert(source_wav, target_wavs, s_encoder_model = None,autovc_model=None, vocoder=None, vocoder_type='slow'):
  S_source = generate_melspectrogram(source_wav).astype(np.float32)
  S_target = [generate_melspectrogram(f) for f in target_wavs]
  sp_source_embedding = generate_speaker_embedding([S_source],s_encoder_model)
  if sp_source_embedding.ndim==1:
    sp_source_embedding = np.expand_dims(sp_source_embedding,axis=0)
  sp_source_mean_embedding = np.mean(sp_source_embedding,axis=0)

  sp_target_embedding = generate_speaker_embedding(S_target,s_encoder_model)
  if sp_target_embedding.ndim==1:
    sp_target_embedding = np.expand_dims(sp_target_embedding,axis=0)
  sp_target_mean_embedding = np.mean(sp_target_embedding,axis=0)

  S_result = convert(S_source, sp_source_mean_embedding, sp_target_mean_embedding,autovc_model)

  if vocoder_type == 'slow':
    waveform = vocode(S_result,vocoder)

  return S_source, S_result,waveform

def generate_synthetic_dataset(source_folder, targets_folder, output_path, mode='slow'):
  import glob
  from pathlib import Path
  import tqdm
  import joblib

  C = load_sencoder()
  G = load_autovc()

  if mode == 'slow':
    V = load_vocoder()

  sources = glob.glob(source_folder + '/*.wav')
  target_subjects = glob.glob(targets_folder + '/*')
  targets = {Path(t).stem: list(glob.glob(t+'/*.wav')) for t in target_subjects}

  for source in tqdm.tqdm(sources):
    source_name = Path(source).stem
    for k,v in tqdm.tqdm(targets.items()):
      S_source, S_result, waveform = voice_convert(source,v,C,G,V,mode)
      sf.write(str(Path(output_path,'{}-{}.wav'.format(source_name,k))),waveform.T,22050)
      joblib.dump(S_result,str(Path(output_path,'{}-{}.mel'.format(source_name,k))))