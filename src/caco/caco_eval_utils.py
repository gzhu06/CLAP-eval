import tensorflow as tf
import tensorflow_io as tfio
from .dataset import _get_mel_spectrogram

def load_from_list(file_path, description):
    
    data_dict = {}
    audio_name = file_path.split('/')[-1].split('.wav')[0]
    data_dict['filename'] = audio_name
    audiowav, _ = tf.audio.decode_wav(tf.io.read_file(file_path))
    audio = audiowav[:, 0]
    data_dict['spectrogram'] = compute_mel_spec_audiomae(audio)
    data_dict['text'] = tf.convert_to_tensor([description])
    data_dict['synthetic_text'] = tf.reshape(tf.convert_to_tensor(()), (0, 1))
    return data_dict

def compute_mel_spec_ast(filepath, hop_size=320, num_mel=128,
                         max_load_audio_len=16000*90):
    audio_tensor, _ = tf.audio.decode_wav(tf.io.read_file(filepath))
    mel_spec = _get_mel_spectrogram(tf.squeeze(audio_tensor, axis=1), 
                                    hop_size, num_mel, max_load_audio_len)
    return mel_spec

def compute_mel_spec_audiomae(audio_tensor, 
                              hop_length: int=160,
                              window_length: int=400,
                              num_mels: int=128,
                              scale: float=0.2,
                              bias: float=0.9):
    
    
    spec = tfio.audio.spectrogram(audio_tensor, nfft=512, window=window_length, stride=hop_length)
    mel_spec = tfio.audio.melscale(spec, rate=16000, mels=num_mels, fmin=0, fmax=16000/2)
    mel_spec = tf.math.log(mel_spec+1e-5) * scale + bias
    return mel_spec

    