import tensorflow as tf
import tensorflow_io as tfio

def load_from_list(file_path, description, ast=False):
    
    data_dict = {}
    audio_name = file_path.split('/')[-1].split('.wav')[0]
    data_dict['filename'] = audio_name
    audiowav, _ = tf.audio.decode_wav(tf.io.read_file(file_path))
    audio = audiowav[:, 0]
    if ast:
        data_dict['spectrogram'] = compute_mel_spec_audiomae(audio, hop_length=320,
                                                             window_length=800,
                                                             fft_size=1024)
    else:
        data_dict['spectrogram'] = compute_mel_spec_audiomae(audio)
    data_dict['text'] = tf.convert_to_tensor([description])
    data_dict['synthetic_text'] = tf.reshape(tf.convert_to_tensor(()), (0, 1))
    return data_dict

def compute_mel_spec_audiomae(audio_tensor, 
                              hop_length: int=160,
                              window_length: int=400,
                              fft_size: int=512,
                              num_mels: int=128,
                              scale: float=0.2,
                              bias: float=0.9):
    
    spec = tfio.audio.spectrogram(audio_tensor, nfft=fft_size, window=window_length, stride=hop_length)
    mel_spec = tfio.audio.melscale(spec, rate=16000, mels=num_mels, fmin=0, fmax=16000/2)
    mel_spec = tf.math.log(mel_spec+1e-5) * scale + bias
    return mel_spec
