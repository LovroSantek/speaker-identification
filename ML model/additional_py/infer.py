import numpy as np
import tensorflow as tf
import librosa

MODEL_PATH = "../speaker_cnn_model_quant.tflite"   # promijeni ako se zove drugačije
WAV_PATH   = "../../Voice recordings/Josip/Test/josip2.wav"

# 1) Učitaj TFLite model i izvuci parametre input/output
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()  # obavezno [web:342]

in0  = interpreter.get_input_details()[0]
out0 = interpreter.get_output_details()[0]

expected_shape = tuple(in0["shape"])  # npr. (1,187,20)
_, T, F = expected_shape              # T=187, F=20
in_scale, in_zp = in0["quantization"]
out_scale, out_zp = out0["quantization"]

print("Expected input:", expected_shape, in0["dtype"], (in_scale, in_zp))

# 2) Učitaj audio
# Ako znaš točan SR koji koristiš u treningu, postavi ga ovdje.
# U train.py je SAMPLERATE=48000. [file:393]
TARGET_SR = 48000
y, sr = librosa.load(WAV_PATH, sr=TARGET_SR, mono=True)  # resample-on-load [web:431]

# 3) Izračun MFCC (20 koeficijenata -> odgovara F=20)
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=F)        # [web:417]
feat = mfcc.T.astype(np.float32)                         # (time, mfcc) kao u train.py [file:393]

print("MFCC raw shape:", feat.shape)

# 4) Pad/trim na točno T frameova
if feat.shape[0] < T:
    pad = np.zeros((T - feat.shape[0], F), dtype=np.float32)
    feat = np.vstack([feat, pad])
else:
    feat = feat[:T, :]

feat = feat[None, :, :]  # (1,T,F)
print("MFCC final shape:", feat.shape)

# 5) Kvantizacija float -> int8 prema modelu
q = np.round(feat / in_scale + in_zp).astype(np.int32)
q = np.clip(q, -128, 127).astype(np.int8)

# 6) Inference
interpreter.set_tensor(in0["index"], q)
interpreter.invoke()  # [web:342]
yq = interpreter.get_tensor(out0["index"])

# 7) Dekvantizacija outputa u približne float scoreove
y_float = out_scale * (yq.astype(np.float32) - out_zp)   # [web:344]

print("Raw output int8:", yq)
print("Output float approx:", y_float)
print("Predicted class index:", int(np.argmax(y_float)))

#q.tofile("golden_input_petarp5_int8.bin")
#yq.tofile("golden_output_petarp5_int8.bin")

