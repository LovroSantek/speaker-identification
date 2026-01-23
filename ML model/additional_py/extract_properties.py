import numpy as np
import tensorflow as tf

model_path = "../speaker_cnn_model_quant.tflite"

interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

ins = interpreter.get_input_details()
outs = interpreter.get_output_details()

print("INPUTS:")
for t in ins:
    print(" name:", t["name"])
    print(" index:", t["index"])
    print(" dtype:", t["dtype"])
    print(" shape:", t["shape"])
    print(" shape_signature:", t.get("shape_signature"))
    print(" quantization:", t.get("quantization"))  # (scale, zero_point)
    print(" quantization_parameters:", t.get("quantization_parameters"))
    print()

print("OUTPUTS:")
for t in outs:
    print(" name:", t["name"])
    print(" index:", t["index"])
    print(" dtype:", t["dtype"])
    print(" shape:", t["shape"])
    print(" quantization:", t.get("quantization"))
    print()
