import pathlib
import textwrap

TFLITE_PATH = pathlib.Path("../speaker_cnn_model_quant.tflite")
OUT_H = pathlib.Path("model_data.h")
OUT_CC = pathlib.Path("model_data.cc")

data = TFLITE_PATH.read_bytes()

def format_bytes_as_hex(bytez, cols=12):
    hexes = [f"0x{b:02x}" for b in bytez]
    lines = [", ".join(hexes[i:i+cols]) for i in range(0, len(hexes), cols)]
    return ",\n  ".join(lines)

h = """\
#pragma once
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

extern const unsigned char g_model[];
extern const unsigned int g_model_len;

#ifdef __cplusplus
}
#endif
"""

cc = f"""\
#include "model_data.h"

alignas(16) const unsigned char g_model[] = {{
  {format_bytes_as_hex(data)}
}};

const unsigned int g_model_len = {len(data)};
"""

OUT_H.write_text(h, encoding="utf-8")
OUT_CC.write_text(cc, encoding="utf-8")

print(f"Wrote {OUT_H} and {OUT_CC} ({len(data)} bytes).")
