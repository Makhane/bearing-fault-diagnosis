import serial
import numpy as np
import sys

NANO_PORT  = "COM5"
NANO_BAUD  = 115200
SAMPLES    = 1536
BYTES_NEED = SAMPLES * 2
SD_PATH    = "G:\\z_input.bin"

def collect():
    print(f"[1/2] 打开 {NANO_PORT} @ {NANO_BAUD}，采集 {SAMPLES} 点...")
    ser = serial.Serial(NANO_PORT, NANO_BAUD, timeout=10)
    ser.flushInput()
    buf = ser.read(BYTES_NEED)
    ser.close()
    if len(buf) != BYTES_NEED:
        print(f"错误：收到 {len(buf)} 字节，需 {BYTES_NEED}")
        sys.exit(1)
    raw = np.frombuffer(buf, dtype=np.int16).astype(np.float32)
    print(f"    原始范围: [{raw.min()}, {raw.max()}]")
    return raw

def preprocess(raw):
    print("[2/2] 转 float → SD 卡（z-score 由板端处理）...")
    return raw.astype(np.float32)

if __name__ == "__main__":
    raw = collect()
    proc = preprocess(raw)
    proc.tofile(SD_PATH)
    print(f"    已写入: {SD_PATH} ({proc.nbytes} bytes)")
    print("    请安全弹出 SD 卡，插入华山派")