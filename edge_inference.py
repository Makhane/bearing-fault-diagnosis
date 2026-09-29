import os
import sys
import time
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    print("错误: 未安装 onnxruntime")
    print("请运行: pip install onnxruntime")
    sys.exit(1)
MODEL_PATH = 'results/wdcnn_trained.onnx'
CLASS_NAMES = ['B007', 'B014', 'B021', 'IR007', 'IR014', 'IR021',
               'OR007', 'OR014', 'OR021', 'Normal']

def zscore_normalize(signal):

    mean = np.mean(signal)
    std = np.std(signal)
    if std < 1e-6:
        std = 1e-6
    return (signal - mean) / std

def segment_signal(signal, segment_length=1536, overlap=0.5):
    step = int(segment_length * (1 - overlap))
    segments = []
    for start in range(0, len(signal) - segment_length + 1, step):
        segments.append(signal[start:start + segment_length])
    return np.array(segments, dtype=np.float32)

def preprocess(segments):

    processed = []
    for seg in segments:
        processed.append(zscore_normalize(seg))
    x = np.array(processed, dtype=np.float32)
    return np.expand_dims(x, axis=1)  # [N, 1, 1536]

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"错误: 找不到模型 {MODEL_PATH}")
        sys.exit(1)

    print("加载 ONNX 模型...")
    session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    print(f"模型输入: {session.get_inputs()[0].shape}")
    print(f"模型输出: {session.get_outputs()[0].shape}")

    if len(sys.argv) < 2:
        print("用法: python edge_inference.py <xxx.mat>")
        sys.exit(1)

    mat_path = sys.argv[1]

    try:
        import scipy.io as sio
        mat_data = sio.loadmat(mat_path)
        signal = None
        for key in ['DE_time', 'de_time', 'FE_time', 'BA_time', 'data', 'X']:
            if key in mat_data and isinstance(mat_data[key], np.ndarray) and mat_data[key].size > 1000:
                signal = mat_data[key].flatten()
                break
        if signal is None:
            for key in mat_data:
                if key.startswith('__'):
                    continue
                val = mat_data[key]
                if isinstance(val, np.ndarray) and val.size > 1000:
                    signal = val.flatten()
                    break
    except Exception as e:
        print(f"读取失败: {e}")
        sys.exit(1)

    if signal is None:
        print("无法读取振动信号")
        sys.exit(1)

    print(f"信号长度: {len(signal)}")

    # 分段（1536点）
    segments = segment_signal(signal)
    print(f"分段数量: {len(segments)} 段 (1536点/段)")

    # 预处理 [N, 1, 1536]
    x = preprocess(segments)

    # 推理计时
    t0 = time.time()
    logits = session.run(None, {input_name: x})[0]
    t1 = time.time()

    # Softmax + 聚合
    probs = softmax(logits)
    avg_probs = np.mean(probs, axis=0)
    pred_label = int(np.argmax(avg_probs))
    pred_class = CLASS_NAMES[pred_label]
    confidence = float(avg_probs[pred_label])

    # 输出结果
    print("\n" + "=" * 50)
    print("轴承故障诊断结果")
    print("=" * 50)
    print(f"预测类别: {pred_class}")
    print(f"置信度:   {confidence*100:.2f}%")
    print(f"推理耗时: {(t1-t0)*1000:.1f} ms")
    print(f"设备状态: {'⚠️ 故障' if pred_class != 'Normal' and confidence > 0.8 else '✅ 正常'}")
    print("-" * 50)
    print("各类别概率:")
    for i, cls in enumerate(CLASS_NAMES):
        print(f"  {cls:8s}: {avg_probs[i]*100:6.2f}%")
    print("=" * 50)

    # 保存结果
    result_file = mat_path.replace('.mat', '_result.txt')
    with open(result_file, 'w') as f:
        f.write(f"文件: {mat_path}\n")
        f.write(f"预测: {pred_class}\n")
        f.write(f"置信度: {confidence*100:.2f}%\n")
        f.write(f"耗时: {(t1-t0)*1000:.1f}ms\n")
    print(f"结果已保存: {result_file}")

if __name__ == '__main__':
    main()