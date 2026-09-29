
Python 3.10/3.11
conda 虚拟环境（推荐）
WSL（含 riscv64-linux-gnu-gcc 交叉编译器）
 二、安装
conda create -n wdcnn python=3.11
conda activate wdcnn
pip install -r requirements.txt
 三、数据预处理
python run_preprocessing.py
读取 CWRU 原始 .mat 数据
分段为 1536 点样本，逐样本 z-score 标准化
输出 dataset.npz（训练集/测试集已划分）
 四、模型训练
python scripts/train_both.py

单独训练
python scripts/quick_train_wdcnn.py   # WDCNN 25轮
python scripts/quick_train_lstm.py    # CNN+LSTM 25轮
 五、导出模型
ONNX（PC端）:  python scripts/export_onnx_now.py
C权重（板端）: python scripts/export_weights_to_c.py
生成 wdcnn_weights.h，编译时链入 wdcnn_infer.c

六、生成图表
python scripts/generate_all_figures.py
输出：训练曲线、混淆矩阵、模型对比柱状图。
 七、启动Web端（可选）
python web_app/app.py
浏览器访问: http://localhost:5000

八、板端部署
Arduino 烧录
用 Arduino IDE 打开 arduino/adxl345_z_binary.ino
开发板选 "Arduino Nano"，处理器选 "ATmega328P (Old Bootloader)"，端口选 COM5
关键参数：115200 波特率，I2C 400kHz，ADXL345 输出率 1600Hz，只读 Z 轴二进制 int16 输出，loop 无 delay
PC 采集与预处理
SD 卡插入 PC，确认盘符（如 G:）
执行：
python pc_to_sd.py
从 COM5 采集 1536 点 int16（3072 字节），转 float32 写入 SD 卡 z_input.bin

SD 卡传输到华山派
安全弹出 SD 卡，插入华山派卡槽（设备节点 /dev/mmcblk1p1）
华山派推理
一键脚本：
/mnt/data/detect.sh
手动执行：
mkdir -p /mnt/sd
mount /dev/mmcblk1p1 /mnt/sd
cp /mnt/sd/z_input.bin /mnt/data/
/mnt/data/wdcnn_infer /mnt/data/z_input.bin /mnt/data/result.txt
cat /mnt/data/result.txt
umount /mnt/sd
板端推理引擎编译（WSL）
cd results
riscv64-linux-gnu-gcc -static wdcnn_infer.c -o wdcnn_infer -O2 -lm
验证：
file wdcnn_infer
预期输出：ELF 64-bit LSB executable, UCB RISC-V, statically linked
部署到华山派：
cp wdcnn_infer /mnt/g/          #复制到 SD 卡/U 盘
在华山派上：cp /mnt/sd/wdcnn_infer /mnt/data/wdcnn_infer && chmod +x /mnt/data/wdcnn_infer
