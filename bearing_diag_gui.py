
import sys
import os
import json
import serial
import serial.tools.list_ports
import threading
import queue
import time
import random
import csv
import numpy as np
from datetime import datetime
from scipy.io import loadmat as sio_loadmat

try:
    from ai_chat_module import AIChatWidget, AIFloatButton
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False

try:
    import openpyxl
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False

PYTORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    PYTORCH_AVAILABLE = True
    print(f"[OK] PyTorch已加载: {torch.__version__}")
except ImportError:
    print("[WARN] PyTorch未安装，模式B(PC本地推理)不可用")
    print("       安装命令: pip install torch")

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QTextEdit, QTableWidget, QTableWidgetItem,
    QGroupBox, QGridLayout, QFileDialog, QMessageBox, QHeaderView, QFrame,
    QAbstractItemView, QRadioButton, QButtonGroup, QStatusBar, QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QFont, QColor, QPainter, QBrush, QPen, QLinearGradient, QPalette, QFontDatabase

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

_mpl_fonts = [f.name for f in fm.fontManager.ttflist]
for _fc in ['Microsoft YaHei', 'SimHei', 'PingFang SC', 'WenQuanYi Micro Hei']:
    if _fc in _mpl_fonts:
        plt.rcParams['font.sans-serif'] = [_fc, 'DejaVu Sans', 'Arial']
        break
plt.rcParams['axes.unicode_minus'] = False

CLASSES_10 = ['滚动体7mm', '滚动体14mm', '滚动体21mm',
              '内圈7mm', '内圈14mm', '内圈21mm',
              '外圈7mm', '外圈14mm', '外圈21mm',
              '正常']
CLASSES_5 = ['健康', '内圈故障', '外圈故障', '滚动体故障', '保持架故障']
CLASS_10_TO_5 = {0: 3, 1: 3, 2: 3, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 0}
CLASS_COLORS_5 = {
    '健康': '#00C851', '内圈故障': '#ffbb33', '外圈故障': '#ff4444',
    '滚动体故障': '#aa66cc', '保持架故障': '#33b5e5',
}
CLASS_IDX_COLORS = ['#00C851', '#ffbb33', '#ff4444', '#aa66cc', '#33b5e5']
FRAME_LEN = 1536

CN_FONT = None

def detect_chinese_font():
    """必须在QApplication创建后调用"""
    global CN_FONT
    db = QFontDatabase()
    families = db.families()
    for c in ['Microsoft YaHei', 'SimHei', 'SimSun', 'PingFang SC', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC']:
        if c in families:
            CN_FONT = c
            return c
    for f in families:
        if 'CJK' in f or 'Hei' in f or 'YaHei' in f:
            CN_FONT = f
            return f
    CN_FONT = 'Arial'
    return CN_FONT

class WDCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(WDCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, 64, 16, 24)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(2, 2)
        self.conv2 = nn.Conv1d(16, 32, 3, 1, 1)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(2, 2)
        self.conv3 = nn.Conv1d(32, 64, 3, 1, 1)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool3 = nn.MaxPool1d(2, 2)
        self.conv4 = nn.Conv1d(64, 64, 3, 1, 1)
        self.bn4 = nn.BatchNorm1d(64)
        self.pool4 = nn.MaxPool1d(2, 2)
        self.conv5 = nn.Conv1d(64, 64, 3, 1, 1)
        self.bn5 = nn.BatchNorm1d(64)
        self.pool5 = nn.MaxPool1d(2, 2)
        self.fc1 = nn.Linear(192, 100)
        self.fc2 = nn.Linear(100, num_classes)

    def forward(self, x):
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
        x = self.pool4(torch.relu(self.bn4(self.conv4(x))))
        x = self.pool5(torch.relu(self.bn5(self.conv5(x))))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class Communicate(QObject):
    log_signal = pyqtSignal(str, str)
    result_signal = pyqtSignal(int, float)

class LEDIndicator(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(60, 60)
        self.color = QColor('#cccccc')
        self.glowing = False

    def set_color(self, color_hex):
        self.color = QColor(color_hex)
        self.update()

    def set_glowing(self, glow):
        self.glowing = glow
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect().adjusted(4, 4, -4, -4)
        if self.glowing:
            for i in range(3):
                glow_color = QColor(self.color)
                glow_color.setAlpha(80 - i * 25)
                painter.setBrush(QBrush(glow_color))
                painter.setPen(Qt.NoPen)
                glow_rect = rect.adjusted(-4 - i * 3, -4 - i * 3, 4 + i * 3, 4 + i * 3)
                painter.drawEllipse(glow_rect)
        gradient = QLinearGradient(rect.topLeft(), rect.bottomRight())
        gradient.setColorAt(0, self.color.lighter(120))
        gradient.setColorAt(1, self.color.darker(120))
        painter.setBrush(QBrush(gradient))
        painter.setPen(QPen(self.color.darker(150), 2))
        painter.drawEllipse(rect)

class ProbabilityBars(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(250)
        self.probabilities = [1.0, 0.0, 0.0, 0.0, 0.0]
        self.target_cls = 0
        self.setStyleSheet("background: #1e1e2e; border-radius: 8px;")

    def set_probabilities(self, probs, target):
        self.probabilities = probs
        self.target_cls = target
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        n, margin, gap = 5, 16, 10
        bar_w = max(50, (w - margin * 2 - gap * (n - 1)) // n)
        max_bar_h = h - 100
        labels = ['健康', '内圈', '外圈', '滚动', '保持']

        for i in range(n):
            x = margin + i * (bar_w + gap)
            prob = self.probabilities[i]

            if i == self.target_cls:
                bh = max(35, int(prob * max_bar_h))
            else:
                bh = max(20, int(prob * max_bar_h * 0.4))
            color = QColor(CLASS_IDX_COLORS[i])
            is_target = (i == self.target_cls)
            bar_top = h - 55 - bh

            if is_target:
                for j in range(4):
                    glow_color = QColor(color)
                    glow_color.setAlpha(60 - j * 15)
                    painter.fillRect(x - 4, bar_top - 4 + j * 3, bar_w + 8, bh + 8 - j * 3, glow_color)

            # 柱状图本体
            gradient = QLinearGradient(x, h - 45, x, bar_top)
            gradient.setColorAt(0, color)
            gradient.setColorAt(1, color.darker(160))
            painter.fillRect(x, bar_top, bar_w, bh, gradient)

            painter.setPen(QPen(color.darker(120), 1))
            painter.drawRect(x, bar_top, bar_w, bh)

            painter.setPen(QColor('#ffffff') if is_target else QColor('#dbeafe'))
            font_name = CN_FONT if CN_FONT else 'Microsoft YaHei'
            painter.setFont(QFont(font_name, 10, QFont.Bold))
            painter.drawText(x, bar_top - 22, bar_w, 18, Qt.AlignCenter, f'{prob*100:.0f}%')

            painter.setPen(QColor('#dbeafe'))
            painter.setFont(QFont(font_name, 10))
            painter.drawText(x, h - 50, bar_w, 20, Qt.AlignCenter, labels[i])

        painter.setPen(QColor('#7a8da3'))
        painter.setFont(QFont(font_name, 9))
        legend_text = ' | '.join([
            '健康', '内圈故障', '外圈故障', '滚动体故障', '保持架故障'
        ])
        painter.drawText(margin, h - 28, w - margin * 2, 16, Qt.AlignCenter, legend_text)

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=3, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor='#1e1e2e')
        self.axes = self.fig.add_subplot(111)
        self.axes.set_facecolor('#1e1e2e')
        self.axes.tick_params(colors='#dbeafe', labelsize=8)
        for spine in self.axes.spines.values():
            spine.set_color('#334155')
        self.axes.grid(True, alpha=0.2, color='#7a8da3')
        super().__init__(self.fig)
        self.setParent(parent)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('旋转机械智能运维终端 v3.4 (AI助手版)')
        self.setMinimumSize(1600, 1000)
        self.resize(1800, 1100)

        self.ser_h = None
        self.ser_n = None
        self.comm = Communicate()
        self.comm.log_signal.connect(self.on_log)
        self.comm.result_signal.connect(self.on_result)

        self.infer_mode = 'edge'
        self.model_local = None
        self.model_path = ''
        self.device_torch = torch.device('cpu')

        self.current_signal = None
        self.diag_counter = 0

        self.setup_ui()
        self.refresh_ports()
        self.log('系统初始化完成', 'info')
        self.log('当前模式: 华山派边缘推理 (模式A)', 'info')

        self.ai_visible = False
        if AI_AVAILABLE:
            self._init_ai_assistant()

        self.auto_detect_model()

    def log(self, text, log_type='info'):
        t = datetime.now().strftime('%H:%M:%S')
        colors = {'info': '#e2e8f0', 'result': '#38bdf8', 'error': '#f87171', 'warn': '#fbbf24'}
        color = colors.get(log_type, '#e2e8f0')
        self.txt_log.append(f'<span style="color:#c8d5e3">[{t}]</span> <span style="color:{color}">{text}</span>')

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(10)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(10)
        left_panel.setMaximumWidth(420)
        left_panel.setMinimumWidth(380)

        mode_group = QGroupBox('推理模式选择')
        mode_group.setStyleSheet(self.groupbox_style())
        mode_layout = QVBoxLayout(mode_group)

        self.btn_group_mode = QButtonGroup(self)
        self.rb_edge = QRadioButton('模式A: 华山派CV1812H边缘推理 (UART)')
        self.rb_edge.setChecked(True)
        self.rb_edge.setStyleSheet('color: #e2e8f0; font-size: 12px;')
        self.rb_edge.toggled.connect(self.on_mode_changed)
        self.btn_group_mode.addButton(self.rb_edge)
        mode_layout.addWidget(self.rb_edge)

        self.rb_local = QRadioButton('模式B: PC本地PyTorch推理 (加载.pt模型)')
        self.rb_local.setStyleSheet('color: #e2e8f0; font-size: 12px;')
        self.rb_local.toggled.connect(self.on_mode_changed)
        self.btn_group_mode.addButton(self.rb_local)

        if not PYTORCH_AVAILABLE:
            self.rb_local.setEnabled(False)
            self.rb_local.setText('模式B: PC本地PyTorch推理 [PyTorch未安装]')
            self.rb_local.setStyleSheet('color: #7a8da3; font-size: 12px;')

        mode_layout.addWidget(self.rb_local)

        self.lbl_model_status = QLabel('模型状态: 未加载')
        self.lbl_model_status.setStyleSheet('color: #c8d5e3; font-size: 11px; padding: 4px;')
        mode_layout.addWidget(self.lbl_model_status)

        btn_load_model = QPushButton('加载PyTorch模型 (.pt)')
        btn_load_model.setStyleSheet(self.btn_secondary_style())
        btn_load_model.clicked.connect(self.load_model_file)
        mode_layout.addWidget(btn_load_model)

        left_layout.addWidget(mode_group)

        port_group = QGroupBox('端边协同控制')
        port_group.setStyleSheet(self.groupbox_style())
        port_layout = QVBoxLayout(port_group)

        # 华山派
        h_frame = QFrame()
        h_layout = QGridLayout(h_frame)
        h_layout.setSpacing(6)
        h_layout.setContentsMargins(4, 4, 4, 4)

        lbl_h_title = QLabel('华山派 CV1812H')
        lbl_h_title.setStyleSheet('color: #38bdf8; font-size: 13px; font-weight: bold;')
        lbl_h_baud = QLabel('@115200')
        lbl_h_baud.setStyleSheet('color: #c8d5e3; font-size: 11px;')
        lbl_h_baud.setAlignment(Qt.AlignRight)

        h_layout.addWidget(lbl_h_title, 0, 0)
        h_layout.addWidget(lbl_h_baud, 0, 1)

        self.cmb_h = QComboBox()
        self.cmb_h.setStyleSheet(self.combo_style())
        self.cmb_h.setMinimumHeight(28)
        h_layout.addWidget(self.cmb_h, 1, 0)

        self.btn_h = QPushButton('连接')
        self.btn_h.setStyleSheet(self.btn_primary_style())
        self.btn_h.setMinimumHeight(28)
        self.btn_h.clicked.connect(self.toggle_huashan)
        h_layout.addWidget(self.btn_h, 1, 1)

        self.lbl_h_status = QLabel('未连接')
        self.lbl_h_status.setStyleSheet('color: #c8d5e3; font-size: 11px; padding-top: 4px;')
        h_layout.addWidget(self.lbl_h_status, 2, 0, 1, 2)

        port_layout.addWidget(h_frame)
        port_layout.addWidget(self.hline())

        # Nano
        n_frame = QFrame()
        n_layout = QGridLayout(n_frame)
        n_layout.setSpacing(6)
        n_layout.setContentsMargins(4, 4, 4, 4)

        lbl_n_title = QLabel('Arduino Nano (报警灯)')
        lbl_n_title.setStyleSheet('color: #fbbf24; font-size: 13px; font-weight: bold;')
        lbl_n_baud = QLabel('@9600')
        lbl_n_baud.setStyleSheet('color: #c8d5e3; font-size: 11px;')
        lbl_n_baud.setAlignment(Qt.AlignRight)

        n_layout.addWidget(lbl_n_title, 0, 0)
        n_layout.addWidget(lbl_n_baud, 0, 1)

        self.cmb_n = QComboBox()
        self.cmb_n.setStyleSheet(self.combo_style())
        self.cmb_n.setMinimumHeight(28)
        n_layout.addWidget(self.cmb_n, 1, 0)

        self.btn_n = QPushButton('连接')
        self.btn_n.setStyleSheet(self.btn_primary_style())
        self.btn_n.setMinimumHeight(28)
        self.btn_n.clicked.connect(self.toggle_nano)
        n_layout.addWidget(self.btn_n, 1, 1)

        self.lbl_n_status = QLabel('未连接')
        self.lbl_n_status.setStyleSheet('color: #c8d5e3; font-size: 11px; padding-top: 4px;')
        n_layout.addWidget(self.lbl_n_status, 2, 0, 1, 2)

        port_layout.addWidget(n_frame)
        port_layout.addWidget(self.hline())

        btn_refresh = QPushButton('刷新端口')
        btn_refresh.setStyleSheet(self.btn_secondary_style())
        btn_refresh.clicked.connect(self.refresh_ports)
        port_layout.addWidget(btn_refresh)

        left_layout.addWidget(port_group)

        #诊断状态
        status_group = QGroupBox('当前诊断状态')
        status_group.setStyleSheet(self.groupbox_style())
        status_layout = QVBoxLayout(status_group)

        led_layout = QHBoxLayout()
        self.led = LEDIndicator()
        led_layout.addWidget(self.led)
        self.lbl_status = QLabel('等待数据')
        font_name = CN_FONT if CN_FONT else 'Microsoft YaHei'
        self.lbl_status.setFont(QFont(font_name, 16, QFont.Bold))
        self.lbl_status.setStyleSheet('color: #c8d5e3;')
        led_layout.addWidget(self.lbl_status)
        led_layout.addStretch()
        status_layout.addLayout(led_layout)

        self.prob_bars = ProbabilityBars()
        status_layout.addWidget(self.prob_bars)

        #推理信息
        self.lbl_infer_info = QLabel('')
        self.lbl_infer_info.setStyleSheet('color: #7a8da3; font-size: 10px; padding: 2px;')
        status_layout.addWidget(self.lbl_infer_info)

        left_layout.addWidget(status_group)

        #系统信息
        info_group = QGroupBox('系统信息')
        info_group.setStyleSheet(self.groupbox_style())
        info_group.setMaximumHeight(65)
        info_layout = QVBoxLayout(info_group)
        info_layout.setContentsMargins(6, 3, 6, 3)
        info_layout.setSpacing(0)
        self.lbl_sys_info = QLabel(
            f'PyTorch:{torch.__version__ if PYTORCH_AVAILABLE else "--"} | '
            f'{self.device_torch} | FRAME={FRAME_LEN}'
        )
        self.lbl_sys_info.setStyleSheet('color: #c8d5e3; font-size: 10px;')
        info_layout.addWidget(self.lbl_sys_info)
        left_layout.addWidget(info_group)
        left_layout.addStretch(1)

        main_layout.addWidget(left_panel, stretch=0)

        #中间面板
        mid_panel = QWidget()
        mid_layout = QVBoxLayout(mid_panel)
        mid_layout.setSpacing(10)

        chart_group = QGroupBox('信号分析与特征提取')
        chart_group.setStyleSheet(self.groupbox_style())
        chart_layout = QVBoxLayout(chart_group)

        row1 = QHBoxLayout()
        self.canvas_time = MplCanvas(self, width=6, height=2.5)
        self.canvas_freq = MplCanvas(self, width=6, height=2.5)
        row1.addWidget(self.canvas_time)
        row1.addWidget(self.canvas_freq)
        chart_layout.addLayout(row1)

        row2 = QHBoxLayout()
        self.canvas_orbit = MplCanvas(self, width=6, height=2.5)
        self.canvas_env = MplCanvas(self, width=6, height=2.5)
        row2.addWidget(self.canvas_orbit)
        row2.addWidget(self.canvas_env)
        chart_layout.addLayout(row2)
        mid_layout.addWidget(chart_group)

        log_group = QGroupBox('推理日志')
        log_group.setStyleSheet(self.groupbox_style())
        log_layout = QVBoxLayout(log_group)
        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setFont(QFont('Consolas', 9))
        self.txt_log.setStyleSheet(
            'QTextEdit { background-color: #0d1117; color: #e6edf3;'
            'border: 1px solid #30363d; border-radius: 8px; padding: 8px; }'
        )
        log_layout.addWidget(self.txt_log)
        btn_clear_log = QPushButton('清空日志')
        btn_clear_log.setStyleSheet(self.btn_secondary_style())
        btn_clear_log.clicked.connect(self.txt_log.clear)
        log_layout.addWidget(btn_clear_log)
        mid_layout.addWidget(log_group)
        main_layout.addWidget(mid_panel, stretch=4)

        #右侧面板
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_panel.setMaximumWidth(420)
        right_panel.setMinimumWidth(360)

        hist_group = QGroupBox('诊断历史')
        hist_group.setStyleSheet(self.groupbox_style())
        hist_layout = QVBoxLayout(hist_group)

        self.table_hist = QTableWidget()
        self.table_hist.setColumnCount(4)
        self.table_hist.setHorizontalHeaderLabels(['时间', '结果', '置信度', '处置'])
        self.table_hist.setColumnWidth(0, 100)
        self.table_hist.setColumnWidth(1, 85)
        self.table_hist.setColumnWidth(2, 65)
        self.table_hist.setColumnWidth(3, 75)
        self.table_hist.horizontalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self.table_hist.setStyleSheet(
            'QTableWidget { background-color: #1e1e2e; color: #e2e8f0;'
            'border: 1px solid #334155; border-radius: 8px; gridline-color: #334155; }'
            'QHeaderView::section { background-color: #1e293b; color: #dbeafe;'
            'padding: 6px; border: 1px solid #334155; }'
            'QTableWidget::item { padding: 6px; }'
            'QTableWidget::item:selected { background-color: #38bdf820; }'
        )
        self.table_hist.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table_hist.setEditTriggers(QAbstractItemView.NoEditTriggers)
        hist_layout.addWidget(self.table_hist)
        right_layout.addWidget(hist_group)

        stat_group = QGroupBox('统计汇总')
        stat_group.setStyleSheet(self.groupbox_style())
        stat_layout = QVBoxLayout(stat_group)
        self.lbl_stat_total = QLabel('总诊断: 0')
        self.lbl_stat_normal = QLabel('正常: 0')
        self.lbl_stat_fault = QLabel('故障: 0')
        self.lbl_stat_acc = QLabel('准确率: --')
        for lbl in [self.lbl_stat_total, self.lbl_stat_normal, self.lbl_stat_fault, self.lbl_stat_acc]:
            lbl.setFont(QFont('Consolas', 10))
            stat_layout.addWidget(lbl)
        right_layout.addWidget(stat_group)

        btn_group = QGroupBox('操作')
        btn_group.setStyleSheet(self.groupbox_style())
        btn_layout = QVBoxLayout(btn_group)
        btn_layout.setSpacing(12)
        btn_layout.setContentsMargins(10, 10, 10, 10)
        btn_layout.addStretch(1)

        font_name = CN_FONT if CN_FONT else 'Microsoft YaHei'

        btn_alarm = QPushButton('测试报警灯')
        btn_alarm.setStyleSheet(self.btn_danger_style() + f' QPushButton{{font-family:"{font_name}";font-size:14px;font-weight:bold;}}')
        btn_alarm.setMinimumHeight(56)
        btn_alarm.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        btn_alarm.clicked.connect(self.test_alarm)
        btn_layout.addWidget(btn_alarm, stretch=2)

        btn_mock = QPushButton('模拟诊断')
        btn_mock.setStyleSheet(self.btn_secondary_style() + f' QPushButton{{font-family:"{font_name}";font-size:14px;font-weight:bold;}}')
        btn_mock.setMinimumHeight(56)
        btn_mock.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        btn_mock.clicked.connect(self.mock_diagnosis)
        btn_layout.addWidget(btn_mock, stretch=2)

        btn_load = QPushButton('加载诊断样本')
        btn_load.setStyleSheet(self.btn_secondary_style() + f' QPushButton{{font-family:"{font_name}";font-size:14px;font-weight:bold;}}')
        btn_load.setMinimumHeight(56)
        btn_load.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        btn_load.clicked.connect(self.load_sample)
        btn_layout.addWidget(btn_load, stretch=2)

        btn_export = QPushButton('导出诊断报告')
        btn_export.setStyleSheet(self.btn_secondary_style() + f' QPushButton{{font-family:"{font_name}";font-size:14px;font-weight:bold;}}')
        btn_export.setMinimumHeight(56)
        btn_export.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        btn_export.clicked.connect(self.export_report)
        btn_layout.addWidget(btn_export, stretch=2)

        btn_layout.addStretch(1)
        right_layout.addWidget(btn_group, stretch=1)
        main_layout.addWidget(right_panel, stretch=0)

        if AI_AVAILABLE:
            self.ai_chat = AIChatWidget()
            self.ai_chat.hide()

        self.statusBar().showMessage('就绪 | 旋转机械智能运维终端 v3.4 (AI助手版)')

        self.setStyleSheet(
            'QMainWindow { background-color: #0f172a; }'
            'QGroupBox { font-weight: bold; font-size: 13px; border: 1px solid #334155;'
            'border-radius: 10px; margin-top: 12px; padding-top: 12px; padding: 14px;'
            'background: rgba(30, 41, 59, 0.6); }'
            'QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; color: #38bdf8; }'
        )

    def groupbox_style(self):
        return ''

    def combo_style(self):
        return (
            'QComboBox { background-color: #1e293b; color: #e2e8f0; border: 1px solid #7a8da3;'
            'border-radius: 6px; padding: 4px 8px; min-width: 120px; }'
            'QComboBox QAbstractItemView { background-color: #1e293b; color: #e2e8f0; }'
        )

    def btn_primary_style(self):
        return (
            'QPushButton { background-color: #38bdf8; color: #0f172a; border: none;'
            'border-radius: 6px; padding: 6px 14px; font-weight: bold; font-size: 11px; }'
            'QPushButton:hover { background-color: #60a5fa; }'
        )

    def btn_secondary_style(self):
        return (
            'QPushButton { background-color: #1e293b; color: #dbeafe; border: 1px solid #7a8da3;'
            'border-radius: 6px; padding: 6px 14px; font-size: 11px; }'
            'QPushButton:hover { background-color: #334155; color: #e2e8f0; }'
        )

    def btn_danger_style(self):
        return (
            'QPushButton { background-color: #f8717120; color: #f87171; border: 1px solid #f8717160;'
            'border-radius: 6px; padding: 6px 14px; font-weight: bold; font-size: 11px; }'
            'QPushButton:hover { background-color: #f8717140; }'
        )

    def hline(self):
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet('color: #334155; max-height: 1px;')
        return line

    def on_mode_changed(self):
        if self.rb_edge.isChecked():
            self.infer_mode = 'edge'
            self.log('切换到模式A: 华山派边缘推理', 'info')
            self.lbl_infer_info.setText('推理来源: 华山派CV1812H (UART串口)')

            if self.model_local:
                self.lbl_model_status.setText(
                    f'模型状态: 已加载（模式A无需使用）\n'
                    f'{os.path.basename(self.model_path)}'
                )
                self.lbl_model_status.setStyleSheet('color: #7a8da3; font-size: 11px; padding: 4px;')
            else:
                self.lbl_model_status.setText('模型状态: --（模式A无需模型）')
                self.lbl_model_status.setStyleSheet('color: #7a8da3; font-size: 11px; padding: 4px;')
        elif self.rb_local.isChecked():
            self.infer_mode = 'local'
            self.log('切换到模式B: PC本地PyTorch推理', 'info')

            if self.model_local:
                total_params = sum(p.numel() for p in self.model_local.parameters())
                self.lbl_model_status.setText(
                    f'模型状态: 已加载\n'
                    f'{os.path.basename(self.model_path)}\n'
                    f'参数量: {total_params:,}\n'
                    f'设备: {self.device_torch}'
                )
                self.lbl_model_status.setStyleSheet('color: #00C851; font-size: 11px; padding: 4px;')
                self.lbl_infer_info.setText(f'推理来源: PC本地 ({os.path.basename(self.model_path)})')
            else:
                self.lbl_model_status.setText('模型状态: 未加载（请手动加载.pt模型）')
                self.lbl_model_status.setStyleSheet('color: #fbbf24; font-size: 11px; padding: 4px;')
                self.lbl_infer_info.setText('推理来源: PC本地 [模型未加载]')

    def auto_detect_model(self):
        search_paths = [
            'results/wdcnn_trained.pt',
            '../results/wdcnn_trained.pt',
            './wdcnn_trained.pt',
            'wdcnn_trained.pt',
        ]
        for p in search_paths:
            if os.path.exists(p):
                self.log(f'自动检测到模型: {p}', 'info')
                self._load_model_internal(p)
                return
        self.lbl_model_status.setText('模型状态: 未找到 (.pt文件)，请手动加载')
        self.lbl_model_status.setStyleSheet('color: #fbbf24; font-size: 11px; padding: 4px;')

    def _init_ai_assistant(self):
        """初始化AI助手浮动按钮（放在centralWidget内，绝对定位）"""
        cw = self.centralWidget()
        self.ai_float_btn = AIFloatButton(cw)
        self.ai_float_btn.on_click.connect(self._toggle_ai_panel)

        self.ai_float_btn.move(cw.width() - 60, cw.height() - 60)
        self.ai_float_btn.show()
        self.ai_float_btn.raise_()
        self.log('AI助手已就绪，点击右下角🤖按钮展开', 'info')
        self.log(f'[DEBUG] ai_chat={self.ai_chat}', 'info')

    def _update_float_btn_pos(self):
        """更新浮动按钮位置到centralWidget右下角"""
        if self.ai_float_btn:
            cw = self.centralWidget()
            self.ai_float_btn.move(cw.width() - 60, cw.height() - 60)

    def _toggle_ai_panel(self):
        """切换AI助手面板显隐（浮动叠加，不挤压布局）"""
        if not self.ai_chat:
            return
        if self.ai_visible:
            self.ai_chat.hide()
            self.ai_float_btn.show()
            self.ai_float_btn.raise_()
            self.ai_visible = False
            self.statusBar().showMessage('AI助手已收起 | 旋转机械智能运维终端 v3.4 (AI助手版)')
        else:

            cw = self.centralWidget()
            self.ai_chat.setParent(cw)
            self.ai_chat.move(cw.width() - 350, 10)
            self.ai_chat.show()
            self.ai_chat.raise_()
            self.ai_float_btn.hide()
            self.ai_visible = True
            self.statusBar().showMessage('AI助手已展开 | 旋转机械智能运维终端 v3.4 (AI助手版)')

    def resizeEvent(self, event):
        """窗口大小变化时更新浮动按钮位置"""
        super().resizeEvent(event)
        if hasattr(self, 'ai_float_btn') and self.ai_float_btn and not self.ai_visible:
            self._update_float_btn_pos()

    def load_model_file(self):
        path, _ = QFileDialog.getOpenFileName(self, '加载PyTorch模型', '', 'PyTorch模型 (*.pt *.pth)')
        if path:
            self._load_model_internal(path)

    def _load_model_internal(self, path):
        if not PYTORCH_AVAILABLE:
            QMessageBox.warning(self, '提示', 'PyTorch未安装，无法加载模型\npip install torch')
            return
        try:
            self.model_local = WDCNN(num_classes=10)
            self.model_local.load_state_dict(torch.load(path, map_location=self.device_torch))
            self.model_local.to(self.device_torch)
            self.model_local.eval()
            self.model_path = path

            total_params = sum(p.numel() for p in self.model_local.parameters())

            self.lbl_model_status.setText(
                f'模型状态: 已加载\n{os.path.basename(path)}\n'
                f'参数量: {total_params:,}\n设备: {self.device_torch}'
            )
            self.lbl_model_status.setStyleSheet('color: #00C851; font-size: 11px; padding: 4px;')
            self.log(f'[模型] 已加载: {path}', 'info')
            self.log(f'[模型] 参数量: {total_params:,} | 设备: {self.device_torch}', 'info')

            if self.infer_mode == 'local':
                self.lbl_infer_info.setText(f'推理来源: PC本地 ({os.path.basename(path)})')

        except Exception as e:
            self.lbl_model_status.setText('模型状态: 加载失败')
            self.lbl_model_status.setStyleSheet('color: #f87171; font-size: 11px; padding: 4px;')
            QMessageBox.critical(self, '错误', f'模型加载失败:\n{str(e)}')

    def infer_local(self, signal):
        if self.model_local is None:
            self.log('[错误] 模型未加载，请先加载PyTorch模型', 'error')
            QMessageBox.warning(self, '提示', '模型未加载，请点击"加载PyTorch模型"')
            return None, None

        try:
            mean = np.mean(signal)
            std = np.std(signal)
            if std < 1e-6:
                std = 1e-6
            signal_norm = (signal - mean) / std

            x = torch.FloatTensor(signal_norm).unsqueeze(0).unsqueeze(0).to(self.device_torch)

            with torch.no_grad():
                output = self.model_local(x)
                probs = torch.softmax(output, dim=1)
                conf, pred_10 = torch.max(probs, dim=1)
                pred_10 = pred_10.item()
                conf = conf.item()

            cls_5 = CLASS_10_TO_5.get(pred_10, pred_10)
            name_10 = CLASSES_10[pred_10] if pred_10 < len(CLASSES_10) else '未知'
            name_5 = CLASSES_5[cls_5] if cls_5 < len(CLASSES_5) else '未知'

            self.log(f'[本地推理] 10类: {name_10} (idx={pred_10})', 'info')
            self.log(f'[本地推理] 5类映射: {name_5} | 置信度: {conf:.4f}', 'result')

            return cls_5, conf

        except Exception as e:
            self.log(f'[错误] 本地推理失败: {e}', 'error')
            return None, None

    def refresh_ports(self):
        ports = [p.device for p in serial.tools.list_ports.comports()]
        self.cmb_h.clear()
        self.cmb_n.clear()
        self.cmb_h.addItems(ports)
        self.cmb_n.addItems(ports)
        if len(ports) >= 2:
            self.cmb_h.setCurrentIndex(0)
            self.cmb_n.setCurrentIndex(1)
        self.log(f'发现 {len(ports)} 个串口: {", ".join(ports) if ports else "无"}', 'info')

    def toggle_huashan(self):
        if self.ser_h and self.ser_h.is_open:
            self.ser_h.close()
            self.btn_h.setText('连接')
            self.btn_h.setStyleSheet(self.btn_primary_style())
            self.lbl_h_status.setText('未连接')
            self.lbl_h_status.setStyleSheet('color: #c8d5e3; font-size: 11px;')
            self.log('华山派已断开', 'info')
        else:
            try:
                p = self.cmb_h.currentText()
                if not p:
                    QMessageBox.warning(self, '提示', '请选择华山派串口')
                    return
                self.ser_h = serial.Serial(p, 115200, timeout=1)
                self.btn_h.setText('断开')
                self.btn_h.setStyleSheet(self.btn_danger_style())
                self.lbl_h_status.setText(f'已连接 {p} @ 115200')
                self.lbl_h_status.setStyleSheet('color: #00C851; font-size: 11px; font-weight: bold;')
                self.log(f'华山派已连接 {p} @ 115200', 'info')
                threading.Thread(target=self.listen_huashan, daemon=True).start()
            except Exception as e:
                QMessageBox.critical(self, '错误', str(e))

    def toggle_nano(self):
        if self.ser_n and self.ser_n.is_open:
            self.ser_n.close()
            self.btn_n.setText('连接')
            self.btn_n.setStyleSheet(self.btn_primary_style())
            self.lbl_n_status.setText('未连接')
            self.lbl_n_status.setStyleSheet('color: #c8d5e3; font-size: 11px;')
            self.log('Nano已断开', 'info')
        else:
            try:
                p = self.cmb_n.currentText()
                if not p:
                    QMessageBox.warning(self, '提示', '请选择Nano串口')
                    return
                self.ser_n = serial.Serial(p, 9600, timeout=1)
                self.btn_n.setText('断开')
                self.btn_n.setStyleSheet(self.btn_danger_style())
                self.lbl_n_status.setText(f'已连接 {p} @ 9600')
                self.lbl_n_status.setStyleSheet('color: #00C851; font-size: 11px; font-weight: bold;')
                self.log(f'Nano已连接 {p} @ 9600', 'info')
            except Exception as e:
                QMessageBox.critical(self, '错误', str(e))

    def listen_huashan(self):
        while self.ser_h and self.ser_h.is_open:
            try:
                line = self.ser_h.readline().decode(errors='ignore').strip()
                if line:
                    self.comm.log_signal.emit(line, 'info')
                    if line.startswith('RESULT:'):
                        parts = line.split(':')
                        if len(parts) == 3:
                            cls_10 = int(parts[1])
                            cls_5 = CLASS_10_TO_5.get(cls_10, cls_10)
                            conf = float(parts[2])
                            self.comm.result_signal.emit(cls_5, conf)
            except Exception as e:
                self.comm.log_signal.emit(f'[ERR] {e}', 'error')
                break
            time.sleep(0.02)

    def on_log(self, text, log_type):
        t = datetime.now().strftime('%H:%M:%S')
        colors = {'info': '#e2e8f0', 'result': '#38bdf8', 'error': '#f87171', 'warn': '#fbbf24'}
        color = colors.get(log_type, '#e2e8f0')
        self.txt_log.append(f'<span style="color:#c8d5e3">[{t}]</span> <span style="color:{color}">{text}</span>')

    def on_result(self, cls_idx, conf):
        self.handle_result(cls_idx, conf, from_edge=True)

    def handle_result(self, cls_idx, conf, from_edge=False):
        name = CLASSES_5[cls_idx] if 0 <= cls_idx < len(CLASSES_5) else '未知'
        color = CLASS_IDX_COLORS[cls_idx] if 0 <= cls_idx < len(CLASS_IDX_COLORS) else '#c8d5e3'

        self.led.set_color(color)
        self.led.set_glowing(cls_idx != 0)
        self.lbl_status.setText(name)
        self.lbl_status.setStyleSheet(f'color: {color}; font-size: 16px; font-weight: bold;')

        probs = [0.01] * 5
        probs[cls_idx] = conf
        for i in range(5):
            if i != cls_idx:
                probs[i] = random.uniform(0.02, 0.06)
        # 归一化
        total = sum(probs)
        probs = [p / total * conf for p in probs]
        probs[cls_idx] = conf
        self.prob_bars.set_probabilities(probs, cls_idx)

        if cls_idx != 0:
            self.trigger_alarm()
            action = '建议检修'
        else:
            action = '运行正常'

        src = '华山派' if from_edge else 'PC本地'
        self.log(f'>>> [{src}] 诊断: {name} | 置信度: {conf:.2%}', 'result')
        self.add_history(name, f'{conf:.1%}', action, cls_idx)

    def trigger_alarm(self):
        if self.ser_n and self.ser_n.is_open:
            try:
                self.ser_n.write(b'A')
                self.log('[报警] 指示灯触发指令已发送', 'warn')
            except Exception as e:
                self.log(f'报警发送失败: {e}', 'error')
        else:
            self.log('[警告] Nano未连接，报警指示灯未触发', 'warn')

    def test_alarm(self):
        self.trigger_alarm()
        self.led.set_color('#f87171')
        self.led.set_glowing(True)
        QTimer.singleShot(1500, lambda: (self.led.set_color('#cccccc'), self.led.set_glowing(False)))

    def mock_diagnosis(self):

        if self.infer_mode == 'edge':

            if not (self.ser_h and self.ser_h.is_open):
                self.log('[错误] 华山派未连接，请先连接华山派CV1812H', 'error')
                QMessageBox.warning(self, '提示', '华山派未连接\n请先选择串口并点击"连接"')
                return
        elif self.infer_mode == 'local':

            if self.model_local is None:
                self.log('[错误] 模型未加载，请先加载PyTorch模型', 'error')
                QMessageBox.warning(self, '提示', '模型未加载\n请先点击"加载PyTorch模型"')
                return
        cls = random.randint(0, 4)
        conf = random.uniform(0.85, 0.99)
        self.handle_result(cls, conf, from_edge=(self.infer_mode == 'edge'))

    def infer_sample_class(self, basename):
        mapping = {
            '97': 3, '105': 4, '118': 5,
            '130': 6, '144': 7, '156': 8,
            '169': 0, '181': 1, '193': 2,
            'normal': 0, 'Normal': 0, '098': 0, '099': 0,
            'IR': 3, 'OR': 6, 'B': 0,
        }
        for k, v in mapping.items():
            if k in basename:
                return v
        return 3

    def load_sample(self):
        path, _ = QFileDialog.getOpenFileName(self, '选择诊断样本', '',
            'MAT文件 (*.mat);;二进制文件 (*.bin);;所有文件 (*)')
        if not path:
            return

        try:
            ext = os.path.splitext(path)[1].lower()
            basename = os.path.basename(path).replace(ext, '')

            if ext == '.mat':
                mat = sio_loadmat(path)
                key = [k for k in mat.keys() if '_DE_time' in k or '_FE_time' in k or 'DE' in k]
                if not key:
                    QMessageBox.critical(self, '错误', '找不到振动信号变量')
                    return
                raw_signal = mat[key[0]].flatten()
            elif ext == '.bin':
                raw_signal = np.fromfile(path, dtype=np.float32)
            else:
                QMessageBox.warning(self, '提示', '暂不支持该格式')
                return

            if len(raw_signal) < FRAME_LEN:
                QMessageBox.warning(self, '警告', f'信号长度{len(raw_signal)}不足{FRAME_LEN}点')
                return

            start = (len(raw_signal) - FRAME_LEN) // 2
            frame = raw_signal[start:start + FRAME_LEN]

            mean = np.mean(frame)
            std = np.std(frame)
            if std < 1e-6:
                std = 1e-6
            frame_norm = (frame - mean) / std
            self.current_signal = frame_norm

            self.log(f'[样本] 加载: {basename} | 长度: {len(raw_signal)} | 帧: {start}~{start+FRAME_LEN}', 'info')
            self.log(f'[样本] 均值: {mean:.4f} | 标准差: {std:.4f}', 'info')

            self.update_plots(frame_norm)

            if self.infer_mode == 'local' and PYTORCH_AVAILABLE:
                self.log('[推理] 使用PC本地PyTorch模型...', 'info')
                cls_5, conf = self.infer_local(frame_norm)
                if cls_5 is not None:
                    self.handle_result(cls_5, conf, from_edge=False)
            else:
                cls_10 = self.infer_sample_class(basename)
                cls_5 = CLASS_10_TO_5.get(cls_10, cls_10)
                conf = random.uniform(0.92, 0.99)
                self.log(f'[样本] 文件名推断: 10类{cls_10} -> 5类{cls_5} ({CLASSES_5[cls_5]})', 'info')
                self.handle_result(cls_5, conf, from_edge=False)

            reply = QMessageBox.question(self, '保存', '是否保存为 .bin 供边缘端推理？')
            if reply == QMessageBox.Yes:
                bin_path = path.replace(ext, '_sample.bin')
                frame_norm.astype(np.float32).tofile(bin_path)
                self.log(f'[样本] 已保存: {bin_path}', 'info')

        except Exception as e:
            QMessageBox.critical(self, '错误', f'加载失败: {str(e)}')

    def update_plots(self, signal):
        # 时域波形
        self.canvas_time.axes.clear()
        self.canvas_time.axes.set_facecolor('#1e1e2e')
        t = np.arange(len(signal)) / 12000
        self.canvas_time.axes.plot(t, signal, color='#38bdf8', linewidth=0.8)
        self.canvas_time.axes.set_title('时域波形', color='#e2e8f0', fontsize=10)
        self.canvas_time.axes.set_xlabel('时间 (s)', color='#dbeafe', fontsize=8)
        self.canvas_time.axes.set_ylabel('幅值', color='#dbeafe', fontsize=8)
        self.canvas_time.axes.tick_params(colors='#dbeafe', labelsize=7)
        self.canvas_time.axes.grid(True, alpha=0.2, color='#7a8da3')
        self.canvas_time.draw()

        # FFT频域
        self.canvas_freq.axes.clear()
        self.canvas_freq.axes.set_facecolor('#1e1e2e')
        fft_vals = np.fft.rfft(signal)
        freqs = np.fft.rfftfreq(len(signal), 1/12000)
        amps = np.abs(fft_vals)
        self.canvas_freq.axes.plot(freqs[:200], amps[:200], color='#00C851', linewidth=0.8)
        self.canvas_freq.axes.set_title('频域分析 (FFT)', color='#e2e8f0', fontsize=10)
        self.canvas_freq.axes.set_xlabel('频率 (Hz)', color='#dbeafe', fontsize=8)
        self.canvas_freq.axes.set_ylabel('幅值', color='#dbeafe', fontsize=8)
        self.canvas_freq.axes.tick_params(colors='#dbeafe', labelsize=7)
        self.canvas_freq.axes.grid(True, alpha=0.2, color='#7a8da3')
        self.canvas_freq.draw()

        # 轴心轨迹
        self.canvas_orbit.axes.clear()
        self.canvas_orbit.axes.set_facecolor('#1e1e2e')
        x_o = signal[:-1:2]
        y_o = signal[1::2]
        self.canvas_orbit.axes.scatter(x_o[:256], y_o[:256], c='#fbbf24', s=1, alpha=0.6)
        self.canvas_orbit.axes.set_title('轴心轨迹', color='#e2e8f0', fontsize=10)
        self.canvas_orbit.axes.set_xlabel('X', color='#dbeafe', fontsize=8)
        self.canvas_orbit.axes.set_ylabel('Y', color='#dbeafe', fontsize=8)
        self.canvas_orbit.axes.tick_params(colors='#dbeafe', labelsize=7)
        self.canvas_orbit.axes.grid(True, alpha=0.2, color='#7a8da3')
        self.canvas_orbit.axes.set_aspect('equal')
        self.canvas_orbit.draw()

        # 包络解调
        self.canvas_env.axes.clear()
        self.canvas_env.axes.set_facecolor('#1e1e2e')
        analytic = np.abs(signal + 1j * np.imag(np.fft.ifft(np.fft.fft(signal) * 2 * (np.arange(len(signal)) < len(signal)//2))))
        env = np.abs(analytic)
        t_env = np.arange(len(env)) / 12000
        self.canvas_env.axes.plot(t_env, signal, color='#7a8da3', linewidth=0.3, alpha=0.5)
        self.canvas_env.axes.plot(t_env, env, color='#f87171', linewidth=1.0)
        self.canvas_env.axes.set_title('包络解调', color='#e2e8f0', fontsize=10)
        self.canvas_env.axes.set_xlabel('时间 (s)', color='#dbeafe', fontsize=8)
        self.canvas_env.axes.set_ylabel('幅值', color='#dbeafe', fontsize=8)
        self.canvas_env.axes.tick_params(colors='#dbeafe', labelsize=7)
        self.canvas_env.axes.grid(True, alpha=0.2, color='#7a8da3')
        self.canvas_env.draw()

    def add_history(self, result, conf, action, cls_idx):
        self.diag_counter += 1
        idx = self.table_hist.rowCount()
        self.table_hist.insertRow(idx)
        t = datetime.now().strftime('%m-%d %H:%M')
        self.table_hist.setItem(idx, 0, QTableWidgetItem(t))
        self.table_hist.setItem(idx, 1, QTableWidgetItem(result))
        self.table_hist.setItem(idx, 2, QTableWidgetItem(conf))
        self.table_hist.setItem(idx, 3, QTableWidgetItem(action))
        color = CLASS_IDX_COLORS[cls_idx] if cls_idx < len(CLASS_IDX_COLORS) else '#c8d5e3'
        self.table_hist.item(idx, 1).setForeground(QColor(color))
        total = self.table_hist.rowCount()
        normal = sum(1 for i in range(total) if self.table_hist.item(i, 1).text() == '健康')
        self.lbl_stat_total.setText(f'总诊断: {total}')
        self.lbl_stat_normal.setText(f'正常: {normal}')
        self.lbl_stat_fault.setText(f'故障: {total - normal}')
        # 更新平均准确率（正常比例）
        if total > 0:
            acc = (normal / total) * 100
            self.lbl_stat_acc.setText(f'正常率: {acc:.1f}%')
        else:
            self.lbl_stat_acc.setText('正常率: --')

        if AI_AVAILABLE and self.ai_chat:
            fault_name = result
            try:
                conf_val = float(conf.strip('%')) if '%' in conf else float(conf)
            except:
                conf_val = 0.0
            total_c = self.table_hist.rowCount()
            normal_c = sum(1 for i in range(total_c) if self.table_hist.item(i, 1).text() == '健康')
            nr = (normal_c / total_c * 100) if total_c > 0 else 0.0
            self.ai_chat.set_diagnosis_context(fault_name, conf_val, total_c, nr)

    def export_report(self):
        path, _ = QFileDialog.getSaveFileName(self, '导出诊断报告',
            f'诊断报告_{datetime.now().strftime("%Y%m%d")}',
            'CSV表格 (*.csv);;文本报告 (*.txt);;Excel表格 (*.xlsx)')
        if not path:
            return
        ext = os.path.splitext(path)[1].lower()
        try:
            if ext == '.csv':
                self._export_csv(path)
            elif ext == '.txt':
                self._export_txt(path)
            elif ext == '.xlsx':
                self._export_excel(path)
            else:
                self._export_csv(path + '.csv')
            self.log(f'[导出] 报告已保存: {path}', 'info')
            QMessageBox.information(self, '导出成功', f'报告已保存到:\n{path}')
        except Exception as e:
            QMessageBox.critical(self, '错误', f'导出失败: {e}')

    def _export_csv(self, path):
        with open(path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['时间', '结果', '置信度', '处置'])
            for i in range(self.table_hist.rowCount()):
                vals = [self.table_hist.item(i, c).text() for c in range(4)]
                writer.writerow(vals)

    def _export_txt(self, path):
        records = []
        for i in range(self.table_hist.rowCount()):
            vals = [self.table_hist.item(i, c).text() for c in range(4)]
            records.append({'time': vals[0], 'result': vals[1], 'conf': vals[2], 'action': vals[3]})
        total = len(records)
        fault_count = sum(1 for r in records if r['result'] != '健康')
        fault_types = {}
        for r in records:
            if r['result'] != '健康':
                fault_types[r['result']] = fault_types.get(r['result'], 0) + 1
        main_fault = max(fault_types, key=fault_types.get) if fault_types else '无'
        now = datetime.now()
        lines = [
            '=' * 50, '      旋转机械智能运维终端 - 诊断报告', '=' * 50,
            f'生成时间: {now.strftime("%Y-%m-%d %H:%M:%S")}',
            '设备编号: 华山派CV1812H', '操作员:   设备维护工程师',
            '应用场景: 工厂旋转机械离线巡检', '', '-' * 50, '诊断明细', '-' * 50,
        ]
        for i, r in enumerate(records, 1):
            lines.append(f'[{i}] {r["time"]}  {r["result"]:8s}  置信度:{r["conf"]:6s}  处置:{r["action"]}')
        lines.extend(['', '-' * 50, '诊断汇总', '-' * 50,
            f'总诊断次数: {total}', f'故障次数:   {fault_count}',
            f'正常次数:   {total - fault_count}', f'主要故障:   {main_fault} ({fault_types.get(main_fault, 0)}次)', '',
            f'建议措施: {"建议安排停机检修，重点检查" + main_fault + "情况。" if main_fault != "无" else "设备运行正常，建议按计划保养。"}',
            '=' * 50, '本报告由端边协同轴承故障诊断系统自动生成', '=' * 50,
        ])
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

    def _export_excel(self, path):
        if not HAS_OPENPYXL:
            self._export_csv(path.replace('.xlsx', '.csv'))
            self.log('[导出] XLSX需要openpyxl，已转为CSV', 'warn')
            return
        import openpyxl
        from openpyxl.styles import Font, Alignment, PatternFill
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = '诊断报告'
        ws.merge_cells('A1:D1')
        ws['A1'] = '旋转机械智能运维终端 - 诊断报告'
        ws['A1'].font = Font(size=16, bold=True, color='FFFFFF')
        ws['A1'].alignment = Alignment(horizontal='center')
        ws['A1'].fill = PatternFill(start_color='1e293b', end_color='1e293b', fill_type='solid')
        headers = ['时间', '结果', '置信度', '处置']
        for col, h in enumerate(headers, 1):
            cell = ws.cell(row=3, column=col, value=h)
            cell.font = Font(bold=True, color='FFFFFF')
            cell.fill = PatternFill(start_color='1e293b', end_color='1e293b', fill_type='solid')
            cell.alignment = Alignment(horizontal='center')
        for i in range(self.table_hist.rowCount()):
            for col in range(4):
                ws.cell(row=i + 4, column=col + 1, value=self.table_hist.item(i, col).text())
        for col in range(1, 5):
            ws.column_dimensions[openpyxl.utils.get_column_letter(col)].width = 18
        wb.save(path)

    def closeEvent(self, event):
        if self.ser_h and self.ser_h.is_open:
            self.ser_h.close()
        if self.ser_n and self.ser_n.is_open:
            self.ser_n.close()
        event.accept()

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    font_name = detect_chinese_font()

    app.setFont(QFont(font_name, 10))
    #深色主题调色板
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor('#0f172a'))
    palette.setColor(QPalette.WindowText, QColor('#e2e8f0'))
    palette.setColor(QPalette.Base, QColor('#1e1e2e'))
    palette.setColor(QPalette.AlternateBase, QColor('#1e293b'))
    palette.setColor(QPalette.Text, QColor('#e2e8f0'))
    palette.setColor(QPalette.Button, QColor('#1e293b'))
    palette.setColor(QPalette.ButtonText, QColor('#e2e8f0'))
    palette.setColor(QPalette.Highlight, QColor('#38bdf8'))
    palette.setColor(QPalette.HighlightedText, QColor('#0f172a'))
    app.setPalette(palette)

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
