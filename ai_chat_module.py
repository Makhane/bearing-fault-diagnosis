# -*- coding: utf-8 -*-
"""
AI助手模块 — 大模型集成
支持所有OpenAI兼容API（OpenAI/通义千问/DeepSeek/Kimi/文心/本地Ollama等）

用法：
    1. 在主窗口中 import 并添加 AIChatWidget
    2. 配置 API Key 和模型参数
    3. 调用 set_diagnosis_context() 传入诊断上下文
"""

import json
import os
import sys
import time

from PyQt5.QtCore import QThread, pyqtSignal, Qt, QSettings
from PyQt5.QtGui import QTextCursor, QFont, QColor
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QLineEdit,
    QPushButton, QLabel, QComboBox, QStackedWidget, QFrame,
    QSizePolicy, QMessageBox, QDialog, QFormLayout, QSpinBox,
    QDoubleSpinBox, QTabWidget
)

class LLMRequestThread(QThread):
    """大模型请求线程 — 流式输出"""
    chunk_signal = pyqtSignal(str)       # 每个token
    done_signal = pyqtSignal()           # 完成
    error_signal = pyqtSignal(str)       # 错误

    def __init__(self, api_key, base_url, model, messages, temperature=0.7, max_tokens=1024):
        super().__init__()
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.messages = messages
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._running = True

    def run(self):
        try:
            import urllib.request
            import urllib.error

            url = f"{self.base_url}/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            data = {
                "model": self.model,
                "messages": self.messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "stream": True
            }

            req = urllib.request.Request(
                url,
                data=json.dumps(data).encode('utf-8'),
                headers=headers,
                method='POST'
            )

            with urllib.request.urlopen(req, timeout=60) as resp:
                for raw_line in resp:
                    if not self._running:
                        break
                    line = raw_line.decode('utf-8', errors='ignore').strip()
                    if not line.startswith('data:'):
                        continue
                    line = line[5:].strip()
                    if line == '[DONE]':
                        break
                    try:
                        chunk = json.loads(line)
                        delta = chunk['choices'][0].get('delta', {})
                        content = delta.get('content', '')
                        if content:
                            self.chunk_signal.emit(content)
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue

            self.done_signal.emit()

        except urllib.error.HTTPError as e:
            err_body = e.read().decode('utf-8', errors='ignore')[:200]
            self.error_signal.emit(f'API错误 [{e.code}]: {err_body}')
        except Exception as e:
            self.error_signal.emit(str(e))

    def stop(self):
        self._running = False


class LLMClient:
    """大模型客户端 — 支持多种预设配置"""

    PRESETS = {
        'OpenAI': {
            'base_url': 'https://api.openai.com/v1',
            'model': 'gpt-3.5-turbo',
        },
        '通义千问': {
            'base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
            'model': 'qwen-turbo',
        },
        'DeepSeek': {
            'base_url': 'https://api.deepseek.com/v1',
            'model': 'deepseek-chat',
        },
        'Kimi (Moonshot)': {
            'base_url': 'https://api.moonshot.cn/v1',
            'model': 'moonshot-v1-8k',
        },
        '本地 Ollama': {
            'base_url': 'http://localhost:11434/v1',
            'model': 'qwen2:7b',
        },
        '自定义': {
            'base_url': '',
            'model': '',
        },
    }

    PROMPT_TEMPLATES = {
        '解释结果': '''你是一位资深的旋转机械故障诊断专家，拥有20年工厂设备维护经验。

当前系统检测到以下诊断结果：
- 故障类型：{fault_name}
- 置信度：{confidence:.1f}%
- 诊断时间：{time_str}

请直接陈述技术事实，不要假设用户姓名
1. 这个故障意味着什么？对设备运行有什么影响？
2. 可能导致这个故障的常见原因有哪些？
3. 建议立即采取什么措施？
4. 如果不及时处理，可能会有什么后果？

请用中文回答，条理清晰，控制在300字以内。''',

        '生成报告': '''请基于以下诊断数据，生成一份专业的设备诊断报告摘要：

设备信息：旋转机械轴承
故障类型：{fault_name}
置信度：{confidence:.1f}%
诊断时间：{time_str}
历史诊断次数：{total_count}
正常率：{normal_rate:.1f}%

请生成一段200字以内的报告摘要，包含：
1. 诊断结论
2. 严重程度评估
3. 建议措施
4. 后续跟踪建议''',

        '处置建议': '''作为轴承故障诊断专家，针对"{fault_name}"故障（置信度{confidence:.1f}%），请给出具体的处置建议：

1. 紧急程度判断（立即停机/尽快维修/继续观察）
2. 具体检查步骤
3. 可能需要更换的部件
4. 维修预计时间
5. 预防措施

请用中文回答，简洁实用。''',
    }

    def __init__(self, api_key='', provider='DeepSeek', base_url='', model='', temperature=0.7):
        self.api_key = api_key
        self.temperature = temperature
        # 使用INI文件存储配置，跨平台兼容
        self.settings = QSettings(
            QSettings.IniFormat, QSettings.UserScope,
            'BearingDiag', 'AIConfig'
        )

        if provider in self.PRESETS and provider != '自定义':
            preset = self.PRESETS[provider]
            self.base_url = preset['base_url']
            self.model = preset['model']
        else:
            self.base_url = base_url
            self.model = model

    def save_config(self):
        self.settings.setValue('api_key', self.api_key)
        self.settings.setValue('base_url', self.base_url)
        self.settings.setValue('model', self.model)
        self.settings.setValue('temperature', self.temperature)
        self.settings.sync()  # 强制写入磁盘

    def load_config(self):
        self.api_key = self.settings.value('api_key', '')
        self.base_url = self.settings.value('base_url', 'https://api.deepseek.com/v1')
        self.model = self.settings.value('model', 'deepseek-chat')
        temp = self.settings.value('temperature', 0.7)
        try:
            self.temperature = float(temp)
        except (TypeError, ValueError):
            self.temperature = 0.7

    def chat(self, messages):
        """发起对话请求，返回LLMRequestThread对象（需连接信号）"""
        thread = LLMRequestThread(
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.model,
            messages=messages,
            temperature=self.temperature
        )
        return thread

    def make_messages(self, system_prompt, user_content):
        """构造消息列表"""
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

    def format_diagnosis_context(self, template_name, **kwargs):
        """填充诊断上下文到Prompt模板"""
        if template_name in self.PROMPT_TEMPLATES:
            return self.PROMPT_TEMPLATES[template_name].format(**kwargs)
        return kwargs.get('user_input', '')


class AIChatWidget(QFrame):
    """AI助手对话面板 — 可折叠"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.llm = LLMClient()
        self.llm.load_config()
        self.current_thread = None
        self.diagnosis_context = {}

        self._build_ui()
        self._apply_style()

    def _build_ui(self):
        self.setFixedWidth(340)
        self.setWindowFlags(Qt.FramelessWindowHint)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ---- 可拖动标题栏 ----
        title_bar = QFrame()
        title_bar.setFixedHeight(40)
        title_bar.setStyleSheet('background:#1e3a5f;border-radius:8px 8px 0 0;')
        title_bar.setCursor(Qt.OpenHandCursor)
        title_layout = QHBoxLayout(title_bar)
        title_layout.setContentsMargins(10, 0, 6, 0)
        title_layout.setSpacing(4)

        self.title_label = QLabel('🤖 AI 智能助手')
        self.title_label.setStyleSheet('font-size:14px;font-weight:bold;color:#f1f5f9;')
        self.title_label.setCursor(Qt.OpenHandCursor)

        self.config_btn = QPushButton('⚙️')
        self.config_btn.setFixedSize(30, 30)
        self.config_btn.setToolTip('配置API')
        self.config_btn.setCursor(Qt.PointingHandCursor)
        self.config_btn.setStyleSheet('background:transparent;color:#94a3b8;font-size:14px;border:none;')

        self.close_btn = QPushButton('✕')
        self.close_btn.setFixedSize(30, 30)
        self.close_btn.setToolTip('关闭')
        self.close_btn.setCursor(Qt.PointingHandCursor)
        self.close_btn.setStyleSheet('background:transparent;color:#94a3b8;font-size:14px;border:none;')
        self.close_btn.setStyleSheet('''
            QPushButton { background:transparent; color:#94a3b8; font-size:14px; border:none; border-radius:4px; }
            QPushButton:hover { background:#ef4444; color:#fff; }
        ''')

        title_layout.addWidget(self.title_label)
        title_layout.addStretch()
        title_layout.addWidget(self.config_btn)
        title_layout.addWidget(self.close_btn)
        layout.addWidget(title_bar)

        # 标题栏拖动事件
        title_bar.mousePressEvent = self._title_mouse_press
        title_bar.mouseMoveEvent = self._title_mouse_move
        title_bar.mouseReleaseEvent = self._title_mouse_release
        self.title_label.mousePressEvent = self._title_mouse_press
        self.title_label.mouseMoveEvent = self._title_mouse_move
        self.title_label.mouseReleaseEvent = self._title_mouse_release

        # ---- 内容容器（带内边距）----
        content = QFrame()
        content.setStyleSheet('background:#1e293b;border-radius:0 0 8px 8px;')
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(10, 10, 10, 10)
        content_layout.setSpacing(8)
        layout.addWidget(content, stretch=1)

        # ---- 快捷指令按钮 ----
        quick = QHBoxLayout()
        quick.setSpacing(6)

        self.btn_explain = QPushButton('🔍 解释结果')
        self.btn_report = QPushButton('📋 生成报告')
        self.btn_advice = QPushButton('💡 处置建议')

        for btn in [self.btn_explain, self.btn_report, self.btn_advice]:
            btn.setFixedHeight(32)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setStyleSheet('''
                QPushButton {
                    background:#334155;border:1px solid #475569;
                    border-radius:6px;color:#e2e8f0;
                    font-size:12px;padding:4px 8px;
                }
                QPushButton:hover { background:#475569;border-color:#64748b; }
            ''')
            quick.addWidget(btn)

        content_layout.addLayout(quick)

        # ---- 对话显示区 ----
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setMinimumHeight(300)
        self.chat_display.setStyleSheet('''
            QTextEdit {
                background:#0f172a;border:1px solid #334155;
                border-radius:8px;color:#e2e8f0;
                font-size:13px;line-height:1.6;
                padding:8px;
            }
        ''')
        content_layout.addWidget(self.chat_display)

        # ---- 输入区 ----
        input_row = QHBoxLayout()
        input_row.setSpacing(6)

        self.input_line = QLineEdit()
        self.input_line.setPlaceholderText('输入问题，按Enter发送...')
        self.input_line.setFixedHeight(36)
        self.input_line.setStyleSheet('''
            QLineEdit {
                background:#1e293b;border:1px solid #475569;
                border-radius:8px;color:#f1f5f9;
                font-size:13px;padding:0 10px;
            }
            QLineEdit:focus { border-color:#60a5fa; }
        ''')

        self.send_btn = QPushButton('发送')
        self.send_btn.setFixedHeight(36)
        self.send_btn.setFixedWidth(60)
        self.send_btn.setCursor(Qt.PointingHandCursor)

        input_row.addWidget(self.input_line)
        input_row.addWidget(self.send_btn)
        content_layout.addLayout(input_row)

        # ---- 状态栏 ----
        self.status_label = QLabel('AI助手就绪')
        self.status_label.setStyleSheet('font-size:11px;color:#64748b;padding:2px;')
        content_layout.addWidget(self.status_label)

        # ---- 连接信号 ----
        self.send_btn.clicked.connect(self.on_send)
        self.input_line.returnPressed.connect(self.on_send)
        self.config_btn.clicked.connect(self.on_config)
        self.btn_explain.clicked.connect(lambda: self.quick_send('解释结果'))
        self.btn_report.clicked.connect(lambda: self.quick_send('生成报告'))
        self.btn_advice.clicked.connect(lambda: self.quick_send('处置建议'))
        self.close_btn.clicked.connect(self._on_close)

        # 拖动状态
        self._drag_pos = None

    def _title_mouse_press(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_pos = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()

    def _title_mouse_move(self, event):
        if event.buttons() == Qt.LeftButton and self._drag_pos is not None:
            self.move(event.globalPos() - self._drag_pos)
            event.accept()

    def _title_mouse_release(self, event):
        self._drag_pos = None

    def _on_close(self):
        """关闭按钮 — 隐藏面板并通知主窗口显示浮动按钮"""
        self.hide()
        # 向上查找主窗口，调用toggle来同步状态并显示浮动按钮
        p = self.parent()
        while p:
            if hasattr(p, '_toggle_ai_panel'):
                p._toggle_ai_panel()
                break
            p = p.parent()

    def _apply_style(self):
        self.setStyleSheet('''
            AIChatWidget {
                background:#1e293b;
                border:2px solid #475569;
                border-radius:10px;
            }
        ''')

    def set_diagnosis_context(self, fault_name, confidence, total_count=0, normal_rate=0.0):
        """设置当前诊断上下文（由主窗口调用）"""
        self.diagnosis_context = {
            'fault_name': fault_name,
            'confidence': confidence,
            'time_str': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_count': total_count,
            'normal_rate': normal_rate,
        }
        self._add_system_message(f'已获取诊断上下文：{fault_name}（置信度{confidence:.1f}%）')

    def quick_send(self, template_name):
        """快捷指令发送"""
        if not self.diagnosis_context:
            self._add_system_message('⚠️ 请先进行一次诊断，再使用此功能')
            return

        prompt = self.llm.format_diagnosis_context(template_name, **self.diagnosis_context)
        self._send_to_llm(prompt, template_name)

    def on_send(self):
        """用户输入发送"""
        text = self.input_line.text().strip()
        if not text:
            return
        self.input_line.clear()

        if not self.llm.api_key:
            self._add_message('user', text)
            self._add_message('ai', '⚠️ API Key未配置，请点击右上角⚙️进行配置。')
            return

        self._add_message('user', text)

        # 如果有诊断上下文，自动附加上下文
        messages = self.llm.make_messages(
            '你是一位旋转机械轴承故障诊断专家，擅长用通俗易懂的语言解释技术问题。',
            text
        )
        self._send_messages(messages)

    def _send_to_llm(self, prompt, display_name=''):
        """发送Prompt到LLM"""
        self._add_message('ai_header', f'🤖 {display_name}...' if display_name else '🤖')

        messages = self.llm.make_messages(
            '你是一位资深的旋转机械故障诊断专家。',
            prompt
        )
        self._send_messages(messages)

    def _send_messages(self, messages):
        """发送消息列表到LLM"""
        if self.current_thread and self.current_thread.isRunning():
            self.current_thread.stop()
            self.current_thread.wait()

        self.send_btn.setEnabled(False)
        self.send_btn.setText('...')
        self.status_label.setText('🔄 AI思考中...')
        self.status_label.setStyleSheet('font-size:11px;color:#60a5fa;')

        self.current_thread = self.llm.chat(messages)
        self.current_thread.chunk_signal.connect(self._on_chunk)
        self.current_thread.done_signal.connect(self._on_done)
        self.current_thread.error_signal.connect(self._on_error)
        self.current_thread.start()

    def _on_chunk(self, text):
        """接收流式输出"""
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text)
        self.chat_display.setTextCursor(cursor)
        self.chat_display.ensureCursorVisible()

    def _on_done(self):
        self._finish_response()

    def _on_error(self, error_msg):
        self._add_message('ai', f'❌ 请求失败：{error_msg}\n\n请检查：\n1. API Key是否正确\n2. 网络连接是否正常\n3. API Base URL是否配置正确')
        self._finish_response()

    def _finish_response(self):
        self.send_btn.setEnabled(True)
        self.send_btn.setText('发送')
        self.status_label.setText('✅ 完成')
        self.status_label.setStyleSheet('font-size:11px;color:#22c55e;')
        self.chat_display.append('')  # 空行分隔

    def _add_message(self, role, text):
        """添加消息到对话区"""
        # 转义HTML特殊字符
        text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        # 保留换行
        text = text.replace('\n', '<br>')

        if role == 'user':
            # 用户消息：右对齐，蓝色气泡
            html = (
                '<table width="100%" cellspacing="0" cellpadding="0"><tr>'
                '<td align="right">'
                '<span style="background:#2563eb;color:#ffffff;padding:8px 12px;'
                'border-radius:12px 12px 2px 12px;font-size:13px;line-height:1.5;'
                'display:inline-block;max-width:260px;text-align:left;">'
                f'{text}</span></td></tr></table>'
            )
        elif role == 'ai_header':
            html = f'<div style="color:#94a3b8;font-size:11px;margin:8px 0 4px 4px;">🤖 {text}</div>'
        elif role == 'ai':
            # AI消息：左对齐，灰色气泡，白色文字确保对比度
            html = (
                '<table width="100%" cellspacing="0" cellpadding="0"><tr>'
                '<td align="left">'
                '<span style="background:#475569;color:#f8fafc;padding:8px 12px;'
                'border-radius:12px 12px 12px 2px;font-size:13px;line-height:1.5;'
                'display:inline-block;max-width:260px;text-align:left;">'
                f'{text}</span></td></tr></table>'
            )
        elif role == 'system':
            html = f'<div style="text-align:center;color:#94a3b8;font-size:11px;margin:4px 0;">{text}</div>'
        else:
            return

        self.chat_display.insertHtml(html)
        self.chat_display.insertHtml('<div></div>')  # 分隔
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.chat_display.setTextCursor(cursor)

    def _add_system_message(self, text):
        self._add_message('system', text)

    def on_config(self):
        """打开配置对话框"""
        dlg = AIConfigDialog(self.llm, self)
        if dlg.exec_() == QDialog.Accepted:
            self.llm.save_config()
            self.status_label.setText(f'✅ 已配置：{self.llm.model}')
            self.status_label.setStyleSheet('font-size:11px;color:#22c55e;')
class AIConfigDialog(QDialog):
    """AI助手配置对话框"""

    def __init__(self, llm_client, parent=None):
        super().__init__(parent)
        self.llm = llm_client
        self.setWindowTitle('AI助手配置')
        self.setMinimumWidth(480)
        self._build_ui()

    def _build_ui(self):
        layout = QFormLayout(self)
        layout.setSpacing(12)

        # 提供商选择
        self.provider_combo = QComboBox()
        self.provider_combo.addItems(list(LLMClient.PRESETS.keys()))
        self.provider_combo.currentTextChanged.connect(self.on_provider_changed)
        layout.addRow('API 提供商：', self.provider_combo)

        # API Key
        self.key_edit = QLineEdit()
        self.key_edit.setEchoMode(QLineEdit.Password)
        self.key_edit.setText(self.llm.api_key)
        self.key_edit.setPlaceholderText('sk-xxxxxxxx 或 your-api-key')
        layout.addRow('API Key：', self.key_edit)

        # Base URL
        self.url_edit = QLineEdit()
        self.url_edit.setText(self.llm.base_url)
        self.url_edit.setPlaceholderText('https://api.example.com/v1')
        layout.addRow('Base URL：', self.url_edit)

        # 模型名称
        self.model_edit = QLineEdit()
        self.model_edit.setText(self.llm.model)
        self.model_edit.setPlaceholderText('gpt-3.5-turbo')
        layout.addRow('模型名称：', self.model_edit)

        # 温度参数
        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setRange(0.0, 2.0)
        self.temp_spin.setSingleStep(0.1)
        self.temp_spin.setValue(self.llm.temperature)
        layout.addRow('温度参数：', self.temp_spin)

        # 说明文字
        note = QLabel(
            '💡 提示：\n'
            '• 推荐使用 DeepSeek（api.deepseek.com），国内访问快\n'
            '• 通义千问需前往 dashscope.aliyun.com 申请Key\n'
            '• 本地Ollama需先安装 ollama.com 并拉取模型\n'
            '• API Key仅存储在本地，不会上传到任何服务器'
        )
        note.setStyleSheet('color:#64748b;font-size:12px;line-height:1.6;')
        layout.addRow(note)

        # 按钮
        btn_row = QHBoxLayout()
        self.test_btn = QPushButton('🧪 测试连接')
        self.test_btn.clicked.connect(self.on_test)
        self.ok_btn = QPushButton('保存')
        self.ok_btn.clicked.connect(self.on_ok)
        self.cancel_btn = QPushButton('取消')
        self.cancel_btn.clicked.connect(self.reject)

        btn_row.addWidget(self.test_btn)
        btn_row.addStretch()
        btn_row.addWidget(self.ok_btn)
        btn_row.addWidget(self.cancel_btn)
        layout.addRow(btn_row)

        self.setStyleSheet('''
            QDialog { background:#1e293b; }
            QLabel { color:#e2e8f0;font-size:13px; }
            QLineEdit, QComboBox, QDoubleSpinBox {
                background:#0f172a;border:1px solid #475569;
                border-radius:6px;color:#f1f5f9;
                padding:6px;font-size:13px;
            }
            QLineEdit:focus, QComboBox:focus { border-color:#60a5fa; }
            QPushButton {
                background:#334155;border:1px solid #475569;
                border-radius:6px;color:#e2e8f0;
                padding:8px 16px;font-size:13px;
            }
            QPushButton:hover { background:#475569; }
            QPushButton#ok_btn { background:#2563eb;border-color:#3b82f6; }
            QPushButton#ok_btn:hover { background:#3b82f6; }
        ''')

    def on_provider_changed(self, name):
        if name in LLMClient.PRESETS and name != '自定义':
            preset = LLMClient.PRESETS[name]
            self.url_edit.setText(preset['base_url'])
            self.model_edit.setText(preset['model'])

    def on_test(self):
        """测试API连接"""
        self._save_to_llm()
        
        if not self.llm.api_key:
            QMessageBox.warning(self, '配置不完整', '请先填写API Key')
            return
        if not self.llm.base_url:
            QMessageBox.warning(self, '配置不完整', '请先填写Base URL')
            return
        
        self.test_btn.setEnabled(False)
        self.test_btn.setText('测试中...')

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say 'API test passed' only."}
        ]
        self.test_thread = self.llm.chat(messages)
        self.test_thread.chunk_signal.connect(lambda t: None)
        self.test_thread.done_signal.connect(lambda: self._test_result(True, ''))
        self.test_thread.error_signal.connect(lambda e: self._test_result(False, e))
        self.test_thread.start()

    def _test_result(self, success, error):
        self.test_btn.setEnabled(True)
        self.test_btn.setText('测试连接')
        if success:
            QMessageBox.information(self, '测试成功', '✅ API连接正常！')
        else:
            QMessageBox.warning(self, '测试失败', f'❌ {error}')

    def on_ok(self):
        self._save_to_llm()
        self.accept()

    def _save_to_llm(self):
        self.llm.api_key = self.key_edit.text().strip()
        self.llm.base_url = self.url_edit.text().strip()
        self.llm.model = self.model_edit.text().strip()
        self.llm.temperature = self.temp_spin.value()

# 4.浮动按钮
class AIFloatButton(QPushButton):
    """AI助手浮动按钮 — 右下角悬浮"""

    # 自定义信号：按钮被点击
    on_click = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__('🤖', parent)
        self.setFixedSize(50, 50)
        self.setStyleSheet('''
            QPushButton {
                background:#2563eb;border:2px solid #3b82f6;
                border-radius:25px;color:#fff;
                font-size:22px;
            }
            QPushButton:hover { background:#3b82f6; }
            QPushButton:pressed { background:#1d4ed8; }
        ''')
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip('AI智能助手 (点击展开)')

        super().clicked.connect(self._on_clicked)

    def _on_clicked(self):
        self.on_click.emit()
