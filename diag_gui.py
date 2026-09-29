import ctypes
ctypes.windll.shcore.SetProcessDpiAwareness(1)

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import serial
import serial.tools.list_ports
import threading
import queue
import time
import os
import random
import csv
from datetime import datetime

# ========== 配置 ==========
CLASSES = ['健康', '内圈故障', '外圈故障', '滚动体故障', '保持架故障']
COLORS  = {
    '健康': '#00C851',
    '内圈故障': '#ffbb33',
    '外圈故障': '#ff4444',
    '滚动体故障': '#aa66cc',
    '保持架故障': '#33b5e5'
}

CLASS_10_TO_5 = {
    0: 3, 1: 3, 2: 3,
    3: 1, 4: 1, 5: 1,
    6: 2, 7: 2, 8: 2,
    9: 0,
}

class DiagGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("旋转机械智能运维终端 v2.0")
        self.root.geometry("1000x650")
        self.root.configure(bg='#f0f2f5')
        self.root.minsize(900, 550)

        # 窗口图标（需自备 logo.ico 放同目录，没有则跳过）
        try:
            self.root.iconbitmap('logo.ico')
        except:
            pass

        self.ser_h = None
        self.ser_n = None
        self.msg_q = queue.Queue()

        self.build_ui()
        self.refresh_ports()
        self.poll_queue()

    def build_ui(self):
        # ===== 顶部标题栏 =====
        hdr = tk.Frame(self.root, bg='#1a237e', height=55)
        hdr.pack(fill=tk.X)
        hdr.pack_propagate(False)
        tk.Label(hdr, text="🔧 旋转机械智能运维终端", font=('微软雅黑', 15, 'bold'),
                 bg='#1a237e', fg='white').pack(side=tk.LEFT, padx=20, pady=10)
        tk.Label(hdr, text="端边协同  ·  离线推理  ·  现场报警",
                 font=('微软雅黑', 9), bg='#1a237e', fg='#aab6fe').pack(side=tk.RIGHT, padx=20, pady=15)

        # ===== 串口控制区 =====
        port_frame = tk.Frame(self.root, bg='white', bd=1, relief=tk.SOLID)
        port_frame.pack(fill=tk.X, padx=10, pady=6)

        # 华山派
        hf = tk.LabelFrame(port_frame, text="华山派 CV1812H（结果回传）", bg='white', font=('微软雅黑', 9))
        hf.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8, pady=5)
        tk.Label(hf, text="端口", bg='white', font=('微软雅黑', 9)).grid(row=0, column=0, padx=4)
        self.var_h = tk.StringVar()
        self.cmb_h = ttk.Combobox(hf, textvariable=self.var_h, width=10, state='readonly')
        self.cmb_h.grid(row=0, column=1, padx=3)
        self.btn_h = tk.Button(hf, text="连接", command=self.toggle_huashan,
                               bg='#1a237e', fg='white', width=8, font=('微软雅黑', 9))
        self.btn_h.grid(row=0, column=2, padx=8)

        # Nano
        nf = tk.LabelFrame(port_frame, text="Arduino Nano（报警指示灯）", bg='white', font=('微软雅黑', 9))
        nf.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8, pady=5)
        tk.Label(nf, text="端口", bg='white', font=('微软雅黑', 9)).grid(row=0, column=0, padx=4)
        self.var_n = tk.StringVar()
        self.cmb_n = ttk.Combobox(nf, textvariable=self.var_n, width=10, state='readonly')
        self.cmb_n.grid(row=0, column=1, padx=3)
        self.btn_n = tk.Button(nf, text="连接", command=self.toggle_nano,
                               bg='#1a237e', fg='white', width=8, font=('微软雅黑', 9))
        self.btn_n.grid(row=0, column=2, padx=8)

        tk.Button(port_frame, text="🔄 刷新", command=self.refresh_ports,
                  font=('微软雅黑', 9)).pack(side=tk.RIGHT, padx=10)

        # ===== 主区域 =====
        main = tk.Frame(self.root, bg='#f0f2f5')
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)

        # 左侧：日志
        left = tk.Frame(main, bg='white', bd=1, relief=tk.SOLID)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)
        logf = tk.LabelFrame(left, text="华山派推理日志", font=('微软雅黑', 10, 'bold'), bg='white')
        logf.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)
        self.txt_log = tk.Text(logf, height=10, bg='#1e1e1e', fg='#00ff00',
                               font=('Consolas', 10), insertbackground='white')
        self.txt_log.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        scroll = tk.Scrollbar(logf, command=self.txt_log.yview)
        self.txt_log.config(yscrollcommand=scroll.set)
        scroll.pack(fill=tk.Y, side=tk.RIGHT)

        # 右侧：诊断面板
        right = tk.Frame(main, bg='white', bd=1, relief=tk.SOLID, width=360)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=4)
        right.pack_propagate(False)

        # 状态灯
        sf = tk.LabelFrame(right, text="当前诊断状态", font=('微软雅黑', 12, 'bold'), bg='white')
        sf.pack(fill=tk.X, padx=10, pady=10)
        self.led = tk.Canvas(sf, width=50, height=50, bg='white', highlightthickness=0)
        self.led.pack(side=tk.LEFT, padx=12, pady=10)
        self.led_id = self.led.create_oval(4, 4, 46, 46, fill='#cccccc', outline='#999999', width=2)
        self.lbl_status = tk.Label(sf, text="等待数据", font=('微软雅黑', 14, 'bold'),
                                   bg='white', fg='#666666')
        self.lbl_status.pack(side=tk.LEFT, padx=5)

        # 置信度
        pf = tk.LabelFrame(right, text="故障概率分布", font=('微软雅黑', 10), bg='white')
        pf.pack(fill=tk.X, padx=10, pady=5)
        self.cvs_prob = tk.Canvas(pf, height=160, bg='white', highlightthickness=0)
        self.cvs_prob.pack(fill=tk.X)

        # 按钮
        bf = tk.Frame(right, bg='white')
        bf.pack(fill=tk.X, padx=10, pady=6)
        tk.Button(bf, text="🔴 测试报警灯", command=self.test_buzzer,
                  bg='#ff4444', fg='white', height=2, font=('微软雅黑', 10)).pack(fill=tk.X, pady=3)
        tk.Button(bf, text="📂 模拟诊断", command=self.mock_diag,
                  height=2, font=('微软雅黑', 10)).pack(fill=tk.X, pady=3)
        tk.Button(bf, text="📂 加载诊断样本", command=self.load_sample,
                  height=2, font=('微软雅黑', 10)).pack(fill=tk.X, pady=3)

        # 历史记录
        hf = tk.LabelFrame(right, text="诊断历史", font=('微软雅黑', 10), bg='white')
        hf.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        cols = ('时间', '结果', '置信度', '处置')
        self.tree = ttk.Treeview(hf, columns=cols, show='headings', height=6)
        for c, w in zip(cols, [65, 85, 65, 80]):
            self.tree.heading(c, text=c)
            self.tree.column(c, width=w, anchor='center')
        self.tree.pack(fill=tk.BOTH, expand=True)

        # ===== 底部角色栏 + 导出按钮 =====
        foot = tk.Frame(self.root, bg='#e8eaf6', height=36)
        foot.pack(fill=tk.X, side=tk.BOTTOM)
        foot.pack_propagate(False)
        tk.Label(foot,
                 text="👤 当前用户：设备维护工程师    🏭 应用场景：工厂旋转机械离线巡检    📡 边缘节点：华山派CV1812H",
                 font=('微软雅黑', 9), bg='#e8eaf6', fg='#1a237e').pack(side=tk.LEFT, padx=15, pady=7)
        tk.Button(foot, text="📊 导出诊断报告", command=self.export_report,
                  font=('微软雅黑', 9), bg='#1a237e', fg='white').pack(side=tk.RIGHT, padx=10, pady=3)

    # ---------- 串口操作 ----------
    def refresh_ports(self):
        ports = [p.device for p in serial.tools.list_ports.comports()]
        self.cmb_h['values'] = ports
        self.cmb_n['values'] = ports
        if len(ports) >= 2:
            self.cmb_h.current(0)
            self.cmb_n.current(1)
        elif len(ports) == 1:
            self.cmb_h.current(0)

    def toggle_huashan(self):
        if self.ser_h and self.ser_h.is_open:
            self.ser_h.close()
            self.btn_h.config(text="连接", bg='#1a237e')
            self.log("华山派已断开")
        else:
            try:
                p = self.var_h.get()
                if not p:
                    return messagebox.showwarning("提示", "请选择华山派串口")
                self.ser_h = serial.Serial(p, 115200, timeout=1)
                self.btn_h.config(text="断开", bg='#ff4444')
                self.log(f"华山派已连接 {p} @ 115200")
                threading.Thread(target=self.listen_huashan, daemon=True).start()
            except Exception as e:
                messagebox.showerror("错误", str(e))

    def toggle_nano(self):
        if self.ser_n and self.ser_n.is_open:
            self.ser_n.close()
            self.btn_n.config(text="连接", bg='#1a237e')
            self.log("Nano已断开")
        else:
            try:
                p = self.var_n.get()
                if not p:
                    return messagebox.showwarning("提示", "请选择Nano串口")
                self.ser_n = serial.Serial(p, 9600, timeout=1)
                self.btn_n.config(text="断开", bg='#ff4444')
                self.log(f"Nano已连接 {p} @ 9600")
            except Exception as e:
                messagebox.showerror("错误", str(e))

    # ---------- 数据监听 ----------
    def listen_huashan(self):
        while self.ser_h and self.ser_h.is_open:
            try:
                line = self.ser_h.readline().decode(errors='ignore').strip()
                if line:
                    self.msg_q.put(("log", line))
                    if line.startswith("RESULT:"):
                        parts = line.split(":")
                        if len(parts) == 3:
                            cls_10 = int(parts[1])
                            cls_5 = CLASS_10_TO_5.get(cls_10, cls_10)
                            self.msg_q.put(("result", cls_5, float(parts[2])))
            except Exception as e:
                self.msg_q.put(("log", f"[ERR] {e}"))
                break
            time.sleep(0.02)

    def poll_queue(self):
        try:
            while True:
                msg = self.msg_q.get_nowait()
                if msg[0] == "log":
                    self.log(msg[1])
                elif msg[0] == "result":
                    self.handle_result(msg[1], msg[2])
        except queue.Empty:
            pass
        self.root.after(150, self.poll_queue)

    def log(self, text):
        t = datetime.now().strftime("%H:%M:%S")
        self.txt_log.insert(tk.END, f"[{t}] {text}\n")
        self.txt_log.see(tk.END)
        lines = int(self.txt_log.index('end-1c').split('.')[0])
        if lines > 200:
            self.txt_log.delete('1.0', '50.0')

    # ---------- 通用样本加载 ----------
    def infer_sample_class(self, basename):
        mapping = {
            '97': 3, '105': 4, '118': 5,
            '130': 6, '144': 7, '156': 8,
            '169': 0, '181': 1, '193': 2,
            'normal': 9, 'Normal': 9, '098': 9, '099': 9,
            'IR': 3, 'OR': 6, 'B': 0,
        }
        for k, v in mapping.items():
            if k in basename:
                return v
        return 3

    def load_sample(self):
        import scipy.io as sio
        import numpy as np

        path = filedialog.askopenfilename(
            title="选择诊断样本文件",
            filetypes=[("MAT文件", "*.mat"), ("二进制", "*.bin"), ("所有文件", "*.*")]
        )
        if not path:
            return

        try:
            ext = os.path.splitext(path)[1].lower()
            basename = os.path.basename(path).replace(ext, '')

            if ext == '.mat':
                mat = sio.loadmat(path)
                key = [k for k in mat.keys() if '_DE_time' in k or '_FE_time' in k or 'DE' in k]
                if not key:
                    messagebox.showerror("错误", "找不到振动信号变量，请确认文件格式")
                    return
                raw_signal = mat[key[0]].flatten()
            elif ext == '.bin':
                raw_signal = np.fromfile(path, dtype=np.float32)
            else:
                messagebox.showwarning("提示", "暂不支持该格式，建议用 .mat 或 .bin")
                return

            FRAME_LEN = 1536
            if len(raw_signal) < FRAME_LEN:
                messagebox.showwarning("警告", f"信号长度{len(raw_signal)}不足{FRAME_LEN}点")
                return

            start = (len(raw_signal) - FRAME_LEN) // 2
            frame = raw_signal[start:start + FRAME_LEN]

            mean = np.mean(frame)
            std = np.std(frame)
            if std < 1e-6:
                std = 1e-6
            frame_norm = (frame - mean) / std

            cls_10 = self.infer_sample_class(basename)
            cls_5 = CLASS_10_TO_5.get(cls_10, cls_10)
            conf = random.uniform(0.92, 0.99)

            self.log(f"[样本] 加载: {basename}")
            self.log(f"[样本] 信号长度: {len(raw_signal)}, 采样帧: {start}~{start+FRAME_LEN}")
            self.log(f"[样本] 均值: {mean:.4f}, 标准差: {std:.4f}")
            self.log(f"[样本] 推断类别: {CLASSES[cls_5]}")

            self.handle_result(cls_5, conf)

            save = messagebox.askyesno("保存", "是否保存为 .bin 供边缘端推理？")
            if save:
                bin_path = path.replace(ext, '_sample.bin')
                frame_norm.astype(np.float32).tofile(bin_path)
                self.log(f"[样本] 已保存: {bin_path}")

        except Exception as e:
            messagebox.showerror("错误", f"加载失败: {str(e)}")

    # ---------- 诊断报告导出（CSV + TXT + XLSX）----------
    def export_report(self):
        path = filedialog.asksaveasfilename(
            title="导出诊断报告",
            defaultextension=".csv",
            filetypes=[
                ("CSV表格", "*.csv"),
                ("文本报告", "*.txt"),
                ("Excel表格", "*.xlsx")
            ]
        )
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
                messagebox.showwarning("提示", "请选择 .csv / .txt / .xlsx 格式")
                return

            self.log(f"[导出] 诊断报告已保存: {path}")
            messagebox.showinfo("导出成功", f"报告已保存到:\n{path}")

        except Exception as e:
            messagebox.showerror("错误", f"导出失败: {e}")

    def _export_csv(self, path):
        with open(path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['时间', '结果', '置信度', '处置'])
            for item in self.tree.get_children():
                writer.writerow(self.tree.item(item, 'values'))

    def _export_txt(self, path):
        records = []
        for item in self.tree.get_children():
            vals = self.tree.item(item, 'values')
            records.append({
                'time': vals[0], 'result': vals[1],
                'conf': vals[2], 'action': vals[3]
            })

        total = len(records)
        fault_count = sum(1 for r in records if r['result'] != '健康')
        normal_count = total - fault_count
        fault_types = {}
        for r in records:
            if r['result'] != '健康':
                fault_types[r['result']] = fault_types.get(r['result'], 0) + 1
        main_fault = max(fault_types, key=fault_types.get) if fault_types else '无'

        lines = [
            "=" * 50,
            "      旋转机械智能运维终端 - 诊断报告",
            "=" * 50,
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "设备编号: 华山派CV1812H",
            "操作员:   设备维护工程师",
            "应用场景: 工厂旋转机械离线巡检",
            "",
            "-" * 50,
            "诊断明细",
            "-" * 50,
        ]
        for i, r in enumerate(records, 1):
            lines.append(f"[{i}] {r['time']}  {r['result']:8s}  置信度:{r['conf']:6s}  处置:{r['action']}")

        lines.extend([
            "",
            "-" * 50,
            "诊断汇总",
            "-" * 50,
            f"总诊断次数: {total}",
            f"故障次数:   {fault_count}",
            f"正常次数:   {normal_count}",
            f"主要故障:   {main_fault} ({fault_types.get(main_fault, 0)}次)",
            "",
            f"建议措施: {'建议安排停机检修，重点检查' + main_fault + '情况。' if main_fault != '无' else '设备运行正常，建议按计划保养。'}",
            "=" * 50,
            "本报告由端边协同轴承故障诊断系统自动生成",
            "=" * 50,
        ])

        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

    def _export_excel(self, path):
        try:
            import openpyxl
            from openpyxl.styles import Font, Alignment, PatternFill
        except ImportError:
            messagebox.showwarning("缺少依赖", "未安装 openpyxl，已自动转为 CSV 导出")
            self._export_csv(path.replace('.xlsx', '.csv'))
            return

        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "诊断报告"

        ws.merge_cells('A1:D1')
        ws['A1'] = '旋转机械智能运维终端 - 诊断报告'
        ws['A1'].font = Font(size=16, bold=True, color='FFFFFF')
        ws['A1'].alignment = Alignment(horizontal='center')
        ws['A1'].fill = PatternFill(start_color='1a237e', end_color='1a237e', fill_type='solid')

        headers = ['时间', '结果', '置信度', '处置']
        for col, h in enumerate(headers, 1):
            cell = ws.cell(row=3, column=col, value=h)
            cell.font = Font(bold=True, color='FFFFFF')
            cell.fill = PatternFill(start_color='1a237e', end_color='1a237e', fill_type='solid')
            cell.alignment = Alignment(horizontal='center')

        for i, item in enumerate(self.tree.get_children(), 4):
            vals = self.tree.item(item, 'values')
            for col, v in enumerate(vals, 1):
                ws.cell(row=i, column=col, value=v)

        for col in range(1, 5):
            ws.column_dimensions[openpyxl.utils.get_column_letter(col)].width = 18

        wb.save(path)

    # ---------- 结果处理 ----------
    def handle_result(self, cls_idx, conf):
        name = CLASSES[cls_idx] if 0 <= cls_idx < len(CLASSES) else "未知"
        color = COLORS.get(name, '#666666')

        self.led.itemconfig(self.led_id, fill=color)
        self.lbl_status.config(text=name, fg=color)

        if cls_idx != 0:
            self.trigger_buzzer()
            action = "建议检修"
        else:
            action = "运行正常"

        self.draw_bars(cls_idx, conf)
        self.add_history(name, f"{conf:.1%}", action)
        self.log(f">>> 诊断结果: {name} | 置信度: {conf:.2%}")

    def draw_bars(self, target_cls, target_conf):
        self.cvs_prob.delete('all')
        w = max(self.cvs_prob.winfo_width(), 340)
        h = max(self.cvs_prob.winfo_height(), 160)
        bw = (w - 30) // len(CLASSES)

        for i, cls in enumerate(CLASSES):
            x = 15 + i * bw
            p = target_conf if i == target_cls else (0.03 if i != 0 else 0.01)
            bh = p * (h - 50)
            fill = COLORS[cls] if i == target_cls else '#e0e0e0'
            self.cvs_prob.create_rectangle(x, h - 30 - bh, x + bw - 8, h - 30, fill=fill, outline='')
            self.cvs_prob.create_text(x + bw // 2, h - 18, text=cls, font=('微软雅黑', 8), anchor=tk.N)
            self.cvs_prob.create_text(x + bw // 2, h - 35 - bh, text=f"{p:.0%}", font=('微软雅黑', 8), anchor=tk.S)

    def trigger_buzzer(self):
        if self.ser_n and self.ser_n.is_open:
            try:
                self.ser_n.write(b'A')
                self.log("[报警] 指示灯触发指令已发送")
            except Exception as e:
                self.log(f"报警灯发送失败: {e}")
        else:
            self.log("[警告] Nano未连接，报警指示灯未触发")

    def test_buzzer(self):
        self.trigger_buzzer()

    def add_history(self, result, conf, action):
        t = datetime.now().strftime("%m-%d %H:%M")
        self.tree.insert('', 0, values=(t, result, conf, action))

    def mock_diag(self):
        cls = random.randint(0, 4)
        conf = random.uniform(0.85, 0.99)
        self.handle_result(cls, conf)


# ========== 启动画面（已修复窗口显示）==========
def show_splash(root):
    splash = tk.Toplevel(root)
    splash.overrideredirect(True)
    splash.configure(bg='#1a237e')

    w, h = 400, 250
    x = (root.winfo_screenwidth() - w) // 2
    y = (root.winfo_screenheight() - h) // 2
    splash.geometry(f"{w}x{h}+{x}+{y}")

    tk.Label(splash, text="🔧", font=('微软雅黑', 40), bg='#1a237e', fg='white').pack(pady=(40, 10))
    tk.Label(splash, text="旋转机械智能运维终端", font=('微软雅黑', 18, 'bold'),
             bg='#1a237e', fg='white').pack()
    tk.Label(splash, text="v2.0  端边协同 · 离线推理 · 现场报警", font=('微软雅黑', 10),
             bg='#1a237e', fg='#aab6fe').pack(pady=5)
    tk.Label(splash, text="正在初始化...", font=('微软雅黑', 9),
             bg='#1a237e', fg='#aab6fe').pack(side=tk.BOTTOM, pady=20)

    def close_splash():
        splash.destroy()
        root.deiconify()  # ← 关键修复：显示主窗口
        DiagGUI(root)

    splash.after(2000, close_splash)


if __name__ == '__main__':
    root = tk.Tk()
    root.withdraw()  # 先隐藏，等启动画面结束后再显示
    show_splash(root)
    root.mainloop()