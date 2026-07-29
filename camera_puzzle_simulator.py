#!/usr/bin/env python3
"""Live OpenMV puzzle restoration simulator.

The application consumes newline-delimited JSON emitted by
``polygon_detection.py``.  Camera polygons update continuously until SET is
pressed; SET freezes that exact snapshot, solves the puzzle and animates the
computed piece motions into the lower half of the camera frame.
"""

from __future__ import annotations

from glob import glob
from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import puzzle_restoration as restoration


try:
    import serial as serial_module
    from serial.tools import list_ports
except ImportError:
    serial_module = None
    list_ports = None


PIECE_COLORS = (
    "#ef8354",
    "#4f8cc9",
    "#59a96a",
    "#a06cd5",
)
TARGET_COLORS = (
    "#a9441f",
    "#235f97",
    "#24783a",
    "#6d399c",
)
MODE_FIXED_QUESTION_ONE = "第一题：固定4片（10cm×6cm）"
MODE_GENERAL = "第二题：通用1～4片"


class CameraPuzzleSimulator:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("OpenMV 拼图自动归位模拟器")
        self.root.geometry("1320x840")
        self.root.minsize(1080, 690)

        self.status = tk.StringVar(value="等待 OpenMV 数据")
        self.data_status = tk.StringVar(value="LIVE · 尚未收到有效碎片")
        self.serial_hint = tk.StringVar(
            value="OpenMV 在 macOS 上通常显示为 /dev/cu.usbmodem…"
        )
        self.port_var = tk.StringVar()
        self.baud_var = tk.StringVar(value="115200")
        self.frame_width_var = tk.StringVar(value="640")
        self.frame_height_var = tk.StringVar(value="480")
        self.divider_y_var = tk.StringVar(value="240")
        self.clearance_cm_var = tk.StringVar(value="0.5")
        self.speed_var = tk.DoubleVar(value=1.0)
        self.mode_var = tk.StringVar(value=MODE_FIXED_QUESTION_ONE)

        self.latest_payload = None
        self.live_pieces: tuple[restoration.Piece, ...] = ()
        self.frozen_pieces: tuple[restoration.Piece, ...] = ()
        self.solution: restoration.PuzzleSolution | None = None
        self.locked = False
        self.solving = False

        self.serial_port = None
        self.serial_thread = None
        self.serial_stop = threading.Event()
        self.event_queue: queue.Queue = queue.Queue()

        self.animation_after_id = None
        self.animation_index = 0
        self.animation_fraction = 0.0
        self.animation_progress: dict[int, float] = {}

        self._build_style()
        self._build_layout()
        self._refresh_ports()
        self.root.after(40, self._poll_events)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_style(self):
        style = ttk.Style()
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("Title.TLabel", font=("Helvetica", 18, "bold"))
        style.configure(
            "Live.TLabel",
            font=("Helvetica", 11, "bold"),
            foreground="#176b3a",
        )
        style.configure(
            "Set.TButton",
            font=("Helvetica", 13, "bold"),
            padding=(15, 11),
        )
        style.configure("Action.TButton", padding=(10, 7))
        style.configure("Treeview", rowheight=25)

    def _build_layout(self):
        header = ttk.Frame(self.root, padding=(16, 12))
        header.pack(fill="x")
        ttk.Label(
            header,
            text="OpenMV 拼图自动归位模拟器",
            style="Title.TLabel",
        ).pack(side="left")
        ttk.Label(
            header,
            textvariable=self.data_status,
            style="Live.TLabel",
        ).pack(side="right", pady=(6, 0))

        body = ttk.Panedwindow(self.root, orient="horizontal")
        body.pack(fill="both", expand=True, padx=14, pady=(0, 10))

        canvas_frame = ttk.Frame(body)
        side = ttk.Frame(body, width=430)
        body.add(canvas_frame, weight=4)
        body.add(side, weight=2)

        self.canvas = tk.Canvas(
            canvas_frame,
            bg="#20252b",
            highlightthickness=0,
        )
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda _event: self._draw_scene())

        self._build_input_panel(side)
        self._build_action_panel(side)
        self._build_result_panel(side)

        footer = ttk.Frame(self.root, padding=(16, 7))
        footer.pack(fill="x")
        ttk.Label(footer, textvariable=self.status).pack(side="left")
        ttk.Label(
            footer,
            text="坐标：原点左上 · X 向右 · Y 向下 · 单位 px",
            foreground="#666",
        ).pack(side="right")

    def _build_input_panel(self, parent):
        panel = ttk.LabelFrame(parent, text="OpenMV 实时输入", padding=10)
        panel.pack(fill="x", padx=(10, 0))

        port_row = ttk.Frame(panel)
        port_row.pack(fill="x", pady=(0, 7))
        ttk.Label(port_row, text="串口").pack(side="left")
        self.port_combo = ttk.Combobox(
            port_row,
            textvariable=self.port_var,
            width=24,
        )
        self.port_combo.pack(side="left", fill="x", expand=True, padx=(8, 5))
        ttk.Button(
            port_row,
            text="刷新",
            command=self._refresh_ports,
        ).pack(side="right")

        connection_row = ttk.Frame(panel)
        connection_row.pack(fill="x")
        ttk.Label(connection_row, text="波特率").pack(side="left")
        ttk.Combobox(
            connection_row,
            textvariable=self.baud_var,
            values=("115200", "230400", "460800", "921600"),
            state="readonly",
            width=10,
        ).pack(side="left", padx=(8, 8))
        self.connect_button = ttk.Button(
            connection_row,
            text="连接",
            command=self._toggle_connection,
            style="Action.TButton",
        )
        self.connect_button.pack(side="left", fill="x", expand=True)

        import_row = ttk.Frame(panel)
        import_row.pack(fill="x", pady=(7, 0))
        ttk.Button(
            import_row,
            text="从文件载入 JSON",
            command=self._load_json_file,
            style="Action.TButton",
        ).pack(side="left", fill="x", expand=True, padx=(0, 3))
        ttk.Button(
            import_row,
            text="粘贴输出 JSON",
            command=self._open_paste_dialog,
            style="Action.TButton",
        ).pack(side="left", fill="x", expand=True, padx=(3, 0))

        ttk.Label(
            panel,
            textvariable=self.serial_hint,
            foreground="#59636b",
            wraplength=390,
        ).pack(anchor="w", pady=(7, 0))

        if serial_module is None:
            ttk.Label(
                panel,
                text="未安装 pyserial；可先使用“载入 JSON”测试",
                foreground="#a55b12",
            ).pack(anchor="w", pady=(7, 0))

    def _build_action_panel(self, parent):
        panel = ttk.LabelFrame(parent, text="快照与动画", padding=10)
        panel.pack(fill="x", padx=(10, 0), pady=(10, 0))

        mode_row = ttk.Frame(panel)
        mode_row.pack(fill="x", pady=(0, 9))
        ttk.Label(mode_row, text="题型").pack(side="left")
        ttk.Combobox(
            mode_row,
            textvariable=self.mode_var,
            values=(MODE_FIXED_QUESTION_ONE, MODE_GENERAL),
            state="readonly",
        ).pack(side="left", fill="x", expand=True, padx=(8, 0))

        geometry_row = ttk.Frame(panel)
        geometry_row.pack(fill="x", pady=(0, 9))
        for label, variable in (
            ("宽", self.frame_width_var),
            ("高", self.frame_height_var),
            ("分界 Y", self.divider_y_var),
            ("第一题间隙 cm", self.clearance_cm_var),
        ):
            group = ttk.Frame(geometry_row)
            group.pack(side="left", fill="x", expand=True, padx=3)
            ttk.Label(group, text=label).pack(anchor="w")
            entry = ttk.Entry(group, textvariable=variable, width=7)
            entry.pack(fill="x")
            entry.bind("<Return>", lambda _event: self._draw_scene())

        self.set_button = ttk.Button(
            panel,
            text="SET · 冻结当前值并计算",
            command=self._freeze_and_solve,
            style="Set.TButton",
        )
        self.set_button.pack(fill="x")

        button_row = ttk.Frame(panel)
        button_row.pack(fill="x", pady=(7, 0))
        self.replay_button = ttk.Button(
            button_row,
            text="重播归位",
            command=self._start_animation,
            state="disabled",
            style="Action.TButton",
        )
        self.replay_button.pack(side="left", fill="x", expand=True, padx=(0, 3))
        ttk.Button(
            button_row,
            text="返回 LIVE",
            command=self._return_to_live,
            style="Action.TButton",
        ).pack(side="left", fill="x", expand=True, padx=(3, 0))

        speed_row = ttk.Frame(panel)
        speed_row.pack(fill="x", pady=(7, 0))
        ttk.Label(speed_row, text="动画速度").pack(side="left")
        ttk.Scale(
            speed_row,
            variable=self.speed_var,
            from_=0.4,
            to=3.0,
        ).pack(side="left", fill="x", expand=True, padx=(8, 0))

    def _build_result_panel(self, parent):
        panel = ttk.LabelFrame(parent, text="计算得到的归位方法", padding=8)
        panel.pack(fill="both", expand=True, padx=(10, 0), pady=(10, 0))

        columns = ("order", "piece", "pick", "place", "angle")
        self.motion_tree = ttk.Treeview(
            panel,
            columns=columns,
            show="headings",
            height=6,
        )
        for column, title, width in (
            ("order", "顺序", 46),
            ("piece", "碎片", 50),
            ("pick", "抓取中心", 105),
            ("place", "目标中心", 105),
            ("angle", "旋转", 65),
        ):
            self.motion_tree.heading(column, text=title)
            self.motion_tree.column(column, width=width, anchor="center")
        self.motion_tree.pack(fill="x")

        self.details = tk.Text(
            panel,
            wrap="none",
            height=14,
            font=("Menlo", 9),
            bg="#f7f8fa",
            relief="flat",
        )
        detail_scroll = ttk.Scrollbar(
            panel,
            orient="vertical",
            command=self.details.yview,
        )
        self.details.configure(yscrollcommand=detail_scroll.set)
        self.details.pack(side="left", fill="both", expand=True, pady=(8, 0))
        detail_scroll.pack(side="right", fill="y", pady=(8, 0))

    def _frame_settings(self):
        try:
            width = int(self.frame_width_var.get())
            height = int(self.frame_height_var.get())
            divider = float(self.divider_y_var.get())
        except ValueError as error:
            raise restoration.PuzzleSolveError(
                "画面宽、高和分界线必须是数字"
            ) from error
        if width <= 0 or height <= 0 or not 0 < divider < height:
            raise restoration.PuzzleSolveError(
                "画面尺寸必须为正，分界 Y 必须位于画面内部"
            )
        return (width, height), divider

    def _clearance_setting(self):
        try:
            clearance_cm = float(self.clearance_cm_var.get())
        except ValueError as error:
            raise restoration.PuzzleSolveError(
                "第一题间隙必须是数字"
            ) from error
        if not 0.0 <= clearance_cm <= 2.0:
            raise restoration.PuzzleSolveError(
                "第一题间隙必须在 0～2 cm 之间"
            )
        return clearance_cm

    def _refresh_ports(self):
        devices = []
        if list_ports is not None:
            devices.extend(port.device for port in list_ports.comports())
        # Some macOS/PySerial combinations omit a USB CDC device from
        # list_ports even though its /dev node is available.
        for pattern in (
            "/dev/cu.usbmodem*",
            "/dev/cu.usbserial*",
            "/dev/tty.usbmodem*",
            "/dev/tty.usbserial*",
        ):
            devices.extend(glob(pattern))
        devices = sorted(
            set(devices),
            key=lambda device: (
                0
                if (
                    "usbmodem" in device.lower()
                    or "usbserial" in device.lower()
                    or "openmv" in device.lower()
                )
                else 1,
                device,
            ),
        )
        self.port_combo["values"] = devices
        if devices and self.port_var.get() not in devices:
            self.port_var.set(devices[0])
        likely_openmv = [
            device
            for device in devices
            if (
                "usbmodem" in device.lower()
                or "usbserial" in device.lower()
                or "openmv" in device.lower()
            )
        ]
        if likely_openmv:
            self.serial_hint.set(
                "发现可能的 OpenMV 串口：%s" % likely_openmv[0]
            )
        elif devices:
            self.serial_hint.set(
                "未发现 USB 串口；/dev/cu.debug-console 不是 OpenMV。"
                "可粘贴终端输出 JSON。"
            )
        else:
            self.serial_hint.set(
                "未发现串口。请重新插拔 OpenMV、关闭 IDE 串口终端后刷新，"
                "或直接粘贴输出 JSON。"
            )

    def _toggle_connection(self):
        if self.serial_port is not None:
            self._disconnect_serial()
            return
        if serial_module is None:
            messagebox.showinfo(
                "需要 pyserial",
                "请先执行：\npython3 -m pip install pyserial",
            )
            return
        port = self.port_var.get().strip()
        if not port:
            messagebox.showwarning("没有串口", "请选择 OpenMV 串口")
            return
        if port.endswith("debug-console"):
            messagebox.showwarning(
                "这不是 OpenMV 串口",
                "/dev/cu.debug-console 是 macOS 调试控制台，不能读取 "
                "OpenMV 输出。\n\n请寻找 /dev/cu.usbmodem…，或使用"
                "“粘贴输出 JSON”。",
            )
            return
        try:
            baudrate = int(self.baud_var.get())
            self.serial_port = serial_module.Serial(
                port=port,
                baudrate=baudrate,
                timeout=0.20,
            )
        except Exception as error:
            self.serial_port = None
            messagebox.showerror(
                "连接失败",
                str(error)
                + "\n\n如果提示设备忙，请先停止 OpenMV IDE 的串口终端，"
                "再重新点击“刷新”。也可以直接粘贴 IDE 中复制的 JSON。",
            )
            return
        self.serial_stop.clear()
        self.serial_thread = threading.Thread(
            target=self._serial_reader,
            daemon=True,
        )
        self.serial_thread.start()
        self.connect_button.configure(text="断开")
        self.status.set("已连接 %s，等待 OpenMV JSON" % port)

    def _disconnect_serial(self):
        self.serial_stop.set()
        serial_port = self.serial_port
        self.serial_port = None
        if serial_port is not None:
            try:
                serial_port.close()
            except Exception:
                pass
        self.connect_button.configure(text="连接")
        self.status.set("串口已断开")

    def _serial_reader(self):
        while not self.serial_stop.is_set():
            serial_port = self.serial_port
            if serial_port is None:
                break
            try:
                raw_line = serial_port.readline()
            except Exception as error:
                if not self.serial_stop.is_set():
                    self.event_queue.put(("serial_error", str(error)))
                break
            if not raw_line:
                continue
            text = raw_line.decode("utf-8", errors="replace").strip()
            if "{" not in text:
                continue
            try:
                payload = restoration.payload_from_json_line(text)
            except restoration.PuzzleSolveError:
                continue
            self.event_queue.put(("payload", payload))

    def _load_json_file(self):
        filename = filedialog.askopenfilename(
            title="选择 OpenMV JSON 或串口日志",
            filetypes=(
                ("JSON / log", "*.json *.log *.txt"),
                ("All files", "*"),
            ),
        )
        if not filename:
            return
        try:
            text = Path(filename).read_text(encoding="utf-8")
            payload = restoration.payload_from_text(text)
            restoration.pieces_from_payload(payload)
            self._accept_payload(payload)
            self.status.set("已载入：%s" % filename)
        except Exception as error:
            messagebox.showerror("载入失败", str(error))

    def _open_paste_dialog(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("粘贴 OpenMV 输出 JSON")
        dialog.geometry("760x520")
        dialog.minsize(590, 390)
        dialog.transient(self.root)

        header = ttk.Frame(dialog, padding=(14, 12, 14, 7))
        header.pack(fill="x")
        ttk.Label(
            header,
            text="粘贴完整 JSON，或包含启动日志和多条 JSON 的终端输出。",
            font=("Helvetica", 11, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            header,
            text="程序会自动选取最后一条完整且有效的摄像头数据。",
            foreground="#5e6870",
        ).pack(anchor="w", pady=(4, 0))

        editor_frame = ttk.Frame(dialog, padding=(14, 0, 14, 8))
        editor_frame.pack(fill="both", expand=True)
        editor = tk.Text(
            editor_frame,
            wrap="none",
            undo=True,
            font=("Menlo", 10),
            bg="#f7f8fa",
        )
        vertical_scroll = ttk.Scrollbar(
            editor_frame, orient="vertical", command=editor.yview
        )
        horizontal_scroll = ttk.Scrollbar(
            editor_frame, orient="horizontal", command=editor.xview
        )
        editor.configure(
            yscrollcommand=vertical_scroll.set,
            xscrollcommand=horizontal_scroll.set,
        )
        editor.grid(row=0, column=0, sticky="nsew")
        vertical_scroll.grid(row=0, column=1, sticky="ns")
        horizontal_scroll.grid(row=1, column=0, sticky="ew")
        editor_frame.rowconfigure(0, weight=1)
        editor_frame.columnconfigure(0, weight=1)

        def paste_clipboard(replace=False):
            try:
                clipboard_text = self.root.clipboard_get()
            except tk.TclError:
                messagebox.showinfo(
                    "剪贴板为空",
                    "没有读取到文本，请在输入框中按 Command+V/Control+V。",
                    parent=dialog,
                )
                return
            if replace:
                editor.delete("1.0", "end")
            editor.insert("insert", clipboard_text)
            editor.focus_set()

        def import_pasted_text(_event=None):
            try:
                payload = restoration.payload_from_text(
                    editor.get("1.0", "end")
                )
                pieces = restoration.pieces_from_payload(payload)
            except Exception as error:
                messagebox.showerror(
                    "JSON 无法采用",
                    str(error),
                    parent=dialog,
                )
                return "break"
            self._accept_payload(payload)
            self.status.set(
                "已从粘贴内容导入 %d 块碎片；可点击 SET" % len(pieces)
            )
            dialog.destroy()
            return "break"

        buttons = ttk.Frame(dialog, padding=(14, 0, 14, 12))
        buttons.pack(fill="x")
        ttk.Button(
            buttons,
            text="清空",
            command=lambda: editor.delete("1.0", "end"),
        ).pack(side="left")
        ttk.Button(
            buttons,
            text="从剪贴板粘贴",
            command=lambda: paste_clipboard(replace=True),
        ).pack(side="left", padx=(7, 0))
        ttk.Button(
            buttons,
            text="取消",
            command=dialog.destroy,
        ).pack(side="right")
        ttk.Button(
            buttons,
            text="导入最后一条有效 JSON",
            command=import_pasted_text,
            style="Set.TButton",
        ).pack(side="right", padx=(0, 7))

        editor.bind("<Command-Return>", import_pasted_text)
        editor.bind("<Control-Return>", import_pasted_text)
        editor.focus_set()
        try:
            clipboard_text = self.root.clipboard_get()
            if isinstance(clipboard_text, str) and "{" in clipboard_text:
                editor.insert("1.0", clipboard_text)
                editor.mark_set("insert", "end")
        except tk.TclError:
            pass
        dialog.grab_set()

    def _poll_events(self):
        try:
            while True:
                kind, value = self.event_queue.get_nowait()
                if kind == "payload":
                    self._accept_payload(value)
                elif kind == "serial_error":
                    self._disconnect_serial()
                    messagebox.showerror("串口读取失败", value)
                elif kind == "solution":
                    self._accept_solution(value)
                elif kind == "solve_error":
                    self._accept_solve_error(value)
        except queue.Empty:
            pass
        self.root.after(40, self._poll_events)

    def _accept_payload(self, payload):
        try:
            pieces = restoration.pieces_from_payload(payload)
        except restoration.PuzzleSolveError as error:
            self.status.set("摄像头数据未采用：%s" % str(error))
            return
        self.latest_payload = payload
        self.live_pieces = pieces
        if self.locked:
            self.data_status.set(
                "SET · 初始值已固定（仍在接收 LIVE 数据）"
            )
        else:
            self.data_status.set(
                "LIVE · 已识别 %d 块碎片" % len(pieces)
            )
            self.status.set(
                "实时数据已更新；确认画面后点击 SET"
            )
            self._draw_scene()

    def _freeze_and_solve(self):
        if self.solving:
            return
        if not self.live_pieces:
            messagebox.showwarning(
                "没有初始值",
                "请先连接 OpenMV，或载入一份摄像头 JSON。",
            )
            return
        try:
            frame_size, divider_y = self._frame_settings()
            clearance_cm = self._clearance_setting()
        except restoration.PuzzleSolveError as error:
            messagebox.showerror("设置错误", str(error))
            return

        self._cancel_animation()
        self.locked = True
        self.solving = True
        self.frozen_pieces = tuple(
            restoration.Piece(piece.piece_id, tuple(piece.vertices))
            for piece in self.live_pieces
        )
        self.solution = None
        self.animation_progress = {}
        self.data_status.set("SET · 初始值已固定")
        selected_mode = self.mode_var.get()
        if selected_mode == MODE_FIXED_QUESTION_ONE:
            self.status.set(
                "正在按PDF第一题固定模板识别4块身份并计算像素比例…"
            )
        else:
            self.status.set("正在计算切割边配对和归位矩阵…")
        self.set_button.configure(state="disabled")
        self.replay_button.configure(state="disabled")
        self._draw_scene()

        worker = threading.Thread(
            target=self._solve_worker,
            args=(
                self.frozen_pieces,
                frame_size,
                divider_y,
                selected_mode,
                clearance_cm,
            ),
            daemon=True,
        )
        worker.start()

    def _solve_worker(
        self,
        pieces,
        frame_size,
        divider_y,
        selected_mode,
        clearance_cm,
    ):
        try:
            if selected_mode == MODE_FIXED_QUESTION_ONE:
                solution = restoration.solve_question_one_fixed(
                    pieces,
                    frame_size=frame_size,
                    divider_y=divider_y,
                    clearance_cm=clearance_cm,
                )
            else:
                solution = restoration.solve_puzzle(
                    pieces,
                    frame_size=frame_size,
                    divider_y=divider_y,
                )
            self.event_queue.put(("solution", solution))
        except Exception as error:
            self.event_queue.put(("solve_error", str(error)))

    def _accept_solution(self, solution):
        if not self.locked:
            return
        self.solving = False
        self.solution = solution
        self.replay_button.configure(state="normal")
        self._show_solution()
        if solution.mode == "fixed_question_one":
            if solution.warnings:
                self.status.set(
                    "第一题已完成实际轮廓归位：比例 %.2f px/cm，"
                    "预留 %.2f cm"
                    % (
                        solution.pixels_per_cm,
                        solution.clearance_cm,
                    )
                )
            else:
                self.status.set(
                    "第一题识别完成：比例 %.2f px/cm，预留 %.2f cm"
                    % (
                        solution.pixels_per_cm,
                        solution.clearance_cm,
                    )
                )
        else:
            self.status.set(
                "计算完成：%d 块碎片，开始播放归位动画"
                % len(solution.pieces)
            )
        self.root.after(250, self._start_animation)

    def _accept_solve_error(self, error_message):
        self.solving = False
        self.status.set("归位计算失败：%s" % error_message)
        messagebox.showerror(
            "归位计算失败",
            error_message
            + "\n\n初始值仍保持冻结，可返回 LIVE 后重新取样。",
        )

    def _show_solution(self):
        for item in self.motion_tree.get_children():
            self.motion_tree.delete(item)
        if self.solution is None:
            return
        template_by_piece = {
            match.piece_id: match
            for match in self.solution.template_matches
        }
        for order, motion in enumerate(self.solution.motions, 1):
            template_match = template_by_piece.get(motion.piece_id)
            piece_label = "P%d" % motion.piece_id
            if template_match is not None:
                piece_label += "/" + template_match.template_id
            self.motion_tree.insert(
                "",
                "end",
                values=(
                    order,
                    piece_label,
                    "%.1f, %.1f" % motion.source_center,
                    "%.1f, %.1f" % motion.target_center,
                    "%+.1f°" % motion.delta_angle_deg,
                ),
            )

        target_x, target_y, target_width, target_height = (
            self.solution.target_bounds
        )
        lines = [
            "实际轮廓目标范围",
            "  origin = (%.2f, %.2f) px" % (target_x, target_y),
            "  size   = %.2f × %.2f px" % (target_width, target_height),
        ]
        if self.solution.mode == "fixed_question_one":
            lines.extend(
                [
                    "  reference = 10.00 × 6.00 cm（仅用于身份和方位）",
                    "  scale    = %.3f px/cm"
                    % self.solution.pixels_per_cm,
                    "  inverse  = %.4f cm/px"
                    % (1.0 / self.solution.pixels_per_cm),
                    "  reserved clearance = %.2f cm"
                    % self.solution.clearance_cm,
                    "  pose tolerance = 2.00 cm（超出仅提示，不拒绝）",
                    "  animation = 保持实际轮廓，只做平移和旋转",
                    "",
                    "第一题固定模板身份",
                ]
            )
            for match in self.solution.template_matches:
                lines.append(
                    "  P%d → %s %s  轮廓=%.1f%%  scale=%.2f px/cm"
                    "  残差RMS=%.2fcm max=%.2fcm  置信=%s"
                    % (
                        match.piece_id,
                        match.template_id,
                        match.template_name,
                        match.normalized_error * 100.0,
                        match.pixels_per_cm,
                        match.observation_rms_cm,
                        match.observation_max_cm,
                        match.confidence,
                    )
                )
            if self.solution.warnings:
                lines.extend(["", "近似匹配提示"])
                for warning in self.solution.warnings:
                    lines.append("  ! " + warning)
        else:
            lines.extend(
                [
                    "  score  = %.3f" % self.solution.score,
                    "",
                    "匹配的切割边",
                ]
            )
            for match in self.solution.matches:
                first_id = self.solution.pieces[match.piece_a].piece_id
                second_id = self.solution.pieces[match.piece_b].piece_id
                lines.append(
                    "  P%d:E%d ↔ P%d:E%d  err=%.2f%%"
                    % (
                        first_id,
                        match.edge_a + 1,
                        second_id,
                        match.edge_b + 1,
                        match.relative_error * 100.0,
                    )
                )
        lines.append("")
        lines.append("当前点 → 目标点的 3×3 矩阵")
        for piece, transform in zip(
            self.solution.pieces, self.solution.transforms
        ):
            matrix = restoration.transform_as_matrix(transform)
            template_match = template_by_piece.get(piece.piece_id)
            matrix_title = "P%d" % piece.piece_id
            if template_match is not None:
                matrix_title += " / " + template_match.template_id
            lines.extend(
                [
                    "",
                    matrix_title,
                    "  [% .6f % .6f % .3f]" % tuple(matrix[0]),
                    "  [% .6f % .6f % .3f]" % tuple(matrix[1]),
                    "  [ 0.000000  0.000000  1.000]",
                ]
            )
        self.details.delete("1.0", "end")
        self.details.insert("1.0", "\n".join(lines))

    def _start_animation(self):
        if self.solution is None:
            return
        self._cancel_animation()
        self.animation_index = 0
        self.animation_fraction = 0.0
        self.animation_progress = {
            piece.piece_id: 0.0 for piece in self.solution.pieces
        }
        self.status.set("归位动画：准备移动第 1 块")
        self._animate_step()

    def _animate_step(self):
        if self.solution is None or not self.locked:
            return
        if self.animation_index >= len(self.solution.motions):
            self.status.set("归位动画完成；可点击“重播归位”再次查看")
            self.animation_after_id = None
            self._draw_scene()
            return

        motion = self.solution.motions[self.animation_index]
        template_match = next(
            (
                match
                for match in self.solution.template_matches
                if match.piece_id == motion.piece_id
            ),
            None,
        )
        motion_label = "P%d" % motion.piece_id
        if template_match is not None:
            motion_label += "/" + template_match.template_id
        step_size = max(0.01, float(self.speed_var.get()) / 44.0)
        self.animation_fraction = min(
            1.0, self.animation_fraction + step_size
        )
        self.animation_progress[motion.piece_id] = self.animation_fraction
        self.status.set(
            "归位动画 %d/%d：%s → (%.1f, %.1f)，旋转 %+.1f°"
            % (
                self.animation_index + 1,
                len(self.solution.motions),
                motion_label,
                motion.target_center[0],
                motion.target_center[1],
                motion.delta_angle_deg,
            )
        )
        self._draw_scene()

        if self.animation_fraction >= 1.0:
            self.animation_index += 1
            self.animation_fraction = 0.0
            delay = 180
        else:
            delay = 28
        self.animation_after_id = self.root.after(
            delay, self._animate_step
        )

    def _cancel_animation(self):
        if self.animation_after_id is not None:
            try:
                self.root.after_cancel(self.animation_after_id)
            except Exception:
                pass
            self.animation_after_id = None

    def _return_to_live(self):
        self._cancel_animation()
        self.locked = False
        self.solving = False
        self.frozen_pieces = ()
        self.solution = None
        self.animation_progress = {}
        self.set_button.configure(state="normal")
        self.replay_button.configure(state="disabled")
        for item in self.motion_tree.get_children():
            self.motion_tree.delete(item)
        self.details.delete("1.0", "end")
        if self.live_pieces:
            self.data_status.set(
                "LIVE · 已识别 %d 块碎片" % len(self.live_pieces)
            )
            self.status.set("已返回实时模式；可重新点击 SET 取样")
        else:
            self.data_status.set("LIVE · 尚未收到有效碎片")
            self.status.set("等待 OpenMV 数据")
        self._draw_scene()

    def _canvas_mapping(self):
        frame_size, _ = self._frame_settings()
        canvas_width = max(10, self.canvas.winfo_width())
        canvas_height = max(10, self.canvas.winfo_height())
        scale = min(
            (canvas_width - 36) / frame_size[0],
            (canvas_height - 36) / frame_size[1],
        )
        offset_x = (canvas_width - (frame_size[0] * scale)) / 2.0
        offset_y = (canvas_height - (frame_size[1] * scale)) / 2.0
        return frame_size, scale, offset_x, offset_y

    @staticmethod
    def _flatten_canvas_points(points, scale, offset_x, offset_y):
        flattened = []
        for point in points:
            flattened.extend(
                (
                    offset_x + (point[0] * scale),
                    offset_y + (point[1] * scale),
                )
            )
        return flattened

    def _draw_scene(self):
        self.canvas.delete("all")
        try:
            frame_size, divider_y = self._frame_settings()
            _, scale, offset_x, offset_y = self._canvas_mapping()
        except restoration.PuzzleSolveError:
            return
        frame_width, frame_height = frame_size
        right = offset_x + (frame_width * scale)
        bottom = offset_y + (frame_height * scale)
        divider_canvas_y = offset_y + (divider_y * scale)

        self.canvas.create_rectangle(
            offset_x,
            offset_y,
            right,
            bottom,
            fill="#eef1f3",
            outline="#b6bec5",
            width=2,
        )
        self.canvas.create_rectangle(
            offset_x,
            divider_canvas_y,
            right,
            bottom,
            fill="#e1e8ed",
            outline="",
        )
        self.canvas.create_line(
            offset_x,
            divider_canvas_y,
            right,
            divider_canvas_y,
            fill="#303840",
            width=3,
        )
        self.canvas.create_text(
            offset_x + 10,
            offset_y + 10,
            text="摄像头实时区域",
            anchor="nw",
            fill="#65717a",
            font=("Helvetica", 11, "bold"),
        )
        self.canvas.create_text(
            offset_x + 10,
            divider_canvas_y + 10,
            text="自动归位目标区域",
            anchor="nw",
            fill="#65717a",
            font=("Helvetica", 11, "bold"),
        )

        pieces = self.frozen_pieces if self.locked else self.live_pieces
        if not pieces:
            self.canvas.create_text(
                (offset_x + right) / 2.0,
                (offset_y + bottom) / 2.0,
                text="连接 OpenMV 或载入 JSON\n收到数据后将在这里实时显示",
                justify="center",
                fill="#6b747b",
                font=("Helvetica", 15),
            )
            return

        transform_by_id = {}
        template_by_piece = {}
        if self.solution is not None:
            transform_by_id = {
                piece.piece_id: transform
                for piece, transform in zip(
                    self.solution.pieces, self.solution.transforms
                )
            }
            template_by_piece = {
                match.piece_id: match
                for match in self.solution.template_matches
            }
            for piece_index, piece in enumerate(self.solution.pieces):
                if len(self.solution.target_polygons) == len(
                    self.solution.pieces
                ):
                    target_polygon = self.solution.target_polygons[
                        piece_index
                    ]
                else:
                    target_polygon = restoration.transform_polygon(
                        piece.vertices,
                        transform_by_id[piece.piece_id],
                    )
                self.canvas.create_polygon(
                    self._flatten_canvas_points(
                        target_polygon, scale, offset_x, offset_y
                    ),
                    fill="",
                    outline=TARGET_COLORS[piece_index % len(TARGET_COLORS)],
                    dash=(6, 4),
                    width=2,
                )

            for motion in self.solution.motions:
                self.canvas.create_line(
                    offset_x + (motion.source_center[0] * scale),
                    offset_y + (motion.source_center[1] * scale),
                    offset_x + (motion.target_center[0] * scale),
                    offset_y + (motion.target_center[1] * scale),
                    fill="#9ba5ad",
                    dash=(4, 4),
                    arrow="last",
                )

        for piece_index, piece in enumerate(pieces):
            progress = self.animation_progress.get(piece.piece_id, 0.0)
            if self.solution is not None:
                displayed_vertices = restoration.interpolate_polygon(
                    piece.vertices,
                    transform_by_id[piece.piece_id],
                    progress,
                )
            else:
                displayed_vertices = list(piece.vertices)
            center = restoration.polygon_centroid(displayed_vertices)
            active = (
                self.solution is not None
                and self.animation_index < len(self.solution.motions)
                and self.solution.motions[
                    self.animation_index
                ].piece_id
                == piece.piece_id
            )
            self.canvas.create_polygon(
                self._flatten_canvas_points(
                    displayed_vertices, scale, offset_x, offset_y
                ),
                fill=PIECE_COLORS[piece_index % len(PIECE_COLORS)],
                outline="#15191d" if not active else "#ffcc33",
                width=4 if active else 2,
            )
            self.canvas.create_text(
                offset_x + (center[0] * scale),
                offset_y + (center[1] * scale),
                text=(
                    "P%d/%s"
                    % (
                        piece.piece_id,
                        template_by_piece[piece.piece_id].template_id,
                    )
                    if piece.piece_id in template_by_piece
                    else "P%d" % piece.piece_id
                ),
                fill="white",
                font=("Helvetica", 12, "bold"),
            )

    def _on_close(self):
        self._cancel_animation()
        self._disconnect_serial()
        self.root.destroy()


def main():
    root = tk.Tk()
    CameraPuzzleSimulator(root)
    root.mainloop()


if __name__ == "__main__":
    main()
