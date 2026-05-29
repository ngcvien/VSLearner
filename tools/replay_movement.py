import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import messagebox, ttk


# --- CAU HINH ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "movement_data"
RECORD_ATMS_SCRIPT = PROJECT_ROOT / "tools" / "record_movement_atms.py"
WIDTH, HEIGHT = 640, 480
DEFAULT_DELAY_MS = 40


# MediaPipe Hand Connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


@dataclass
class MovementSample:
    path: Path
    label: str
    frames: int
    modified: float
    size: int

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def modified_text(self) -> str:
        return datetime.fromtimestamp(self.modified).strftime("%Y-%m-%d %H:%M:%S")

    @property
    def size_text(self) -> str:
        return f"{self.size / 1024:.1f} KB"


def draw_hand_from_landmarks(canvas: np.ndarray, landmarks: list[float]) -> None:
    """Ve ban tay len canvas tu danh sach 63 gia tri landmark."""
    h, w, _ = canvas.shape
    points: list[tuple[int, int]] = []

    for i in range(0, min(len(landmarks), 63), 3):
        x = landmarks[i]
        y = landmarks[i + 1]
        px, py = int(x * w), int(y * h)
        points.append((px, py))
        cv2.circle(canvas, (px, py), 4, (0, 255, 255), -1)

    for p1_idx, p2_idx in HAND_CONNECTIONS:
        if p1_idx < len(points) and p2_idx < len(points):
            cv2.line(canvas, points[p1_idx], points[p2_idx], (0, 255, 0), 2)


def render_frame(
    frame_landmarks: list[float] | None,
    label: str,
    frame_index: int,
    total_frames: int,
) -> np.ndarray:
    canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    if frame_landmarks:
        draw_hand_from_landmarks(canvas, frame_landmarks)

    cv2.putText(
        canvas,
        f"Label: {label}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        canvas,
        f"Frame: {frame_index + 1}/{max(total_frames, 1)}",
        (10, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (210, 210, 210),
        1,
    )
    return canvas


def load_sample_metadata(path: Path) -> MovementSample | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            data: dict[str, Any] = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    stat = path.stat()
    label = str(data.get("label") or path.parent.name)
    sequence = data.get("sequence") or []
    return MovementSample(
        path=path,
        label=label,
        frames=len(sequence),
        modified=stat.st_mtime,
        size=stat.st_size,
    )


class ReplayMovementApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Movement Replay")
        self.root.minsize(1040, 620)

        self.samples: list[MovementSample] = []
        self.filtered_samples: list[MovementSample] = []
        self.selected_sample: MovementSample | None = None
        self.sequence: list[list[float]] = []
        self.frame_index = 0
        self.is_playing = False
        self.after_id: str | None = None
        self.sort_column = "label"
        self.sort_reverse = False
        self.photo: ImageTk.PhotoImage | None = None

        self.search_var = tk.StringVar()
        self.label_filter_var = tk.StringVar(value="Tat ca")
        self.speed_var = tk.IntVar(value=DEFAULT_DELAY_MS)
        self.frame_var = tk.IntVar(value=0)
        self.status_var = tk.StringVar(value="")

        self._build_ui()
        self.refresh_samples()

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        main = ttk.Frame(self.root, padding=10)
        main.grid(row=0, column=0, sticky="nsew")
        main.columnconfigure(0, weight=0)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        left = ttk.Frame(main)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        left.columnconfigure(0, weight=1)
        left.rowconfigure(1, weight=1)

        filter_bar = ttk.Frame(left)
        filter_bar.grid(row=0, column=0, sticky="ew")
        filter_bar.columnconfigure(0, weight=1)
        filter_bar.columnconfigure(1, weight=1)

        ttk.Label(filter_bar, text="Tim hanh dong").grid(row=0, column=0, sticky="w")
        ttk.Label(filter_bar, text="Loc nhan").grid(row=0, column=1, sticky="w", padx=(8, 0))

        search_entry = ttk.Entry(filter_bar, textvariable=self.search_var, width=20)
        search_entry.grid(row=1, column=0, sticky="ew", pady=(4, 8))
        search_entry.bind("<KeyRelease>", lambda _event: self.apply_filter())

        self.label_filter = ttk.Combobox(
            filter_bar,
            textvariable=self.label_filter_var,
            state="readonly",
            width=18,
            values=("Tat ca",),
        )
        self.label_filter.grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=(4, 8))
        self.label_filter.bind("<<ComboboxSelected>>", lambda _event: self.apply_filter())

        columns = ("label", "file", "frames", "modified", "size")
        self.tree = ttk.Treeview(left, columns=columns, show="headings", height=22)
        headings = {
            "label": "Hanh dong",
            "file": "File",
            "frames": "Frames",
            "modified": "Ngay sua",
            "size": "Dung luong",
        }
        widths = {
            "label": 110,
            "file": 140,
            "frames": 70,
            "modified": 145,
            "size": 80,
        }
        for column in columns:
            self.tree.heading(
                column,
                text=headings[column],
                command=lambda c=column: self.sort_by(c),
            )
            self.tree.column(column, width=widths[column], anchor="w")
        self.tree.column("frames", anchor="center")
        self.tree.column("size", anchor="e")
        self.tree.grid(row=1, column=0, sticky="nsew")
        self.tree.bind("<<TreeviewSelect>>", self.on_select_sample)
        self.tree.bind("<Double-1>", lambda _event: self.play())

        scrollbar = ttk.Scrollbar(left, orient="vertical", command=self.tree.yview)
        scrollbar.grid(row=1, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=scrollbar.set)

        left_buttons = ttk.Frame(left)
        left_buttons.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        ttk.Button(left_buttons, text="Tai lai", command=self.refresh_samples).pack(side="left")
        ttk.Button(left_buttons, text="Mo thu muc data", command=self.open_data_dir).pack(
            side="left", padx=(8, 0)
        )
        ttk.Button(left_buttons, text="Xoa mau", command=self.delete_selected_samples).pack(
            side="left", padx=(8, 0)
        )
        ttk.Button(left_buttons, text="Record ATMS", command=self.run_record_movement_atms).pack(
            side="left", padx=(8, 0)
        )

        right = ttk.Frame(main)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)

        self.canvas_label = ttk.Label(right, anchor="center")
        self.canvas_label.grid(row=0, column=0, sticky="nsew")

        controls = ttk.Frame(right)
        controls.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        controls.columnconfigure(5, weight=1)

        ttk.Button(controls, text="Play", command=self.play).grid(row=0, column=0, padx=(0, 6))
        ttk.Button(controls, text="Pause", command=self.pause).grid(row=0, column=1, padx=(0, 6))
        ttk.Button(controls, text="Stop", command=self.stop).grid(row=0, column=2, padx=(0, 14))
        ttk.Button(controls, text="< Frame", command=self.previous_frame).grid(
            row=0, column=3, padx=(0, 6)
        )
        ttk.Button(controls, text="Frame >", command=self.next_frame).grid(row=0, column=4)

        self.frame_scale = ttk.Scale(
            controls,
            from_=0,
            to=0,
            variable=self.frame_var,
            command=self.on_seek,
        )
        self.frame_scale.grid(row=0, column=5, sticky="ew", padx=12)

        ttk.Label(controls, text="Delay ms").grid(row=0, column=6, padx=(0, 4))
        ttk.Spinbox(
            controls,
            from_=10,
            to=300,
            increment=5,
            width=6,
            textvariable=self.speed_var,
        ).grid(row=0, column=7)

        ttk.Label(right, textvariable=self.status_var, anchor="w").grid(
            row=2, column=0, sticky="ew", pady=(8, 0)
        )

    def refresh_samples(self) -> None:
        self.pause()
        samples: list[MovementSample] = []
        if DATA_DIR.exists():
            for path in DATA_DIR.glob("*/*.json"):
                sample = load_sample_metadata(path)
                if sample:
                    samples.append(sample)

        self.samples = samples
        labels = ["Tat ca"] + sorted({sample.label for sample in samples}, key=str.lower)
        self.label_filter.configure(values=labels)
        if self.label_filter_var.get() not in labels:
            self.label_filter_var.set("Tat ca")
        self.apply_filter()
        self.status_var.set(f"Da tai {len(samples)} mau tu {DATA_DIR}")

    def apply_filter(self) -> None:
        keyword = self.search_var.get().strip().lower()
        selected_label = self.label_filter_var.get()
        if keyword or selected_label != "Tat ca":
            self.filtered_samples = [
                sample
                for sample in self.samples
                if (selected_label == "Tat ca" or sample.label == selected_label)
                and (not keyword or keyword in sample.label.lower() or keyword in sample.name.lower())
            ]
        else:
            self.filtered_samples = list(self.samples)
        self._sort_filtered()
        self._render_tree()

    def sort_by(self, column: str) -> None:
        if self.sort_column == column:
            self.sort_reverse = not self.sort_reverse
        else:
            self.sort_column = column
            self.sort_reverse = False
        self._sort_filtered()
        self._render_tree()

    def _sort_filtered(self) -> None:
        key_map = {
            "label": lambda item: item.label.lower(),
            "file": lambda item: item.name.lower(),
            "frames": lambda item: item.frames,
            "modified": lambda item: item.modified,
            "size": lambda item: item.size,
        }
        self.filtered_samples.sort(
            key=key_map.get(self.sort_column, key_map["label"]),
            reverse=self.sort_reverse,
        )

    def _render_tree(self) -> None:
        selected_path = str(self.selected_sample.path) if self.selected_sample else None
        for item in self.tree.get_children():
            self.tree.delete(item)

        for sample in self.filtered_samples:
            item_id = str(sample.path)
            self.tree.insert(
                "",
                "end",
                iid=item_id,
                values=(
                    sample.label,
                    sample.name,
                    sample.frames,
                    sample.modified_text,
                    sample.size_text,
                ),
            )

        if selected_path and self.tree.exists(selected_path):
            self.tree.selection_set(selected_path)

    def on_select_sample(self, _event: tk.Event) -> None:
        selected = self.tree.selection()
        if not selected:
            return

        path = Path(selected[0])
        sample = next((item for item in self.samples if item.path == path), None)
        if not sample:
            return

        self.load_sequence(sample)

    def load_sequence(self, sample: MovementSample) -> None:
        self.pause()
        try:
            with sample.path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            messagebox.showerror("Loi doc file", f"Khong the doc file:\n{sample.path}\n\n{exc}")
            return

        sequence = data.get("sequence") or []
        self.selected_sample = sample
        self.sequence = sequence
        self.frame_index = 0
        self.frame_var.set(0)
        self.frame_scale.configure(to=max(len(sequence) - 1, 0))
        self.show_current_frame()

    def show_current_frame(self) -> None:
        label = self.selected_sample.label if self.selected_sample else ""
        total = len(self.sequence)
        frame = self.sequence[self.frame_index] if total else None
        canvas = render_frame(frame, label, self.frame_index, total)
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        self.photo = ImageTk.PhotoImage(image=image)
        self.canvas_label.configure(image=self.photo)
        self.frame_var.set(self.frame_index)

        if self.selected_sample:
            self.status_var.set(
                f"{self.selected_sample.label} | {self.selected_sample.name} | "
                f"frame {self.frame_index + 1}/{max(total, 1)}"
            )

    def play(self) -> None:
        if not self.sequence:
            messagebox.showinfo("Chua chon du lieu", "Hay chon mot mau movement trong danh sach.")
            return
        if self.is_playing:
            return
        if self.frame_index >= len(self.sequence) - 1:
            self.frame_index = 0
        self.is_playing = True
        self._schedule_next_frame()

    def pause(self) -> None:
        self.is_playing = False
        if self.after_id:
            self.root.after_cancel(self.after_id)
            self.after_id = None

    def stop(self) -> None:
        self.pause()
        self.frame_index = 0
        self.show_current_frame()

    def previous_frame(self) -> None:
        if not self.sequence:
            return
        self.pause()
        self.frame_index = max(0, self.frame_index - 1)
        self.show_current_frame()

    def next_frame(self) -> None:
        if not self.sequence:
            return
        self.pause()
        self.frame_index = min(len(self.sequence) - 1, self.frame_index + 1)
        self.show_current_frame()

    def on_seek(self, value: str) -> None:
        if not self.sequence:
            return
        index = int(float(value))
        if index != self.frame_index:
            self.frame_index = max(0, min(len(self.sequence) - 1, index))
            self.show_current_frame()

    def _schedule_next_frame(self) -> None:
        if not self.is_playing:
            return

        self.show_current_frame()
        if self.frame_index >= len(self.sequence) - 1:
            self.is_playing = False
            return

        self.frame_index += 1
        delay = max(10, int(self.speed_var.get()))
        self.after_id = self.root.after(delay, self._schedule_next_frame)

    def open_data_dir(self) -> None:
        if not DATA_DIR.exists():
            messagebox.showwarning("Khong tim thay", f"Thu muc khong ton tai:\n{DATA_DIR}")
            return
        os.startfile(DATA_DIR)

    def delete_selected_samples(self) -> None:
        selected = self.tree.selection()
        if not selected:
            messagebox.showinfo("Chua chon mau", "Hay chon mau can xoa trong danh sach.")
            return

        paths = [Path(item) for item in selected]
        names = "\n".join(path.name for path in paths[:8])
        if len(paths) > 8:
            names += f"\n... va {len(paths) - 8} file khac"

        confirmed = messagebox.askyesno(
            "Xac nhan xoa",
            f"Ban co chac muon xoa {len(paths)} mau nay?\n\n{names}",
        )
        if not confirmed:
            return

        self.pause()
        deleted_count = 0
        errors: list[str] = []

        data_root = DATA_DIR.resolve()
        for path in paths:
            try:
                resolved_path = path.resolve()
                resolved_path.relative_to(data_root)
                if resolved_path.suffix.lower() != ".json":
                    raise ValueError("Chi duoc xoa file .json")
                resolved_path.unlink()
                deleted_count += 1

                try:
                    resolved_path.parent.rmdir()
                except OSError:
                    pass
            except (OSError, ValueError) as exc:
                errors.append(f"{path.name}: {exc}")

        if self.selected_sample and self.selected_sample.path in paths:
            self.selected_sample = None
            self.sequence = []
            self.frame_index = 0
            self.frame_var.set(0)
            self.frame_scale.configure(to=0)
            self.canvas_label.configure(image="")
            self.photo = None

        self.refresh_samples()

        if errors:
            messagebox.showwarning(
                "Xoa chua hoan tat",
                f"Da xoa {deleted_count} mau.\n\nLoi:\n" + "\n".join(errors[:8]),
            )
        else:
            self.status_var.set(f"Da xoa {deleted_count} mau.")

    def run_record_movement_atms(self) -> None:
        if not RECORD_ATMS_SCRIPT.exists():
            messagebox.showerror(
                "Khong tim thay script",
                f"Khong tim thay file:\n{RECORD_ATMS_SCRIPT}",
            )
            return

        creationflags = 0
        if os.name == "nt":
            creationflags = subprocess.CREATE_NEW_CONSOLE

        try:
            subprocess.Popen(
                [sys.executable, str(RECORD_ATMS_SCRIPT)],
                cwd=str(RECORD_ATMS_SCRIPT.parent),
                creationflags=creationflags,
            )
        except OSError as exc:
            messagebox.showerror("Khong the chay recorder", str(exc))
            return

        self.status_var.set("Da mo record_movement_atms.py trong cua so rieng.")


def main() -> None:
    root = tk.Tk()
    app = ReplayMovementApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.pause(), root.destroy()))
    root.mainloop()


if __name__ == "__main__":
    main()
