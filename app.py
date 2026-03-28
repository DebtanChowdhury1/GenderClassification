from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageOps, ImageTk

from inference import DEFAULT_MODEL_PATH, GenderClassifier


APP_TITLE = "Gender Classification"
PREVIEW_SIZE = (320, 320)


class GenderClassifierApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("760x620")
        self.root.minsize(700, 560)

        self.model_path = tk.StringVar(value=str(DEFAULT_MODEL_PATH))
        self.status_var = tk.StringVar(value="Load a trained model and choose an image.")
        self.prediction_var = tk.StringVar(value="Prediction: -")
        self.confidence_var = tk.StringVar(value="Confidence: -")
        self.image_path_var = tk.StringVar(value="No image selected")
        self.device_var = tk.StringVar(value="Device: loading...")

        self.classifier: GenderClassifier | None = None
        self.preview_image = None

        self._build_ui()
        self._load_model_on_startup()

    def _build_ui(self) -> None:
        self.root.configure(padx=16, pady=16)

        title_label = ttk.Label(self.root, text=APP_TITLE, font=("Segoe UI", 22, "bold"))
        title_label.pack(anchor="w")

        subtitle_label = ttk.Label(
            self.root,
            text="Upload an image and run local gender classification inference.",
            font=("Segoe UI", 10),
        )
        subtitle_label.pack(anchor="w", pady=(0, 12))

        model_frame = ttk.LabelFrame(self.root, text="Model")
        model_frame.pack(fill="x", pady=(0, 12))
        model_frame.columnconfigure(1, weight=1)

        ttk.Label(model_frame, text="Checkpoint").grid(row=0, column=0, padx=8, pady=8, sticky="w")
        ttk.Entry(model_frame, textvariable=self.model_path).grid(
            row=0, column=1, padx=8, pady=8, sticky="ew"
        )
        ttk.Button(model_frame, text="Browse", command=self._browse_model).grid(
            row=0, column=2, padx=8, pady=8
        )
        ttk.Button(model_frame, text="Load Model", command=self._load_model).grid(
            row=0, column=3, padx=8, pady=8
        )
        ttk.Label(model_frame, textvariable=self.device_var).grid(
            row=1, column=0, columnspan=4, padx=8, pady=(0, 8), sticky="w"
        )

        image_frame = ttk.LabelFrame(self.root, text="Image")
        image_frame.pack(fill="both", expand=True, pady=(0, 12))
        image_frame.columnconfigure(0, weight=1)
        image_frame.rowconfigure(1, weight=1)

        controls = ttk.Frame(image_frame)
        controls.grid(row=0, column=0, sticky="ew", padx=8, pady=8)
        controls.columnconfigure(1, weight=1)

        ttk.Button(controls, text="Choose Image", command=self._browse_image).grid(
            row=0, column=0, padx=(0, 8)
        )
        ttk.Label(controls, textvariable=self.image_path_var).grid(row=0, column=1, sticky="w")

        self.preview_label = ttk.Label(
            image_frame,
            text="Image preview",
            anchor="center",
            relief="solid",
        )
        self.preview_label.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))

        result_frame = ttk.LabelFrame(self.root, text="Result")
        result_frame.pack(fill="x")

        ttk.Label(result_frame, textvariable=self.prediction_var, font=("Segoe UI", 14, "bold")).pack(
            anchor="w", padx=8, pady=(8, 4)
        )
        ttk.Label(result_frame, textvariable=self.confidence_var, font=("Segoe UI", 11)).pack(
            anchor="w", padx=8, pady=(0, 4)
        )
        ttk.Label(result_frame, textvariable=self.status_var, font=("Segoe UI", 10)).pack(
            anchor="w", padx=8, pady=(0, 8)
        )

    def _browse_model(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select Model Checkpoint",
            filetypes=[("PyTorch Model", "*.pth"), ("All Files", "*.*")],
        )
        if selected:
            self.model_path.set(selected)

    def _load_model_on_startup(self) -> None:
        if Path(self.model_path.get()).exists():
            self._load_model()
        else:
            self.status_var.set("Checkpoint not found. Train the notebook first to create artifacts/best_model.pth.")
            self.device_var.set("Device: unavailable until model is loaded")

    def _load_model(self) -> None:
        try:
            self.classifier = GenderClassifier(self.model_path.get())
        except Exception as exc:
            self.classifier = None
            self.device_var.set("Device: unavailable")
            self.status_var.set("Failed to load model.")
            messagebox.showerror("Model Load Error", str(exc))
            return

        self.device_var.set(
            f"Device: {self.classifier.device.type} | Architecture: {self.classifier.architecture}"
        )
        self.status_var.set(f"Model loaded from {Path(self.model_path.get()).resolve()}")

    def _browse_image(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.webp"),
                ("All Files", "*.*"),
            ],
        )
        if not selected:
            return

        self.image_path_var.set(selected)
        self._show_preview(selected)
        self._predict(selected)

    def _show_preview(self, image_path: str) -> None:
        image = Image.open(image_path).convert("RGB")
        preview = ImageOps.contain(image, PREVIEW_SIZE)
        self.preview_image = ImageTk.PhotoImage(preview)
        self.preview_label.configure(image=self.preview_image, text="")

    def _predict(self, image_path: str) -> None:
        if self.classifier is None:
            messagebox.showwarning("Model Not Loaded", "Load a trained model before predicting.")
            return

        try:
            result = self.classifier.predict(image_path)
        except Exception as exc:
            self.status_var.set("Prediction failed.")
            messagebox.showerror("Prediction Error", str(exc))
            return

        female_prob = result.probabilities.get("female", 0.0) * 100
        male_prob = result.probabilities.get("male", 0.0) * 100
        self.prediction_var.set(f"Prediction: {result.label}")
        self.confidence_var.set(f"Confidence: {result.confidence * 100:.2f}%")
        self.status_var.set(f"Female: {female_prob:.2f}% | Male: {male_prob:.2f}%")


def main() -> None:
    root = tk.Tk()
    style = ttk.Style()
    if "vista" in style.theme_names():
        style.theme_use("vista")
    app = GenderClassifierApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
