from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.models import efficientnet_b0, efficientnet_b2, efficientnet_b3, mobilenet_v2, resnet50


DEFAULT_MODEL_PATH = Path("artifacts/best_model.pth")
DEFAULT_CLASS_NAMES = ["female", "male"]
DEFAULT_ARCHITECTURE = "efficientnet_b2"


@dataclass
class PredictionResult:
    label: str
    confidence: float
    raw_class: str
    probabilities: dict[str, float]


def get_device() -> torch.device:
    """Return CUDA when available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_model(architecture: str = DEFAULT_ARCHITECTURE, num_classes: int = 2) -> nn.Module:
    """Create the architecture used for inference from checkpoint metadata."""
    if architecture == "mobilenet_v2":
        model = mobilenet_v2(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    if architecture in {"efficientnet_b0", "efficientnet_b2", "efficientnet_b3"}:
        builder = {
            "efficientnet_b0": efficientnet_b0,
            "efficientnet_b2": efficientnet_b2,
            "efficientnet_b3": efficientnet_b3,
        }[architecture]
        model = builder(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    if architecture == "resnet50":
        model = resnet50(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    raise ValueError(f"Unsupported architecture: {architecture}")
    return model


def build_predict_transform(image_size: int) -> transforms.Compose:
    """Build the inference transform expected by the trained model."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class GenderClassifier:
    """Load a trained checkpoint and run single-image inference."""

    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH, device: torch.device | None = None):
        self.model_path = Path(model_path)
        self.device = device or get_device()
        self.model: nn.Module | None = None
        self.class_to_idx: dict[str, int] = {}
        self.idx_to_class: dict[int, str] = {}
        self.image_size = 128
        self.class_names = DEFAULT_CLASS_NAMES.copy()
        self.architecture = DEFAULT_ARCHITECTURE
        self._load()

    def _load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found at: {self.model_path.resolve()}. "
                "Train the model first and ensure best_model.pth exists."
            )

        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.class_names = checkpoint.get("class_names", DEFAULT_CLASS_NAMES)
        self.class_to_idx = checkpoint.get(
            "class_to_idx",
            {class_name: idx for idx, class_name in enumerate(self.class_names)},
        )
        self.idx_to_class = {idx: class_name for class_name, idx in self.class_to_idx.items()}
        self.image_size = checkpoint.get("image_size", 128)
        self.architecture = checkpoint.get("architecture", "mobilenet_v2")

        self.model = create_model(
            architecture=self.architecture,
            num_classes=len(self.class_names),
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def predict(self, image_path: str | Path) -> PredictionResult:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        transform = build_predict_transform(self.image_size)
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1).squeeze(0)

        probability_map = {
            self.idx_to_class[idx]: float(probabilities[idx].item())
            for idx in sorted(self.idx_to_class)
        }
        predicted_idx = int(torch.argmax(probabilities).item())
        raw_class = self.idx_to_class[predicted_idx].lower()
        label = "Male" if raw_class == "male" else "Female"
        confidence = probability_map[raw_class]

        return PredictionResult(
            label=label,
            confidence=confidence,
            raw_class=raw_class,
            probabilities=probability_map,
        )
