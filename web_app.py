from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
MODEL_PATH = BASE_DIR / "artifacts" / "best_model.pth"

app = FastAPI(title="Gender Classification")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def load_classifier():
    # Import the heavy PyTorch inference stack lazily so the web server can bind
    # to Render's port quickly during startup.
    from inference import GenderClassifier

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {MODEL_PATH.resolve()}. "
            "Train the model first so artifacts/best_model.pth exists."
        )
    return GenderClassifier(model_path=MODEL_PATH)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"request": request, "result": None, "error": None},
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, image: UploadFile = File(...)):
    if not image.filename:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={"request": request, "result": None, "error": "Please upload an image file."},
            status_code=400,
        )

    suffix = Path(image.filename).suffix or ".jpg"
    try:
        classifier = load_classifier()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_path = Path(temp_file.name)
        temp_file.write(await image.read())

    try:
        prediction = classifier.predict(temp_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()

    result = {
        "label": prediction.label,
        "confidence": f"{prediction.confidence * 100:.2f}%",
        "female_probability": f"{prediction.probabilities.get('female', 0.0) * 100:.2f}%",
        "male_probability": f"{prediction.probabilities.get('male', 0.0) * 100:.2f}%",
        "architecture": classifier.architecture,
        "device": classifier.device.type,
    }

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"request": request, "result": result, "error": None},
    )
