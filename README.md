# Gender Classification

This project includes:

- `gender_classification.ipynb`: research and training notebook
- `inference.py`: reusable prediction module
- `app.py`: local desktop UI for end users
- `web_app.py`: deployable FastAPI web app for Render

## 1. Train The Model

Open `gender_classification.ipynb` in VS Code, select the `.venv` kernel, and run all cells.

Training creates:

- `artifacts/best_model.pth`
- `artifacts/history.csv`
- `artifacts/classification_report.csv`
- `artifacts/labels.json`

## 2. Launch The UI

Run:

```powershell
.\.venv\Scripts\python.exe app.py
```

Then:

1. Load the trained model checkpoint if it is not loaded automatically.
2. Choose an image.
3. View the predicted label and confidence.

## 3. Run The Web App Locally

Run:

```powershell
.\.venv\Scripts\python.exe -m uvicorn web_app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`.

## 4. Deploy On Render

This repo includes `render.yaml` for a Python web service.

Important:

- Render can deploy the web app, not the Tkinter desktop UI.
- Free Render web services spin down after 15 minutes of inactivity.
- Render's filesystem is ephemeral, so the deployed service needs `artifacts/best_model.pth` present in the repo or available through another storage approach.

Recommended lightweight deployment:

1. Push code to GitHub.
2. If your trained model is small enough and acceptable to commit, place it at `artifacts/best_model.pth`.
3. Because `.gitignore` excludes `artifacts/`, add the checkpoint explicitly with:

```powershell
git add -f artifacts/best_model.pth
```
4. In Render, create a new Blueprint or Web Service from the repo.
5. Render will use:
   - build: `pip install -r requirements.txt`
   - start: `uvicorn web_app:app --host 0.0.0.0 --port $PORT`
6. Visit the deployed URL and upload an image.

## Notes

- The default checkpoint path is `artifacts/best_model.pth`.
- The UI runs fully locally.
- If no checkpoint exists yet, train the notebook first.
