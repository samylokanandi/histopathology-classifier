# Breast Histopathology Classifier

Full-stack web app (React + FastAPI + PyTorch) for breast histopathology image
classification. Upload a patch, get a real-time IDC/non-IDC prediction with a
Grad-CAM heatmap showing which regions of the image drove the decision.

## Architecture

```
medvision/
├── model/              # training pipeline (run this first, locally or on Colab)
│   ├── prepare_data.py #   reorganizes the raw Kaggle dataset into train/val folders
│   ├── train.py        #   fine-tunes a ResNet18, saves model.pth
│   ├── gradcam.py       #   Grad-CAM implementation, shared with backend
│   └── requirements.txt
├── backend/             # FastAPI inference server
│   ├── main.py          #   POST /predict — accepts an image, returns prediction + heatmap
│   ├── model_utils.py
│   └── requirements.txt
└── frontend/            # React (Vite) upload UI
    └── src/
```

## Step 1 — Get the data

Dataset: ["Breast Histopathology Images"](https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images)
on Kaggle (~277k 50x50 patches, IDC positive/negative).

Easiest path: open a [Google Colab](https://colab.research.google.com/) notebook (free GPU), then:

```python
import kagglehub
path = kagglehub.dataset_download("paultimothymooney/breast-histopathology-images")
```

That handles Kaggle auth automatically inside Colab. If working locally instead,
download via the Kaggle CLI (`kaggle datasets download -d paultimothymooney/breast-histopathology-images`)
and unzip it.

## Step 2 — Reorganize into train/val folders

```bash
cd model
pip install -r requirements.txt
python prepare_data.py --raw_dir /path/to/raw_data --out_dir ./data --max_per_class 8000
```

`--max_per_class 8000` caps it to ~16k total patches for a fast first run (good
enough to validate the whole pipeline end to end). Drop the flag once you want
to train on the full dataset for your best model.

## Step 3 — Train

```bash
python train.py --data_dir ./data --epochs 8 --batch_size 64
```

This saves the best checkpoint to `model.pth`. On a Colab GPU, 8 epochs on the
capped dataset should take well under 10 minutes. Watch `val_acc` in the printed
output — published baselines on this dataset land around 85-90% accuracy with a
ResNet-style model, so that's a reasonable target.

Copy the resulting `model.pth` into `backend/`.

## Step 4 — Run the backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Check `http://localhost:8000/health` — it should report `model_loaded: true`.

## Step 5 — Run the frontend

```bash
cd frontend
npm install
npm run dev
```

Opens on `http://localhost:5173` by default, talking to the backend at
`http://localhost:8000`. To point at a different backend URL (e.g. once deployed),
set `VITE_API_URL` in a `.env` file in `frontend/`.

## Step 6 — Deploy (so the resume's [Live] link is real)

A simple free-tier path:
- **Backend**: [Render](https://render.com) or [Railway](https://railway.app) — both support a Python web service from a GitHub repo. Set the start command to `uvicorn main:app --host 0.0.0.0 --port $PORT`. Note: model.pth needs to ship with the repo or be pulled from somewhere at startup (e.g. a small file in the repo if it's under ~100MB, otherwise host it on Hugging Face Hub or a GitHub release and download it on startup).
- **Frontend**: [Vercel](https://vercel.com) or [Netlify](https://netlify.com) — connect the repo, set the root to `frontend/`, set `VITE_API_URL` to your deployed backend URL.

## What's still ahead

- [ ] Train the model and confirm a reasonable val accuracy
- [ ] Smoke-test the backend locally with a few sample patches
- [ ] Polish/test the frontend against the running backend
- [ ] Push to a public GitHub repo
- [ ] Deploy backend + frontend, update the resume's [Live]/[GitHub] links
- [ ] Be ready to explain Grad-CAM, the train/val split choice, and what you'd improve with more time (this is exactly the kind of thing live-coding interviewers probe on)
