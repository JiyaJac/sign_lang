# Sign Language Detector — Streamlit Version

This wraps your original `execution.py` logic almost line-for-line — same
MediaPipe calls, same model, same letter/space/del/next logic — just running
inside a Streamlit web app instead of a local OpenCV window.

## Files
```
streamlit-sign-app/
├── app.py             ← your execution.py logic, adapted for the browser camera
├── requirements.txt
└── model.pkl          ← you add this (copy from your existing project)
```

## Step 1 — Add your model

Copy your trained `model.pkl` into this folder, next to `app.py`.

## Step 2 — Run it locally (test first)

```bash
pip install -r requirements.txt
streamlit run app.py
```

It opens in your browser automatically. Click "Start" on the camera widget
and allow camera access.

## Step 3 — Deploy for free

**Option A: Streamlit Community Cloud (easiest)**
1. Push this folder to a GitHub repo
2. Go to https://share.streamlit.io
3. "New app" → select your repo → set main file to `app.py`
4. Deploy — you get a public URL instantly

**Option B: Hugging Face Spaces**
1. Go to https://huggingface.co/new-space
2. Choose "Streamlit" as the SDK
3. Upload these files (or connect via git)
4. It builds and hosts automatically

Both are free and made for exactly this kind of app — Python + camera + ML
model, no separate frontend/backend split needed.

## What's different from execution.py

- `cv2.imshow()` / `cv2.waitKey()` are removed — Streamlit's camera widget
  replaces the OpenCV window entirely
- State (`word`, `current_letter`, `count`, `prev_prediction`, `inserted`)
  now lives inside a `SignProcessor` class instead of loose variables,
  because Streamlit needs it to persist across video frames
- Everything else — MediaPipe calls, model.predict, the space/del/next
  logic — is copied directly from your script