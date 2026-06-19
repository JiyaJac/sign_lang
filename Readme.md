# ✋ Sign Language Detector

A real-time sign language detector that uses hand-tracking and a trained
machine learning model to recognize signs through your webcam, build up
letters into words, and display them live — right in the browser.

<img width="1918" height="963" alt="Screenshot 2026-06-19 114749" src="https://github.com/user-attachments/assets/ed4421c3-a879-4d78-8806-c2c96c853688" />


## How it works

1. Your webcam feed is captured in the browser
2. [MediaPipe](https://github.com/google/mediapipe) detects your hand and extracts 21 landmark points (x, y, z)
3. A trained `RandomForestClassifier` predicts which sign you're making from those points
4. Holding a sign steady for a few frames "locks it in" as the current letter
5. Special signs control the output:
   - `next` → adds the current letter to the word
   - `space` → adds a space
   - `del` → deletes the last letter

## Project structure

```
.
├── app.py                  # Streamlit app — the deployed detector
├── model.pkl               # trained RandomForestClassifier
├── requirements.txt        # dependencies for the deployed app
│
└── Required/                # not deployed — used once to produce model.pkl
    ├── Data_Collection.ipynb     # captures hand landmarks + labels into a CSV
    ├── Execution.ipynb      # predicts before making app
    ├── Training.ipynb       # trains the model from the CSV, saves model.pkl
    └── data1.csv             # collected training samples
```

> The `training/` scripts were used once, offline, to create `model.pkl`.
> The deployed app only loads that file and runs predictions — it doesn't
> retrain or touch the CSV.

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

This opens the app in your browser. Click **Start** on the camera widget and
allow camera access.

## Retraining the model (optional)

If you want to collect your own data and retrain:

```bash
cd training
python collect_data.py     # collects landmarks for each sign into data1.csv
python train_model.py      # trains a new model.pkl from data1.csv
```

Then copy the new `model.pkl` into the project root to use it in `app.py`.

## Deployment

This app is built to deploy for free on either:

- **[Streamlit Community Cloud](https://share.streamlit.io)** — connect this repo, set the main file to `app.py`, deploy
- **[Hugging Face Spaces](https://huggingface.co/new-space)** — choose the Streamlit SDK, upload these files

Both platforms support the webcam + Python + ML stack this app needs out of
the box.

## Tech stack

- [Streamlit](https://streamlit.io) — web app framework
- [streamlit-webrtc](https://github.com/whitphx/streamlit-webrtc) — browser camera streaming
- [MediaPipe](https://github.com/google/mediapipe) — hand landmark detection
- [scikit-learn](https://scikit-learn.org) — RandomForestClassifier
- [OpenCV](https://opencv.org) — image processing

## License

MIT
