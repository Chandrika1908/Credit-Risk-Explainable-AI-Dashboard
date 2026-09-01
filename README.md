# 💳 Credit Risk Explainable AI Dashboard

An end-to-end machine learning application that predicts loan default risk
**and explains every prediction** using SHAP — built to demonstrate the full
ML lifecycle: data generation, model training, model comparison, explainable
AI, and interactive deployment.

**Live demo:** _add your Streamlit Cloud URL here after deploying_

---

## Why this project

Most portfolio projects stop at "trained a model, got 90% accuracy."
This one goes further, because that's what real ML work actually requires:

- **Model comparison, not a single model** — Logistic Regression, Random
  Forest, and XGBoost are trained side by side with ROC curves and a
  full metrics table, so the trade-offs (precision vs. recall) are visible.
- **Explainability, not a black box** — SHAP values show *why* the model
  flagged an applicant as high-risk, both globally (which features matter
  most overall) and locally (why *this* applicant scored the way they did).
- **Interactivity, not a static notebook** — a live what-if simulator lets
  you build a hypothetical applicant and watch the risk score and its
  explanation update in real time.
- **Zero external dependencies at deploy time** — the dataset is generated
  synthetically with realistic, hand-tuned relationships, so there's no
  broken download link or stale Kaggle CSV to break the deployment.

## Tech stack

| Layer | Tools |
|---|---|
| App / UI | Streamlit, Plotly |
| ML | scikit-learn, XGBoost |
| Explainability | SHAP |
| Data | NumPy, pandas (synthetic generator) |

## Project structure

```
credit-risk-explainable-ai/
├── app.py                    # Main Streamlit app (4 tabs)
├── src/
│   ├── data_generator.py     # Synthetic applicant data generator
│   ├── model_trainer.py      # Trains & evaluates 3 models
│   └── shap_utils.py         # SHAP explanation helpers
├── .streamlit/
│   └── config.toml           # Theme
├── requirements.txt
├── LICENSE
└── README.md
```

## Run locally

```bash
git clone https://github.com/YOUR_USERNAME/credit-risk-explainable-ai.git
cd credit-risk-explainable-ai
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## Deploying for free on Streamlit Community Cloud

**No credit card is required for this.** You only need a GitHub account.
The free tier gives you unlimited public apps (one private app), roughly
1 GB of memory per app, and apps sleep after ~12 hours of no traffic
(they wake up automatically on the next visit, with a short delay).

### Step 1 — Push this repo to GitHub

```bash
cd credit-risk-explainable-ai
git init
git add .
git commit -m "Initial commit: Credit Risk Explainable AI Dashboard"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/credit-risk-explainable-ai.git
git push -u origin main
```

If you don't have a GitHub repo yet: go to github.com → **New repository**
→ name it `credit-risk-explainable-ai` → **Public** → Create, then run the
commands above.

### Step 2 — Deploy on Streamlit Community Cloud

1. Go to **share.streamlit.io**
2. Click **Sign in with GitHub** and authorize Streamlit (no card, no payment step)
3. Click **Create app** → **From existing repo**
4. Select:
   - Repository: `YOUR_USERNAME/credit-risk-explainable-ai`
   - Branch: `main`
   - Main file path: `app.py`
5. Click **Deploy**

Streamlit installs `requirements.txt` and launches the app. First build
takes 2–4 minutes (installing scikit-learn/XGBoost/SHAP). You'll get a
public URL like:

```
https://YOUR_USERNAME-credit-risk-explainable-ai.streamlit.app
```

### Step 3 — Share it

Put the live URL in this README, your resume, and your LinkedIn. Recruiters
and reviewers can open it with zero setup — that's the entire point of the
free hosting tier.

### If the deploy fails

- Check the build logs in the Streamlit Cloud dashboard (usually a
  version mismatch — the pinned `requirements.txt` versions above are
  tested to work together on Streamlit Cloud's Python 3.11 runtime).
- Memory errors: this app is tuned to stay under 1 GB (SHAP sampling is
  capped at 250 rows). If you expand the dataset size significantly,
  reduce `max_samples` in `app.py`'s explainability tab.

---

## Making it *your* standout project

This repo is a strong foundation — to make it genuinely yours for a
portfolio or interview, consider:

- Swap the synthetic generator for a real anonymized dataset (e.g. LendingClub)
  and note the change in modeling assumptions this requires.
- Add a **fairness audit tab** — check whether the model's error rates
  differ across `home_ownership` groups (a real concern in credit scoring).
- Add model persistence (`joblib`) plus a `train.py` script so retraining
  doesn't have to happen on every app cold start.
- Write up the project as a blog post / LinkedIn article walking through
  the modeling and explainability decisions — this is often what actually
  gets a project noticed, more than the code itself.

## License

MIT — see [LICENSE](LICENSE).
