# Deployment Guide: GitHub + Streamlit Community Cloud

This project is fully built, tested, and committed to a local Git
repository. Publishing it requires **your own GitHub account
credentials**, which an AI assistant cannot supply on your behalf. Follow
the steps below (each takes 1-2 minutes).

---

## Step 1 - Authenticate GitHub CLI (one-time)

GitHub CLI (`gh` v2.100.0) has been downloaded and extracted to
`C:\Users\Lenovo\gh-cli\bin\gh.exe`, and this folder has been added to
your **user PATH**. Open a **new** terminal (so the updated PATH takes
effect) in the project folder and run:

```powershell
gh auth login --web
```

- Choose **GitHub.com** → **HTTPS** → **Yes** (authenticate Git with your
  GitHub credentials) → it will print a one-time code and open your
  browser. Paste the code, sign in, and authorize.

*(If `gh` is still not recognized after opening a new terminal, call it
by its full path: `& "C:\Users\Lenovo\gh-cli\bin\gh.exe" auth login --web`.)*

## Step 2 - Create the GitHub repository and push

From the project folder (`AI-Disease-Prediction`):

```powershell
gh repo create AI-Disease-Prediction --public --source=. --remote=origin --push
```

- Use `--private` instead of `--public` if you prefer a private repo.
- This creates the repo under your GitHub account, adds it as the
  `origin` remote, and pushes the already-committed code in one step.

**Alternative (no `gh` CLI):** create an empty repository manually on
github.com (do NOT initialize it with a README/license), then:

```powershell
git remote add origin https://github.com/<your-username>/AI-Disease-Prediction.git
git branch -M main
git push -u origin main
```

You will be prompted to sign in via the Git Credential Manager popup the
first time.

## Step 3 - Deploy to Streamlit Community Cloud

1. Go to **https://share.streamlit.io** and sign in with your GitHub
   account (this authorizes Streamlit to read your repos - your GitHub
   password/token is never shared with this project).
2. Click **"New app"**.
3. Select:
   - **Repository:** `<your-username>/AI-Disease-Prediction`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Click **"Deploy"**. Streamlit Cloud will automatically install
   `requirements.txt` and launch the app - typically live within 2-5
   minutes.

### Important pre-deployment notes

- The trained model artifacts (`models/*.pkl`, `*.json`) and the
  processed dataset (`data/processed/disease_dataset.csv`) are already
  committed to the repo, so the deployed app works immediately without
  needing to re-run `scripts/prepare_data.py` / `scripts/train_model.py`
  on the cloud.
- No secrets or API keys are required for the app to function (see
  `docs/ai_transparency.md` - no external LLM is called by default), so
  the Streamlit Cloud "Secrets" panel can be left empty.
- If you ever exceed Streamlit Cloud's free-tier resource limits, disable
  the t-SNE tab default computation or reduce `TARGET_RECORDS_PER_DISEASE`
  in `scripts/prepare_data.py` and retrain with a smaller dataset.

## Step 4 - Verify the live deployment

Once deployed, confirm:
- Home page loads with dataset/model metrics.
- AI Health Assistant chat responds to a test message (e.g. "I have
  fever and headache").
- Data Science Dashboard renders all charts.

## Keeping the deployed app updated

Any future `git push` to the `main` branch automatically redeploys the
Streamlit Cloud app (auto-redeploy is enabled by default for GitHub-linked
apps).
