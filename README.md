# 🛡️ AIOps CI/CD & Cloud Failure Detection, Predictive Maintenance & Automated Remediation

End-to-end modular Python platform that ingests CI/CD and cloud infrastructure telemetry, detects anomalies, predicts failure probability over the next 30 minutes, isolates root causes via **SHAP & Decision-Tree Paths**, and executes human-approved remediations via the **GitHub Actions REST API**.

## 🚀 Launch in GitHub Codespaces

This repository includes a pre-configured `.devcontainer/devcontainer.json` specification for **GitHub Codespaces**:
1. Click **Code** → **Codespaces** → **Create codespace on main**.
2. Codespaces will automatically:
   - Provision a Python 3.11 container with the GitHub CLI (`gh`) pre-installed.
   - Install all dependencies from `requirements.txt`.
   - Run `python data_loader.py && python train.py` to generate the 80/20 dataset split and train the Neural Network, AutoML, 30-Minute Horizon Forecaster, and SHAP RCA models.
   - Launch `streamlit run app.py` on port `8501` and open the live interactive dashboard preview.

## 📂 Repository Structure

| File | Responsibility |
| :--- | :--- |
| `data_loader.py` | Downloads open-source system/build logs, detects Google Drive mounts (`/content/drive/MyDrive/devops_aiops/data/`), scrubs secrets/UUIDs/IPs, extracts TF-IDF + 14 numerical telemetry features, and creates a stratified 80/20 Train/Test split. |
| `train.py` | Benchmarks a Deep Neural Network (`TensorFlow/Keras` + NumPy Adam MLP) against AutoML (`FLAML` / `LightGBM+Optuna` / `Scikit-Learn`), trains the 30-minute failure risk forecaster, fits the Explainable RCA engine (`SHAP` + Decision-Tree rules), and serializes `.h5`, `.pkl`, and `.json` artifacts. |
| `connector.py` | Polls live GitHub Actions workflow runs (`/repos/{owner}/{repo}/actions/runs`), verifies `X-Hub-Signature-256` webhooks, and streams live or simulated telemetry trajectories. |
| `remediation.py` | Executes one-click GitHub REST API remediations for Pipeline Timeouts, Dependency Conflicts, Cloud OOM/Disk Exhaustion, and Expired Secrets, while logging operator approvals to SQLite (`hitl_remediation_audit.db`) and CSV (`hitl_feedback_retraining.csv`). |
| `app.py` | Interactive Streamlit dashboard featuring fleet health KPIs, real-time failure alerts, a 30-minute risk gauge, SHAP attribution charts, and human-in-the-loop approval buttons. |
| `.devcontainer/devcontainer.json` | GitHub Codespaces configuration automating package installation, model training, and port `8501` forwarding. |
| `deploy_to_github.py` | Utility script to create a GitHub repository and push all files directly from Google Drive or local storage using a GitHub Personal Access Token. |
