Social Support AI — End-to-End Workflow Automation

📌 Overview

This project automates the government social support application workflow using AI/ML + Orchestration.
It reduces processing time from weeks to a few minutes by automating:
	•	Data ingestion (CSV, PDFs, IDs, JSON credit reports)
	•	Canonical profile creation (structured applicant profile)
	•	Validation (via Pydantic rules)
	•	Feature export for training
	•	ML model prediction (XGBoost)
	•	LLM explanation (Mistral via Ollama)
	•	Interactive UI (Streamlit)

The prototype is fully local and works end-to-end with synthetic datasets.
🏗️ Architecture
Workflow:
	1.	Streamlit UI → user starts orchestrator, views logs, fetches applicant profiles, and runs explanations.
	2.	Orchestrator → coordinates pipeline steps.
	3.	MongoDB → stores raw, canonical, validated profiles and predictions.
	4.	Validation → enforces schema + rules using Pydantic v2.
	5.	XGBoost model → predicts probability of approval.
	6.	Mistral LLM → generates human-readable decision summary & explanation.

📂 Project Structure
social-support-ai/
│── data/                # synthetic dataset
│── models/              # trained ML model (XGBoost joblib)
│── output/              # pipeline outputs
│── src/                 # source code
│   ├── orchestrator.py  # orchestrates entire workflow
│   ├── streamlit_app.py # Streamlit frontend
│   ├── upload_raw_to_mongo.py
│   ├── create_canonical_profiles.py
│   ├── validation/      # schema + validation scripts
│   ├── train_xgb.py     # ML training
│   ├── llm_explainer_mistral.py
│── requirements.txt     # dependencies
│── README.md            # this file

🧪 Example Workflow
	1.	Run orchestrator → Upload raw → Canonicalize → Validate → Export → Train.
	2.	Check logs in Streamlit (bottom of page).
	3.	Fetch applicant profile (e.g., APP100041).
	4.	Run LLM Explainer → See decision summary, evidence, and final decision.

⸻

✅ Features Solved from Problem Statement
	•	Manual Data Gathering → Automated ingestion of PDFs, JSON, CSV, and IDs.
	•	Semi-Automated Validation → Strict schema validation via Pydantic v2.
	•	Inconsistent Information → Canonical profiles unify all sources.
	•	Time-Consuming Reviews → Orchestrator pipeline finishes in minutes.
	•	Subjective Decisions → Standardized rules + ML probability + LLM explanation.

⸻

👤 Author

Created by sudhanshshaw-cmd
