# 🛡️ OpenEnv: Fraud Detection & Transaction Monitoring

An interactive, real-world LLM evaluation environment built on the **OpenEnv** specification. 

This repository simulates the day-to-day workflow of a Bank Fraud Investigator (Trust & Safety). It tests an AI agent's ability to reason over tabular financial data, recognize chronological and geographical anomalies, and make deterministic formatting decisions strictly adhering to defined JSON schemas.

---

## 📖 Table of Contents
1. [Overview & Motivation](#overview--motivation)
2. [Environment Specifications](#environment-specifications)
3. [Task Difficulties](#task-difficulties)
4. [Repository Structure](#repository-structure)
5. [Setup & Installation](#setup--installation)
6. [Usage & Execution](#usage--execution)
7. [Evaluation Logging](#evaluation-logging)

---

## 🎯 Overview & Motivation
Standard LLM benchmarks often rely on multiple-choice questions or toy grid-world games. This environment forces the agent to perform a concrete corporate task: **Transaction Monitoring**. 

The agent is provided with a user's transaction history and a newly pending transaction. It must analyze the velocity, geographical metadata, and monetary amounts to decide whether to approve the transaction, decline it, or freeze the account entirely. 

---

## ⚙️ Environment Specifications

### Observation Space
The agent receives a strict JSON object (defined via Pydantic) representing the current state of the user's account:
* `pending_transaction`: The transaction currently awaiting approval (Amount, Merchant, Location, Time since last transaction).
* `user_account_history`: A chronological list of past transactions.
* `account_age_days`: The age of the account in days.

### Action Space
The agent must reply with a strictly formatted JSON object:
* `decision` (Literal): Must be exactly `"approve"`, `"decline"`, or `"freeze_account"`.
* `reasoning` (String): A brief explanation of the analytical logic used to reach the decision.

### Reward Function
The environment provides a meaningful reward signal over the trajectory (0.0 to 1.0):
* **+1.0 (Perfect):** The agent matches the exact ground-truth decision.
* **+0.5 (Partial):** The agent identifies fraud but selects the wrong mitigation severity (e.g., declining a transaction when a full account freeze was necessary).
* **-1.0 (Penalty):** False Positives (blocking a legitimate user) or False Negatives (approving blatant fraud).

---

## 📊 Task Difficulties

The environment features three distinct programmatic tasks with deterministic graders:

| Level | Scenario | Ground Truth | Description |
| :--- | :--- | :--- | :--- |
| **Easy** | Geo-Anomaly | `freeze_account` | A massive transaction occurs in a high-risk foreign location just minutes after a local domestic transaction. |
| **Medium** | Velocity Attack | `decline` | A series of rapid, low-dollar transactions (card testing) is immediately followed by a large, high-value purchase attempt. |
| **Hard** | Subtle Takeover | `decline` | Transaction amounts appear normal, but there is a subtle, unusual shift in the merchant category and IP location. |

---

## 🗂️ Repository Structure

```text
.
├── models.py           # Pydantic schemas for Observations and Actions
├── env.py              # Core OpenEnv logic (reset, step, state, graders)
├── app.py              # FastAPI web server for Hugging Face Space deployment
├── inference.py        # Baseline evaluation script with strict stdout logging
├── openenv.yaml        # OpenEnv metadata and entrypoint configuration
├── requirements.txt    # Python package dependencies
├── Dockerfile          # Containerization for deployment
└── README.md           # Project documentation

🚀 Setup & Installation
Option 1: Docker (Recommended)
This project is containerized for easy deployment to Hugging Face Spaces or local execution.

Bash
# Build the image
docker build -t openenv-fraud-detection .

# Run the container (exposes the FastAPI server on port 7860)
docker run -p 7860:7860 openenv-fraud-detection
Option 2: Local Python Environment
Ensure you have Python 3.10+ installed.

Bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt

💻 Usage & Execution
1. Running the Inference Baseline
To run the evaluation script (inference.py), you must configure three environment variables. The script uses the standard OpenAI Python client to interface with any compatible LLM endpoint.

Required Environment Variables:

API_BASE_URL: The API endpoint for the LLM.

MODEL_NAME: The model identifier to use for inference.

HF_TOKEN: Your Hugging Face / API key.

Execution:

Bash
export API_BASE_URL="[https://api.your-provider.com/v1](https://api.your-provider.com/v1)"
export MODEL_NAME="your-chosen-model"
export HF_TOKEN="your-api-key"

python inference.py
2. Running the API Server
The repository includes a FastAPI server (app.py) to satisfy Hugging Face Space deployment requirements. It provides a health check at the root \, and exposes the /reset, /step, and /state endpoints.

Bash
uvicorn app:app --host 0.0.0.0 --port 7860

📝 Evaluation Logging
The inference.py script utilizes strict stdout formatting required by the OpenEnv validation suite. Evaluators parsing the logs will find them formatted exactly as follows:

[START] task=<task_name>

[STEP] action=<json_string> reward=<float>

[END] task=<task_name> score=<float>

Example Output:

Plaintext
[START] task=easy
[STEP] action={"decision": "freeze_account", "reasoning": "Impossible travel time between NY and Moscow in 5 mins."} reward=1.0
[END] task=easy score=1.0