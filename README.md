This is the complete, production-ready `README.md` for your hackathon starter kit. It combines the technical setup, the specific rules of Federated Learning, and the advanced pipeline instructions into a single, professional guide for the contestants.

---

# MIMIC Sepsis Prediction: Federated Learning Hackathon

Welcome to the MIMIC Sepsis FL Hackathon! Your mission is to build a machine learning model that predicts the onset of sepsis **6 hours before clinical prediction** using ICU time-series data.

Because we are using highly sensitive patient data from the MIMIC-III database, you will never see the raw dataset. Instead, you will act as the **Central Server** in a Federated Learning (FL) setup, orchestrating the training of your model across 5 isolated hospital environments.

---

## 1. Environment & Private Repository Setup

You have been assigned a dedicated cloud Virtual Machine (Team VM). To protect your competitive advantage and maintain version control, **you must work within your own private repository.**

### Step A: Connect to your VM

1. Open your local terminal (or Git Bash / PowerShell).
2. Connect via SSH using the credentials provided by the organizer:
   ```bash
   ssh hackadmin@<YOUR_TEAM_PUBLIC_IP>
   ```

### Step B: Create your Team Repository

1. Log into your GitHub/GitLab account and create a **New Private Repository**.
2. **Do not** initialize it with a README or .gitignore.
3. Invite the Organizer as a collaborator so your code can be reviewed.

### Step C: Initialize and Pull Starter Code

On your Team VM, run the following to link the starter code to your private repo:

```bash
# 1. Clone the starter code
git clone <STARTER_REPO_URL> mimic-sepsis
cd mimic-sepsis

# 2. Re-link to your private repo
git remote remove origin
git remote add origin <YOUR_PRIVATE_REPO_URL>
git push -u origin main

# 3. Setup Virtual Environment (Required)
python3 -m venv env
source env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 2. Architecture: The Invisible Hospitals

- **The Clients:** 5 Docker containers (representing 5 different hospital sites) are running in the background on a private network.
- **The Connection:** These clients are in a retry-loop, attempting to connect to your Team VM on **Port 8080**.
- **The Workflow:** When you run `server.py`, the "handshake" begins. Your model architecture is sent to the hospitals, trained locally on their private data, and the resulting weights are sent back to you for aggregation.

---

## 3. Advanced Feature Engineering & Pipelines

The starter kit uses a **Scikit-learn Pipeline**. This is your primary tool for manipulating data you cannot see.

### How to add new features:

Modify `baseline_model/model.py`. Any transformation you add to the `Pipeline` object will be executed **locally** at each hospital site. The current baseline includes:

1.  **Custom Ratios:** A `FunctionTransformer` that calculates the **Shock Index** ($HR/SBP$).
2.  **Interactions:** `PolynomialFeatures` to create multiplications between vital signs.
3.  **Normalization:** A `StandardScaler` that learns the mean/std of each specific hospital.
4.  **Dimensionality Reduction:** `PCA` to compress the feature space.

**Note:** Only the classifier weights (`clf`) are averaged by the server, but the entire pipeline logic is preserved in your final model.

---

## 4. Evaluation & Scoring

Sepsis prediction is a high-stakes clinical problem. We prioritize safety over simple accuracy.

### Live Feedback (During Training)

Every round, the hospital clients will evaluate your model on their **private validation sets**. Your server will print:

- **TOTAL_COST:** The primary metric for the leaderboard.
- **AUROC / F1-Score:** To help you tune model confidence.
- **Confusion Matrix (TP, TN, FP, FN):** To see exactly where the model is failing.

### Final Scoring (The Leaderboard)

At the deadline, organizers will pull your `final_model.pkl` and run it against a **secret holdout set**. The winner is determined by the lowest **Clinical Cost Score**:

$$\text{Cost} = \text{False Positives} + (\text{False Negatives} \times 5)$$

> **Clinical Rationale:** A False Positive is a "false alarm" that causes alert fatigue. A False Negative is a "missed diagnosis" that can lead to patient death. Therefore, our metric penalizes missed cases 5 times more heavily.

---

## 5. Suggested Workflow

1.  **Understand the Data:** Read `DATA_DICTIONARY.md` to learn about the 40 clinical features.
2.  **Simulation:** Test your code logic locally before running on the VM:
    ```bash
    python local_simulation/generate_dummy_data.py
    python local_simulation/simulate_local.py
    ```
3.  **Train on Real Data:** On your Team VM, ensure your `env` is active and start the server:
    ```bash
    python server.py
    ```

---

## 6. Rules and Constraints

- **The 5-Client Rule:** You must keep `min_fit_clients=5` and `min_available_clients=5` in `server.py`. If you lower this, your model will not see all the data and will be **disqualified**.
- **Imbalance:** Sepsis is rare (~2% of the data). Consider using class weights in your `LogisticRegression` or specialized sampling techniques within your pipeline.
- **Submission:** Your `custom_strategy.py` is configured to save `final_model.pkl`. To submit, run:
  ```bash
  bash scripts/package_submission.sh
  ```

---

**Organized by MIMIC Sepsis FL Hackathon Committee 2026**
