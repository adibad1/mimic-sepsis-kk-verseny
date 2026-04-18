# MIMIC Sepsis Prediction: Federated Learning Hackathon

Welcome to the MIMIC Sepsis FL Hackathon! Your mission is to build a machine learning model that predicts the onset of sepsis 6 hours before clinical prediction.

Because we are using highly sensitive ICU data, you will never see the raw dataset. Instead, you will act as the Central Server in a Federated Learning (FL) setup, orchestrating the training of your model across 5 isolated hospital environments.

## 1. Connecting to Your Environment

You have been assigned a dedicated cloud Virtual Machine (Team VM). This is your workspace.

1. Open your local terminal (or Git Bash / PowerShell).
2. Connect via SSH using the credentials provided by the organizer:

   ```bash
   ssh hackadmin@<YOUR_TEAM_PUBLIC_IP>
   ```

3. Clone this repository onto your Team VM:
   ```bash
   git clone <YOUR_STARTER_REPO_URL>
   cd mimic-sepsis-starter
   pip install -r requirements.txt
   ```

## 2. How the Architecture Works

- **The Clients:** 5 Docker containers (one for each hospital partition) are continuously running in the background on a private network.
- **The Connection:** They are actively trying to connect to your Team VM on Port 8080. If your server isn't running, they will retry in a loop using an exponential backoff strategy.
- **Your Job:** Run `server.py`. Once started, the 5 waiting hospital clients will connect and begin the federated training rounds.

## 3. Suggested Workflow

1. **Understand the Data:** Read `DATA_DICTIONARY.md` to understand the 40 feature columns.
2. **Test Locally First:** Since you cannot see the real data, generate dummy data and run a local simulation on your laptop to debug your model architecture:
   ```bash
   python local_simulation/generate_dummy_data.py
   python local_simulation/simulate_local.py
   ```
3. **Modify the Baseline:** Edit the files in `baseline_model/` to build your custom pipeline.
4. **Train on Real Data:** On your Team VM, start your server to connect to the real hospitals:
   ```bash
   python server.py
   ```
   _Note: Your strategy is strictly locked to require exactly 5 clients. Do not change min_fit_clients=5!_

## 4. Rules and Constraints

- **Do not change the client limit:** In `server.py`, the strategy parameters must strictly require 5 clients. If you lower this, your model will not see all the data and your submission will be disqualified.
- **Missing Data:** Medical time-series data is notoriously messy. The hospital clients automatically impute missing values with the column median. If a column is entirely missing at a specific hospital, it is filled with 0.0. Your model architecture must be able to handle these imputed values robustly.
- **Resource Limits:** Your Team VM has limited CPU/RAM. Be mindful of batch sizes and model complexity.

## 5. Submission

When the time is up, ensure your `custom_strategy.py` saves your final weights as `final_model.pkl`. Package it for the organizers:

```bash
bash scripts/package_submission.sh
```

This script will copy your model into the `/home/hackadmin/submission/` directory. The organizers will automatically collect the models from this folder at the deadline to evaluate them against the hidden holdout set.
