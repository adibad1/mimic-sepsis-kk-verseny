import numpy as np
import cloudpickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
 
 
def get_model():
    def medical_feature_engineering(X):
        # Alap indexek mentése [cite: 96]
        hr    = X[:, [0]]   # Pulzus
        sbp   = X[:, [3]]   # Vérnyomás
        map_  = X[:, [4]]   # MAP
        resp  = X[:, [6]]   # Légzés
        temp  = X[:, [2]]   # Hőmérséklet
        lact  = X[:, [22]]  # Laktát
        wbc   = X[:, [31]]  # Fehérvérsejt
        age   = X[:, [34]]  # Kor
        iculos= X[:, [39]]  # Eltöltött idő
 
        # 1. Klinikai indexek (Interakciók)
        shock_index = hr / (sbp + 1e-6) # [cite: 100]
        # NEWS (National Early Warning Score) elemek imitálása
        # A szepszisben a vérnyomás esik, a pulzus nő -> a hányadosuk négyzetesen is fontos lehet
        shock_index_sq = np.square(shock_index) 
        # 2. Nem-lineáris transzformációk (hogy a LogReg "görbéket" is lásson)
        # A laktát és a pulzus nem lineárisan veszélyes: a magas érték exponenciálisan rosszabb
        log_lactate = np.log1p(np.abs(lact))
        hr_squared  = np.square(hr / 100.0)
 
        # 3. Klinikai küszöbök (Flag-ek) 
        lactate_high = (lact > 2.0).astype(float)
        wbc_danger   = ((wbc > 12.0) | (wbc < 4.0)).astype(float) # A túl alacsony WBC is szepszis jel!
        hypotension  = (sbp < 90.0).astype(float)
 
        # 4. Időfaktor (A kockázat az idővel nem lineárisan nő)
        time_risk = np.sqrt(iculos + 1e-6)
 
        return np.hstack([
            X, 
            shock_index, shock_index_sq, log_lactate, 
            hr_squared, lactate_high, wbc_danger, 
            hypotension, time_risk
        ])
 
    model =Pipeline([
    ("engineering", FunctionTransformer(medical_feature_engineering)),
 
    # Neural networkhez KÖTELEZŐ
    ("scaler", StandardScaler()),
 
    ("clf", MLPClassifier(
        # ---- ARCHITEKTÚRA ----
        hidden_layer_sizes=(32, 16),
        activation="relu",
 
        # ---- OPTIMIZÁLÁS (FL‑BARÁT) ----
        solver="adam",
        learning_rate_init=1e-3,
        max_iter=1,              # FL step = 1 optimal lépés
        warm_start=True,         # globális súlyok továbbélése
 
        # ---- REGULARIZÁCIÓ (FP kontroll) ----
        alpha=5e-4,              # L2 – FP‑robbanás ellen
 
        # ---- STABILITÁS ----
        batch_size=128,
        shuffle=True,
        tol=1e-3,
 
        random_state=42,
    )),
])
 
    model.fit(np.zeros((5, 40)), np.array([0, 1, 0, 1, 0])) # [cite: 97]
    return model
 
# --- A get_parameters, set_parameters, save/load marad változatlan [cite: 98, 104] ---
def get_model_parameters(model):
    return [model.named_steps["clf"].coef_, model.named_steps["clf"].intercept_]
 
def set_model_parameters(model, parameters):
    model.named_steps["clf"].coef_      = parameters[0]
    model.named_steps["clf"].intercept_ = parameters[1]
 
def save_model(model, path="final_model.pkl"):
    with open(path, "wb") as f:
        cloudpickle.dump(model, f)
 
def load_model(path="final_model.pkl"):
    with open(path, "rb") as f:
        return cloudpickle.load(f)