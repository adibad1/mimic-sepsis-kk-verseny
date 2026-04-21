import numpy as np
import cloudpickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
 
def get_model():
    def medical_feature_engineering(X):
        # Indexek a FEATURE_COLUMNS alapján
        hr    = X[:, [0]]   # Pulzus
        sbp   = X[:, [3]]   # Szisztolés vérnyomás
        map_  = X[:, [4]]   # MAP
        dbp   = X[:, [5]]   # Diasztolés vérnyomás
        resp  = X[:, [6]]   # Légzésszám
        temp  = X[:, [2]]   # Hőmérséklet
        bun   = X[:, [15]]  # BUN
        creat = X[:, [19]]  # Kreatinin
        lact  = X[:, [22]]  # Laktát
        wbc   = X[:, [31]]  # WBC
        age   = X[:, [34]]  # Kor
        iculos= X[:, [39]]  # ICULOS
 
        # --- Klinikai mutatók ---
        shock_index      = hr / (sbp + 1e-6)
        bun_creat_ratio  = bun / (creat + 1e-6)
        map_hr_ratio     = map_ / (hr + 1e-6)
        resp_hr_ratio    = resp / (hr + 1e-6)
        pulse_pressure   = sbp - dbp
        # --- Dinamikus rizikó indikátorok ---
        # Sirs-szerű jelek: magas pulzus + magas légzésszám vagy láz
        fever_flag       = (temp > 38.0).astype(float)
        tachycardia      = (hr > 90.0).astype(float)
        tachypnea        = (resp > 20.0).astype(float)
        # Súlyos állapot jelzők
        lactate_flag     = (lact > 2.0).astype(float)
        wbc_flag         = (wbc > 12.0).astype(float)
        icu_long_flag    = (iculos > 48).astype(float) # 48 óra után nő a kockázat
 
        # Összetett rizikó pontszám (több tünet együttese)
        composite_risk   = shock_index + lactate_flag + wbc_flag + tachycardia + tachypnea
 
        return np.hstack([
            X,
            shock_index, bun_creat_ratio,
            map_hr_ratio, resp_hr_ratio,
            pulse_pressure, lactate_flag,
            wbc_flag, icu_long_flag,
            fever_flag, composite_risk
        ])
 
    model = Pipeline([
        ("engineering", FunctionTransformer(medical_feature_engineering)),
        ("scaler",      StandardScaler()),
        ("clf",         LogisticRegression(
                            warm_start=True,
                            max_iter=1,
                            # Drasztikusan megemelt súly az FN csökkentése érdekében
                            class_weight={0: 1, 1: 45}, 
                            C=0.8, # Kicsit több szabadság a tanulásnak
                            solver="saga",
                            tol=1e-3,
                        )),
    ])
 
    # 40 alap + 10 új feature = 50 bemenet
    model.fit(np.zeros((5, 40)), np.array([0, 1, 0, 1, 0]))
    return model
 
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