import numpy as np
import cloudpickle
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
 
def get_model():
    def medical_feature_engineering(X):
        # Alap indexek kinyerése
        hr    = X[:, [0]]   # Pulzus
        sbp   = X[:, [3]]   # Vérnyomás
        map_  = X[:, [4]]   # MAP
        resp  = X[:, [6]]   # Légzés
        temp  = X[:, [2]]   # Hőmérséklet
        lact  = X[:, [22]]  # Laktát
        wbc   = X[:, [31]]  # Fehérvérsejt
        iculos= X[:, [39]]  # Eltöltött idő
 
        # Feature Engineering (az eddigi legjobb orvosi logikák)
        shock_index = hr / (sbp + 1e-6)
        shock_index_sq = np.square(shock_index)
        log_lactate = np.log1p(np.abs(lact))
        hr_squared  = np.square(hr / 100.0)
        lactate_high = (lact > 2.0).astype(float)
        wbc_danger   = ((wbc > 12.0) | (wbc < 4.0)).astype(float)
        hypotension  = (sbp < 90.0).astype(float)
        time_risk = np.sqrt(iculos + 1e-6)
 
        return np.hstack([
            X, 
            shock_index, shock_index_sq, log_lactate, 
            hr_squared, lactate_high, wbc_danger, 
            hypotension, time_risk
        ])
 
    model = Pipeline([
        ("engineering", FunctionTransformer(medical_feature_engineering)),
        ("scaler",      StandardScaler()),
        ("clf",         MLPClassifier(
                            hidden_layer_sizes=(32, 16),
                            activation='relu',
                            solver='adam',
                            max_iter=1,          # FL körönként 1 iteráció
                            warm_start=True,     # Kell a folyamatos tanuláshoz
                            alpha=0.01,          # Regularizáció
                            random_state=42,
                            batch_size=128
                        )),
    ])
 
    # Inicializálás dummy adatokkal (40 bemeneti változó)
    model.fit(np.zeros((5, 40)), np.array([0, 1, 0, 1, 0]))
    return model
 
# --- JAVÍTOTT PARAMÉTERKEZELÉS MLP-HEZ ---
 
def get_model_parameters(model):
    clf = model.named_steps["clf"]
    # Az MLP-nél a coefs_ és intercepts_ listák, ezeket fűzzük össze
    return clf.coefs_ + clf.intercepts_
 
def set_model_parameters(model, parameters):
    clf = model.named_steps["clf"]
    # Meghatározzuk a rétegek számát (hidden layers + output layer)
    # hidden_layer_sizes=(32, 16) -> ez 2 hidden + 1 output = 3 réteg
    n_layers = len(clf.hidden_layer_sizes) + 1
    # Szétosztjuk a kapott listát súlyokra és eltolásokra
    clf.coefs_ = parameters[:n_layers]
    clf.intercepts_ = parameters[n_layers:]
 
def save_model(model, path="final_model.pkl"):
    with open(path, "wb") as f:
        cloudpickle.dump(model, f)
 
def load_model(path="final_model.pkl"):
    with open(path, "rb") as f:
        return cloudpickle.load(f)