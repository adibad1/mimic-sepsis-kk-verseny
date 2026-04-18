# DATA_DICTIONARY.md: Sepsis Prediction Features

This document describes the 40 clinical features provided in the hospital datasets. All features are time-series observations sampled at hourly intervals.

## 1. Primary Target

| Column          | Description                                                                        | Values                        |
| :-------------- | :--------------------------------------------------------------------------------- | :---------------------------- |
| **SepsisLabel** | The target variable. Indicates sepsis onset (6 hours prior to clinical diagnosis). | `0` (No Sepsis), `1` (Sepsis) |

---

## 2. Vital Signs

These are measured frequently at the bedside. They are often the first indicators of systemic distress.

| Column    | Unit        | Description                        |
| :-------- | :---------- | :--------------------------------- |
| **HR**    | bpm         | Heart Rate                         |
| **O2Sat** | %           | Pulse Oximetry (Oxygen Saturation) |
| **Temp**  | °C          | Core body temperature              |
| **SBP**   | mmHg        | Systolic Blood Pressure            |
| **MAP**   | mmHg        | Mean Arterial Pressure             |
| **DBP**   | mmHg        | Diastolic Blood Pressure           |
| **Resp**  | breaths/min | Respiration Rate                   |
| **EtCO2** | mmHg        | End-tidal Carbon Dioxide           |

---

## 3. Laboratory Values (Vital & Metabolic)

Note: Labs are sampled less frequently than vitals. Expect high rates of missingness (NaNs) which are auto-imputed at the hospital client.

### Blood Gases & Acid-Base

| Column         | Unit   | Description                                 |
| :------------- | :----- | :------------------------------------------ |
| **BaseExcess** | mmol/L | Measure of metabolic alkalosis/acidosis     |
| **HCO3**       | mmol/L | Bicarbonate levels                          |
| **FiO2**       | %      | Fraction of Inspired Oxygen                 |
| **pH**         | -      | Blood acidity                               |
| **PaCO2**      | mmHg   | Partial pressure of CO2 from arterial blood |
| **SaO2**       | %      | Oxygen saturation from arterial blood       |

### Electrolytes & Metabolic

| Column               | Unit   | Description                                      |
| :------------------- | :----- | :----------------------------------------------- |
| **AST**              | U/L    | Aspartate Aminotransferase (Liver function)      |
| **BUN**              | mg/dL  | Blood Urea Nitrogen (Kidney function)            |
| **Alkalinephos**     | U/L    | Alkaline Phosphatase                             |
| **Calcium**          | mg/dL  | Total Calcium                                    |
| **Chloride**         | mmol/L | Chloride levels                                  |
| **Creatinine**       | mg/dL  | Serum Creatinine (Kidney function)               |
| **Bilirubin_direct** | mg/dL  | Direct Bilirubin                                 |
| **Glucose**          | mg/dL  | Serum Glucose                                    |
| **Lactate**          | mmol/L | Lactic Acid (Indicator of tissue hypoxia/sepsis) |
| **Magnesium**        | mmol/L | Magnesium levels                                 |
| **Phosphate**        | mg/dL  | Phosphate levels                                 |
| **Potassium**        | mmol/L | Potassium levels                                 |
| **Bilirubin_total**  | mg/dL  | Total Bilirubin                                  |
| **TroponinI**        | ng/mL  | Cardiac Troponin I (Heart stress)                |

### Hematology

| Column         | Unit      | Description                              |
| :------------- | :-------- | :--------------------------------------- |
| **Hct**        | %         | Hematocrit (Volume of Red Blood Cells)   |
| **Hgb**        | g/dL      | Hemoglobin                               |
| **PTT**        | sec       | Partial Thromboplastin Time (Clotting)   |
| **WBC**        | cells/mcL | White Blood Cell Count (Immune response) |
| **Fibrinogen** | mg/dL     | Fibrinogen (Clotting)                    |
| **Platelets**  | cells/mcL | Platelet Count                           |

---

## 4. Demographics & Context

| Column          | Unit  | Description                                       |
| :-------------- | :---- | :------------------------------------------------ |
| **Age**         | Years | Patient age (capped at 90)                        |
| **Gender**      | -     | `0` (Female), `1` (Male)                          |
| **Unit1**       | -     | Administrative identifier for MICU (Medical ICU)  |
| **Unit2**       | -     | Administrative identifier for SICU (Surgical ICU) |
| **HospAdmTime** | Hours | Time between hospital admission and ICU admission |
| **ICULOS**      | Hours | Total ICU Length of Stay at current hour          |

---

## 5. Important Notes for Competitors

- **Missingness:** Medical data is sparse. If a specific laboratory value (e.g., Bilirubin) was not measured in a specific hour, the client-side `dataset_loader.py` has already performed **Forward Fill** and **Median Imputation**.
- **Scaling:** The raw values are provided. It is highly recommended to implement a `StandardScaler` or `MinMaxScaler` within your model pipeline to ensure gradients remain stable during Federated rounds.
- **Correlation:** Features like `MAP`, `SBP`, and `DBP` are highly correlated. Dimensionality reduction or careful feature selection may improve model robustness across different hospital environments.
