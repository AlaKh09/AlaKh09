# Reditus

**Patient segmentation and 30-day readmission prediction for diabetic patients**

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Work%20in%20progress-yellow)

> **Project Reditus** is developed by **Team Mentalist**.
>
> ⚠️ **This is NOT the final version.** The README describes the current state of the notebook `mentalist_part2.ipynb`. The methodology, the models and **all the numbers below may change** before the final delivery.

---

## 📌 Overview

Hospital readmissions within 30 days are costly for hospitals and harmful for patients. Reditus works on the **Diabetes 130-US Hospitals (1999–2008)** dataset and follows the **CRISP-DM** methodology to answer two business questions:

| | Business objective | Data science task |
|---|---|---|
| **BO1: Classification** | Identify, at discharge, the patients most likely to be readmitted within 30 days, so follow-up can focus on them | Predict `readmitted == '<30'` (≈ 11 % positive class) |
| **BO2: Segmentation** | Adapt care pathways to different types of stays | Unsupervised clustering (K-Means) into 3 to 6 interpretable segments |

Segmentation describes **what kind of stay** it is; classification estimates **who is likely to come back**.

---

## 📊 Dataset

- **Source:** Diabetes 130-US Hospitals for Years 1999–2008 (UCI Machine Learning Repository), 10 years of clinical care across 130 US hospitals
- **Raw size:** 101,766 hospital stays × 50 variables (71,518 unique patients)
- **File used by the notebook:** `diabetic_data.csv` (demographics, diagnoses, lab tests, medication columns, admission/discharge info, target `readmitted`)
- **Target:** `readmitted_30 = 1` if `readmitted == '<30'`, otherwise `0`
- **Class imbalance:** 11,357 positives out of 101,766 stays (**11.16 %**), so accuracy is **not** used as the main metric

---

## 🔬 Methodology

The notebook `mentalist_part2.ipynb` follows the CRISP-DM phases:

| Phase | Content |
|---|---|
| **0. State of the art** | Positioning against the literature and a realistic performance benchmark (AUC around 0.6–0.7) |
| **1. Business understanding** | BO1 / BO2 translated into data science objectives |
| **2. Data understanding** | Structure, types, statistics, missing values (`?`), target, distributions, outliers, correlations, data quality issues |
| **3. Data preparation** | Missing values, identifiers, grouping of diagnoses and specialties, target creation, patient-level split, preprocessing, PCA |
| **4. Modeling** | K-Means clustering (with and without PCA); KNN and Decision Tree classifiers tuned with grouped cross-validation |
| **5. Evaluation** | Cluster quality, stability (ARI), profiling, χ² test; ROC/PR-AUC, threshold choice, feature importance, sensitivity analysis, fairness |
| **6. Deployment** | Deployment plan, simulation on 5 test stays, maintenance plan |

### Key data preparation choices
- **Dataset after preparation:** 101,766 stays × 46 columns, no missing values, no duplicated rows.
- **Missing values:** `?` converted to `NaN`; `weight` (≈ 96.9 % missing) dropped; `race`, diagnoses, `medical_specialty` (≈ 49 %) and `payer_code` (≈ 40 %) filled with `Unknown`; `max_glu_serum` and `A1Cresult` filled with `Not measured`, since a missing test can itself be informative.
- **Variables removed:** `encounter_id` (identifier), `examide` and `citoglipton` (constant), and the original `readmitted` column (target leakage). `patient_nbr` is kept only to group stays by patient and is never used as a feature.
- **High-cardinality variables:** `diag_1/2/3` grouped into 10 clinical categories (Circulatory, Respiratory, Digestive, Diabetes, …); `medical_specialty` reduced from 73 to 28 categories (specialties under 0.1 % merged into `Other`).
- **Outliers:** not removed, because extreme values (long stays, many visits) can be real clinical situations.
- **No data leakage:** train/test split **by patient** (`GroupShuffleSplit`, 80 / 20): 81,613 training stays and 20,153 test stays, **0 patients in both sets**. Preprocessing is fitted on the training set only.
- **Preprocessing:** `age` as an ordinal variable, One-Hot Encoding for categorical variables (including the three numeric ID codes for admission type, discharge disposition and admission source), `StandardScaler` for numeric variables → **222 features** for classification, **226 features** for segmentation.
- **Dimensionality reduction (segmentation):** PCA keeps **60 components (95.1 % of the variance)**.
- **Imbalance:** `class_weight="balanced"` for the decision tree; for KNN the imbalance is handled by choosing the decision threshold.

---

## 📈 Results (current version)

### BO1: Readmission prediction (test set)

| Model | ROC-AUC | PR-AUC | Readmissions captured in top 20 % |
|---|---|---|---|
| Dummy (baseline) | 0.500 | 0.107 | 20.3 % |
| KNN (k = 400) | 0.639 | 0.184 | 35.4 % |
| **Decision Tree (selected)** | **0.648** | **0.187** | **35.7 %** |

- Hyperparameters tuned with `GridSearchCV` on the training set only, using **patient-grouped cross-validation** (`GroupKFold`) and PR-AUC as the score. Selected tree: `max_depth = 10`, `min_samples_leaf = 100`.
- The model was chosen on cross-validated PR-AUC of the training set (tree 0.204 vs KNN 0.201); the test set was used only for measurement. The two models are almost tied.
- Flagging the top 20 % of stays catches about **36 % of readmissions** (about 1.8× random).
- **Decision threshold = 0.41** (maximizes F2 on out-of-fold predictions). On the test set: recall **0.807**, precision 0.131, F1 0.226, F2 0.398. This threshold flags roughly two thirds of the patients, so the alert is **not selective**; ranking patients by score and following the highest ones is more practical.
- A ROC-AUC of about 0.65 is consistent with published results on this kind of problem.
- **Most influential variables (tree importances):** prior inpatient visits (`number_inpatient`, 0.39) and the discharge type (`discharge_disposition_id`: home, deceased, rehabilitation transfer).
- **Sensitivity analysis:** removing deceased / hospice stays (1,958 training and 465 test stays) barely changes performance (ROC-AUC 0.648 → 0.645, PR-AUC 0.187 → 0.187). They are still kept in the current version.
- **Fairness (descriptive):** no recall gap above 0.10 by gender or race. ⚠️ **Alert for younger patients** (ages 20–40, recall about 0.68–0.70 vs 0.81 overall). Some groups are small (< 500 stays) and unstable.

### BO2: Patient segmentation (K-Means, k = 3, with PCA)

k = 2 had the best silhouette (0.083) but was rejected because BO2 requires 3 to 6 segments. Among k = 3 to 6, k = 3 had the best silhouette.

| Segment | Share | Profile | 30-day readmission |
|---|---|---|---|
| 0: Short stays, light profile | 32.0 % | 2.9 days, ~12 medications, younger patients, few prior visits | 8.0 % |
| 1: Long stays, heavy care | 22.9 % | 7.8 days, ~25 medications, many procedures | 12.8 % |
| 2: Older, multi-pathology, frequent users | 45.1 % | 3.7 days, ~14 medications, oldest patients, most prior inpatient/emergency visits | 12.6 % |

- **Stable:** mean ARI of 0.988 across 5 random seeds.
- **Weakly separated:** silhouette = 0.064, Davies-Bouldin = 3.33.
- **Risk differences:** the χ² test is significant (V de Cramér = 0.069, weak association). Segment 0 has a clearly lower risk, but **segments 1 and 2 are not distinguishable** in risk (p = 0.53); they differ by care profile.
- The segmentation is done on **stays**, not on patients, so one patient can appear in several segments. Segment names are proposals that still need clinical validation. Use the segments as a **descriptive tool** only.

### Limitations
- Moderate predictive power: many decisive factors (social support, adherence, post-discharge care) are not in the data.
- Low precision (≈ 13 %) at the chosen threshold; only two simple models (KNN, Decision Tree) are compared so far.
- Data from 1999–2008, with no clinical validation of the segments and no external validation.
- Lower recall for patients aged 20–40; fairness assessed descriptively only.
- Deceased / hospice stays are still in the data in this version.
- This system is a decision-support aid, not a replacement for clinical judgment.

### 🚧 Known work in progress
- Ensemble models (e.g. Random Forest, Gradient Boosting) and probability calibration are not yet part of the notebook.
- Some text cells in the notebook still show older values than the executed outputs; they will be aligned in the next version.
- Models are not exported yet: deployment is currently a simulation on 5 test stays.

---

## 📁 Project structure

```
Mentalist/
├── mentalist_part2.ipynb       # CRISP-DM notebook (modeling, evaluation, deployment)
├── diabetic_data.csv           # Raw dataset
├── diabetic_data_clean.csv     # Cleaned dataset (generated by the notebook)
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🚀 Getting started

**1. Clone the repository**
```bash
git clone https://github.com/AlaKh09/Mentalist.git
cd Mentalist
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```
Main libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `scipy`.

**3. Run the notebook**
```bash
jupyter notebook mentalist_part2.ipynb
```
Place `diabetic_data.csv` next to the notebook. The notebook was developed on Google Colab (it mounts Google Drive and downloads the cleaned CSV at the end), so remove or skip the `google.colab` cells when running locally.

> Note: the KNN grid search takes several minutes.

---

## 🛠️ Tech stack

Python · Pandas · NumPy · Matplotlib · Seaborn · scikit-learn · SciPy · Jupyter

---

## 👥 Team Mentalist

| Name | Role | GitHub | LinkedIn |
|---|---|---|---|
| Ala Khelil | | [AlaKh09](https://github.com/AlaKh09) | [Profile](https://www.linkedin.com/in/ala-khelil-3829b72a7/) |
| Yassine Smaoui | | | |
| Nour Toulgui | | | |
| Abdelhamid Ben Lakhal | | | |
| Ahmed Sellami | | | |

---

## 📚 References

- Strack B., DeShazo J. P., Gennings C., Olmo J. L., Ventura S., Cios K. J., Clore J. N. (2014). *Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records.* BioMed Research International, 2014, 781670. doi:10.1155/2014/781670
- Strack B. et al. (2014). *Diabetes 130-US Hospitals for Years 1999–2008* [dataset]. UCI Machine Learning Repository.

---

## 📄 License

See `LICENSE`.

---

## 📫 Contact

**Ala Khelil**
- 📧 Email: [alakhelil09@gmail.com](mailto:alakhelil09@gmail.com)
- 💼 LinkedIn: [linkedin.com/in/ala-khelil-3829b72a7](https://www.linkedin.com/in/ala-khelil-3829b72a7/)
- 🐙 GitHub: [AlaKh09](https://github.com/AlaKh09)
