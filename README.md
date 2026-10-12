# Reditus

**Patient segmentation and 30-day readmission prediction for diabetic patients**

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![Status](https://img.shields.io/badge/Status-In%20progress-yellow)

> **Project Reditus** is developed by **Team Mentalist**.
> ⚠️ The results below come from the current version of the notebook. This is **not the final version** and the numbers may change.

---

## 📌 Overview

Hospital readmissions within 30 days are costly for hospitals and harmful for patients. Reditus works on the **Diabetes 130-US Hospitals (1999–2008)** dataset and follows the **CRISP-DM** methodology to answer two business questions:

| | Business objective | Data science task |
|---|---|---|
| **BO1: Classification** | Identify, at discharge, the patients most likely to be readmitted within 30 days, so follow-up can focus on them | Predict `readmitted == '<30'` (≈ 11 % positive class) |
| **BO2: Segmentation** | Adapt care pathways to different types of stays | Unsupervised clustering (K-Means) into 3 to 6 interpretable segments |

Segmentation explains **who** the patients are; classification predicts **who will come back**.

---

## 📊 Dataset

- **Source:** Diabetes 130-US Hospitals for Years 1999–2008 (UCI Machine Learning Repository), 10 years of clinical care across 130 US hospitals
- **Raw size:** 101,766 hospital stays × 50 variables
- **Files used:**
  - `diabetic_data.csv`: the stays (demographics, diagnoses, lab tests, 23 medication columns, admission/discharge info, target `readmitted`)
  - `IDS_mapping.csv`: dictionary for admission type, discharge disposition and admission source IDs
- **Target:** `readmitted_30 = 1` if `readmitted == '<30'`, otherwise `0`
- **Class imbalance:** 11.4 % positives (about 1 for every 8 negatives), so accuracy is **not** used as the main metric

---

## 🔬 Methodology

The notebook `Reditus_notebook_corrige.ipynb` covers every CRISP-DM phase:

| Phase | Content |
|---|---|
| **0. State of the art** | Positioning against the literature and a realistic performance benchmark |
| **1. Business understanding** | BO1 / BO2 translated into data science objectives |
| **2. Data understanding** | Structure, quality, target, distributions, outliers, correlations, relations with the target |
| **3. Data preparation** | Missing values, cohort definition, clinical groupings, feature engineering, outliers, imbalance handling, patient-level split, preprocessing pipelines, PCA |
| **4. Modeling** | K-Means clustering; 6 classifiers tuned with grouped cross-validation |
| **5. Evaluation** | Cluster quality and stability (ARI), profiling, statistical test; confusion matrices, threshold choice, calibration, interpretation, fairness, ablations |
| **6. Deployment** | Single serialized pipeline, deployment simulation, maintenance plan |

### Key data preparation choices
- **Cohort:** 101,766 → **99,340 stays** (69,987 patients). Stays ending in death or hospice were removed because those patients cannot be readmitted; 3 stays with invalid gender were also removed.
- **No data leakage:** train/test split **by patient** (no patient appears in both sets), stratified on the target, with preprocessing inside the pipelines.
- **Missing values:** `weight` (≈ 97 % missing) dropped; other gaps kept as `Unknown` / `Not measured`, since a missing test can itself be informative.
- **Feature engineering:** aggregated medication and diagnosis features; `log1p` on skewed utilization counts.
- **Imbalance:** 4 strategies compared **inside** cross-validation (none, `class_weight`, SMOTE, undersampling). `class_weight='balanced'` was kept: no measurable ranking loss, no synthetic data.
- **Segmentation space:** 16 interpretable numeric/binary variables → **11 PCA components (86.4 % variance)**.

---

## 📈 Results (current version)

### BO1: Readmission prediction (test set)

| Model | ROC-AUC | PR-AUC | Readmissions captured in top 20 % |
|---|---|---|---|
| Dummy (baseline) | 0.500 | 0.114 | 20.0 % |
| Logistic Regression | 0.660 | 0.208 | 37.4 % |
| KNN | 0.647 | 0.197 | 35.3 % |
| Decision Tree | 0.649 | 0.206 | 35.9 % |
| Random Forest | 0.664 | 0.219 | 38.8 % |
| **Gradient Boosting (final model)** | **0.665** | **0.224** | **38.4 %** |

- The final model was selected by cross-validated PR-AUC on the training set; the test set was used once, for measurement only.
- Flagging the top 20 % of stays catches **38.4 % of readmissions** (about 1.9× random), with an alert precision of about 21–22 %.
- A ROC-AUC of about 0.665 is consistent with published results on this kind of problem.
- The most influential variables are **prior inpatient visits** (`number_inpatient`) and the **discharge type**.
- Scores are calibrated (isotonic regression) so they can be shown as a risk percentage.

### BO2: Patient segmentation (K-Means, k = 3)

| Segment | Share | Profile | 30-day readmission |
|---|---|---|---|
| 0: No antidiabetic treatment | 22.8 % | 4.1 days, no active antidiabetic drug | 9.9 % |
| 1: Long stays, polymedication, adjusted treatment | 32.9 % | 5.8 days, about 21 medications | 13.4 % |
| 2: Short stays, stable treatment | 44.3 % | 3.5 days, stable treatment | 10.7 % |

The segments are readable and large enough to be useful, but statistically **weakly separated** (silhouette ≈ 0.12, Cramér's V = 0.045) and **stability is not yet demonstrated** (bootstrap ARI ≈ 0.65). They should be used as a **descriptive tool** only.

### Limitations
- Moderate predictive power: many decisive factors (social support, adherence, post-discharge care) are not in the data.
- Data from 1999–2008, with no clinical validation of the segments yet.
- Fairness was assessed descriptively only.
- This system is a decision-support aid, not a replacement for clinical judgment.

---

## 📁 Project structure

```
Reditus/
├── Reditus_notebook_corrige.ipynb   # Full CRISP-DM notebook
├── diabetic_data.csv                # Raw dataset
├── IDS_mapping.csv                  # ID dictionary (admission / discharge)
├── diabetic_data_clean.csv          # Cleaned dataset (generated by the notebook)
├── reditus_classifier.joblib        # Serialized classification pipeline (generated)
├── reditus_segmenter.joblib         # Serialized segmentation pipeline (generated)
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
Main libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `imbalanced-learn`, `scipy`, `joblib`.

**3. Run the notebook**
```bash
jupyter notebook Reditus_notebook_corrige.ipynb
```
The notebook looks for `diabetic_data.csv` next to itself (or in a `data/` folder). It also runs on Google Colab, where it asks you to upload the file.

---

## 🛠️ Tech stack

Python · Pandas · NumPy · Matplotlib · Seaborn · scikit-learn · imbalanced-learn · Jupyter

---

## 👥 Team Mentalist

| Name | Role | GitHub | LinkedIn |
|---|---|---|---|
| Ala Khelil | | [AlaKh09](https://github.com/AlaKh09) | [Profile](https://www.linkedin.com/in/ala-khelil-3829b72a7/) |
|Yassine Smaoui | | | |
| Nour toulgui | | | |
| abdelhmaid ben lakhal| | | |
| Ahmed sellami| | | |

---

## 📚 References

- Strack B., DeShazo J. P., Gennings C., Olmo J. L., Ventura S., Cios K. J., Clore J. N. (2014). *Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records.* BioMed Research International, 2014, 781670. doi:10.1155/2014/781670
- Strack B. et al. (2014). *Diabetes 130-US Hospitals for Years 1999–2008* [dataset]. UCI Machine Learning Repository.

---

## 📄 License

[Add your license here, e.g. MIT. See `LICENSE`.]

---

## 📫 Contact

**Ala Khelil**
- 📧 Email: [alakhelil09@gmail.com](mailto:alakhelil09@gmail.com)
- 💼 LinkedIn: [linkedin.com/in/ala-khelil-3829b72a7](https://www.linkedin.com/in/ala-khelil-3829b72a7/)
- 🐙 GitHub: [AlaKh09](https://github.com/AlaKh09)
