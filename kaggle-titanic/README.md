# Kaggle Titanic — Survival Prediction

Predicting which passengers survived the Titanic ([Kaggle competition](https://www.kaggle.com/competitions/titanic)) with scikit-learn, reaching **~85.0% cross-validated accuracy**.

---

## Dataset
📂 Download `train.csv` and `test.csv` from: https://www.kaggle.com/competitions/titanic/data
and place them in a `data/` folder next to the notebooks.

---

## Notebooks

**`titanic.ipynb` — v1, first baseline**
- Logistic Regression on Pclass, Sex, Age, Fare (+ SibSp, Parch, FamilySize)
- Manual cleaning, then refactored into a `Pipeline` + `ColumnTransformer`

**`titanic_v2.ipynb` — v2, improved solution**
- **Evaluation:** repeated stratified 5-fold CV (15 rounds) instead of a single noisy split
- **Feature engineering:** Title from name (Mr / Mrs / Miss / Master / Rare), fare per person, ticket group size, family size, travelling alone, cabin known
- **Model comparison:** Logistic Regression, SVC, Random Forest, Gradient Boosting — all inside the same pipeline
- **Tuning:** `GridSearchCV` over Gradient Boosting hyperparameters
- **Woman-child group rule:** families tended to share a fate; boys whose family's women/children all survived are predicted to survive, women whose family's women/children all died are predicted to die (computed leak-free, from training folds only)

---

## Results (cross-validated accuracy)

| Approach | CV accuracy |
|---|---|
| Gender rule (female survives) | 78.7% |
| v1 Logistic Regression | 78.9% |
| Logistic Regression + engineered features | 82.5% |
| Gradient Boosting (tuned) | 84.5% |
| **+ woman-child group rule** | **85.0%** |

Final predictions: `submission_v2.csv`.

---

## Run it

```bash
uv sync
uv run jupyter notebook titanic_v2.ipynb
```

**Stack:** Python 3.12, pandas, scikit-learn, matplotlib
