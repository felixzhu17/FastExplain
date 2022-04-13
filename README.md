# FastExplain
> Fit Fast, Explain Fast

## Installing
```
pip install fast-explain
``` 
## About FastExplain
FastExplain provides an **out of the box** methodology for users to **quickly explore data**, with **flexibility to fine-tune** if needed.
- **Automated fitting** of machine learning models with hyperparameter search
- **Aesthetic display** of explanatory methods ready for reporting
- **Connected interface** for all models and related explanatory methods

## Quickstart
### Automated Fitting
``` python
from FastExplain import model_data
from FastExplain.datasets import load_titanic_data
df = load_titanic_data()
classification = model_data(df, 'Survived')
``` 
### Aesthetic Display
``` python
from FastExplain.explain import plot_one_way_analysis, plot_ale
plot_one_way_analysis(classification.m, "Age")
plot_ale(classification.m, "Age")
``` 

### Connected Interface
``` python
classification.plot_one_way_analysis("Age")
classification.plot_ale("Age")
classification.shap_dependence_plot("Age")
classification.error
``` 

## Models Supported
- Random Forest
- XGBoost
- Explainable Boosting Machine

## Exploratory Methods Supported:
- One-way Analysis
- Two-way Analysis
- Feature Importance Plots
- ALE Plots
- Explainable Boosting Methods
- SHAP Values
- Partial Dependence Plots
- Sensitivity Analysis
























