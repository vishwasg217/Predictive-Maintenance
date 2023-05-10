import pandas as pd

# create dataframe with scores
data = {
    "Model": ["Logistic Regression", "SVC", "Decision Tree", "Random Forest"],
    "Accuracy": [83.872917, 95.873263, 99.153932, 99.257533],
    "Precision": [86.385152, 94.675068, 99.097007, 98.959671],
    "Recall": [89.948119, 99.390402, 99.636835, 99.935149],
    "F1": [88.130639, 96.975449, 99.366188, 99.445018],
}
scores_df = pd.DataFrame(data)

# find model with highest F1 score
best_model_idx = scores_df["F1"].idxmax()
best_model_name = scores_df.loc[best_model_idx, "Model"]

# select best model from list of models
models = ["lr", "svc", "dt", "rf"]  # example list of models
best_model = models[best_model_idx]

print("Best model is:", best_model_name)
print("Selected model is:", best_model)
