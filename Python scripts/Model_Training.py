import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_data(script_location):
    input_path = script_location / 'Data/Processed/Data_output.csv'
    df = pd.read_csv(input_path)
    df = df[df['basket_size'].notna()]
    bins = [-1, 4, 7, 14, np.inf]
    labels = [0, 1, 2, 3]
    df["next_order_class"] = pd.cut(df["days_until_next_order"],bins=bins,labels=labels).astype(int)
    return df

def more_data(df):
    df['ordering_speed_change'] = df['average_of_3_last_orders'] - df['average_days_between_orders']
    df['is_weekend'] = df['order_dow'].isin([0,5,6]).astype(int)
    df['basket_change'] = df['basket_size'] - df['average_basket_size']
    return df

def split_data(df):
    X = df.drop(columns = ['days_until_next_order','order_id','next_order_class'])
    idx = X.groupby('user_id')['Previous_orders_number'].idxmax()
    X_train = X.drop(index=idx)
    X_test = X.loc[idx]
    y = df['next_order_class']
    y_train = y.drop(index=idx)
    y_test = y.loc[idx]
    return X_train, X_test, y_train, y_test

def model_test(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

def feature_importance(model,X_train):
    importance = model.feature_importances_

    feature_importance = pd.DataFrame({"feature": X_train.columns,"importance": importance})
    feature_importance = feature_importance.sort_values(by="importance",ascending=False)
    print(feature_importance)
    return feature_importance


def test_model(X_train, y_train):
    model = XGBClassifier(objective="multi:softprob", eval_metric="mlogloss", tree_method="hist", num_class = 4)
    params = {
        "max_depth": [5,6,7],
        "learning_rate": [0.03,0.05,0.1],
        "n_estimators": [300,500,700],
        "min_child_weight": [5,10,20],
        "subsample": [0.7,0.8,0.9],
        "colsample_bytree": [0.7,0.8,0.9],
        "gamma": [0,0.1,0.5],
        "reg_alpha": [0,0.1,1],
        "reg_lambda": [1,5,10]}
    search = RandomizedSearchCV(
        model,
        params,
        n_iter=40,
        cv=2,
        scoring = 'f1_macro',
        verbose=2,
        n_jobs=2)
    search.fit(X_train, y_train)
    print(search.best_params_)
    print(search.best_score_)

def correlation_heatmap(df,script_location):
    important_features = [
        "days_since_prior_order",
        "min_interval",
        "average_of_3_last_orders",
        "average_days_between_orders",
        "Previous_orders_number",
        "interval_std",
        "max_interval",
        "reorder_percentage",
        "basket_size",
        "basket_ratio",
        "days_until_next_order"]
    corr = df.corr(numeric_only=True)

    plt.figure(figsize=(12,10))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(script_location / "Figures/correlation_heatmap.png", dpi=300, bbox_inches="tight")
    plt.show()

def feature_importance_graph(X_train, model, script_location):
    importance = pd.DataFrame({"feature": X_train.columns, "importance": model.feature_importances_})
    importance = importance.sort_values("importance",ascending=False)

    plt.figure(figsize=(10,6))
    sns.barplot(data=importance.head(10), x="importance", y="feature")
    plt.title("Top 10 Feature Importance")
    plt.tight_layout()
    plt.savefig(script_location / "Figures/feature_importance_graph.png", dpi=300, bbox_inches="tight")
    plt.show()

def confusion_matrix_graph(model, X_test, y_test, script_location):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, normalize="true")

    plt.figure(figsize=(7,5))
    sns.heatmap(cm, annot=True, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig(script_location / "Figures/confusion_matrix_graph.png", dpi=300, bbox_inches="tight")
    plt.show()



def main():
    script_location = Path(__file__).absolute().parent.parent
    df = load_data(script_location)
    X_train, X_test, y_train, y_test = split_data(df)
    model = XGBClassifier(objective="multi:softprob", eval_metric="mlogloss", tree_method="hist",
                    subsample = 0.8,
                    reg_lambda = 10,
                    reg_alpha = 0.9,
                    max_depth=6,
                    num_class=4,
                    n_estimators = 350,
                    colsample_bytree = 0.8,
                    learning_rate=0.035,
                    min_child_weight=10)
    model.fit(X_train, y_train)
    model.save_model(script_location = Path(__file__).absolute().parent.parent / "Models/xgboost_model.json")
    confusion_matrix_graph(model, X_test, y_test)
    feature_importance_graph(X_train, model)
    correlation_heatmap(df)

# currently not used functions: more_data(), test_model
if __name__ == "__main__":
    main()






