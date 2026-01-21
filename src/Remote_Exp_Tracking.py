import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import dagshub
dagshub.init(repo_owner='akhil6july2003', repo_name='MLOps_Experiment_Tracking_Via_MLFlow', mlflow=True)


mlflow.set_tracking_uri("https://dagshub.com/akhil6july2003/MLOps_Experiment_Tracking_Via_MLFlow.mlflow")

wine = load_wine()
X = wine.data
y = wine.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=43)

# Define the parameters for RF model
max_depth = 7
n_estimators = 15

# mlflow.autolog()
mlflow.set_experiment('MLOps_Exp1')

with mlflow.start_run():
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=43)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    mlflow.log_metric("accuracy", accuracy)

    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("n_estimators", n_estimators)

    # Creating a confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True , fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')

    plt.savefig("Confusion_Matrix.png")

    mlflow.log_artifact("Confusion_Matrix.png")
    mlflow.log_artifact(__file__)

    mlflow.set_tags({"Author" : "Akhil", "Project" : "Wine Classification"})

    mlflow.sklearn.log_model(rf, "Random_Forest_Model")
