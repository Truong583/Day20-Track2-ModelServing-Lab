import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
import joblib
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

EVAL_THRESHOLD = 0.70

def check_data_drift(y_train):
    """Bonus 5: Kiểm tra phân phối nhãn dữ liệu."""
    counts = y_train.value_counts(normalize=True)
    report = "\n--- DATA DISTRIBUTION REPORT ---\n"
    drift_warning = False
    for label, ratio in counts.items():
        report += f"Lớp {label}: {ratio:.2%}\n"
        if ratio < 0.10:
            drift_warning = True
            print(f"WARNING: Lớp {label} chỉ chiếm {ratio:.2%}, dưới ngưỡng 10%!")
    
    if drift_warning:
        report += "CẢNH BÁO: Dữ liệu bị lệch nhãn nghiêm trọng!\n"
    else:
        report += "Dữ liệu phân phối ổn định.\n"
    return report, counts.to_dict()

def train(
    params: dict,
    model_type: str = "random_forest",
    data_path: str = "data/train_phase1.csv",
    eval_path: str = "data/eval.csv",
) -> float:
    """
    Huấn luyện mô hình nâng cao với nhiều thuật toán và báo cáo chi tiết.
    """
    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    X_train = df_train.drop("target", axis=1)
    y_train = df_train["target"]
    X_eval = df_eval.drop("target", axis=1)
    y_eval = df_eval["target"]

    # Bonus 5: Kiểm tra drift
    drift_report_text, drift_metrics = check_data_drift(y_train)

    with mlflow.start_run():
        # Log model type
        mlflow.log_param("model_type", model_type)
        mlflow.log_params(params)

        # Bonus 2: Hỗ trợ nhiều thuật toán
        if model_type == "random_forest":
            model = RandomForestClassifier(**params, random_state=42)
        elif model_type == "gradient_boosting":
            model = GradientBoostingClassifier(**params, random_state=42)
        elif model_type == "logistic_regression":
            model = LogisticRegression(**params, random_state=42, max_iter=1000)
        else:
            model = RandomForestClassifier(**params, random_state=42)

        model.fit(X_train, y_train)

        # Dự đoán và tính toán metrics
        preds = model.predict(X_eval)
        acc = accuracy_score(y_eval, preds)
        f1 = f1_score(y_eval, preds, average="weighted")
        
        # Bonus 3: Tính Precision, Recall
        precision = precision_score(y_eval, preds, average="weighted")
        recall = recall_score(y_eval, preds, average="weighted")
        cls_report = classification_report(y_eval, preds)

        # Log metrics lên MLflow
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        
        # Log drift metrics
        for label, ratio in drift_metrics.items():
            mlflow.log_metric(f"drift_ratio_class_{label}", ratio)

        mlflow.sklearn.log_model(model, "model")

        # Bonus 3: Tạo báo cáo hiệu suất chi tiết
        os.makedirs("outputs", exist_ok=True)
        report_content = f"--- MODEL PERFORMANCE REPORT ---\n"
        report_content += f"Model Type: {model_type}\n"
        report_content += f"Accuracy: {acc:.4f}\n"
        report_content += f"F1 Score: {f1:.4f}\n"
        report_content += f"Precision: {precision:.4f}\n"
        report_content += f"Recall: {recall:.4f}\n"
        report_content += "\nDetailed Classification Report:\n"
        report_content += cls_report
        report_content += drift_report_text

        with open("outputs/report.txt", "w", encoding="utf-8") as f:
            f.write(report_content)

        with open("outputs/metrics.json", "w") as f:
            json.dump({
                "accuracy": acc, 
                "f1_score": f1, 
                "precision": precision, 
                "recall": recall,
                "label_distribution": drift_metrics
            }, f)

        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")
        
        print(f"[{model_type}] Accuracy: {acc:.4f} | F1: {f1:.4f}")
        return acc

if __name__ == "__main__":
    with open("params.yaml") as f:
        config = yaml.safe_load(f)
    
    # Đọc model_type từ params.yaml (mặc định là random_forest)
    m_type = config.get("model_type", "random_forest")
    
    # Tách params của model ra (bỏ qua field model_type)
    m_params = {k: v for k, v in config.items() if k != "model_type"}
    
    train(m_params, model_type=m_type)
