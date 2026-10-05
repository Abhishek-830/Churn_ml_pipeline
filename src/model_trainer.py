import os
import joblib
import pandas as pd

from src.logger import logger

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)


class ModelTrainer:

    def __init__(self):

        self.train_path = os.path.join(
            "artifacts",
            "train_transformed.csv"
        )

        self.test_path = os.path.join(
            "artifacts",
            "test_transformed.csv"
        )

        self.model_path = os.path.join(
            "artifacts",
            "model.pkl"
        )

        self.metrics_path = os.path.join(
            "artifacts",
            "model_metrics.csv"
        )

        self.confusion_matrix_path = os.path.join(
            "artifacts",
            "confusion_matrix.csv"
        )

    def initiate_model_training(self):

        logger.info("Model training started")

        print("Starting model training...")

        # Load transformed data
        train_data = pd.read_csv(
            self.train_path
        )

        test_data = pd.read_csv(
            self.test_path
        )

        logger.info(
            "Transformed train and test data loaded"
        )

        # Separate features and target
        X_train = train_data.drop(
            columns=["Exited"]
        )

        y_train = train_data["Exited"]

        X_test = test_data.drop(
            columns=["Exited"]
        )

        y_test = test_data["Exited"]

        # Convert DataFrame to NumPy
        # This keeps training and prediction format consistent
        X_train = X_train.values
        X_test = X_test.values

        # Models
        models = {

            "Logistic Regression":
                LogisticRegression(
                    max_iter=1000
                ),

            "Random Forest":
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=42
                )
        }

        best_model = None
        best_accuracy = 0
        best_model_name = None

        # Store evaluation results
        evaluation_results = []

        # Train models
        for model_name, model in models.items():

            logger.info(
                f"Training model: {model_name}"
            )

            print(
                f"\nTraining: {model_name}"
            )

            model.fit(
                X_train,
                y_train
            )

            predictions = model.predict(
                X_test
            )

            # Metrics
            accuracy = accuracy_score(
                y_test,
                predictions
            )

            precision = precision_score(
                y_test,
                predictions
            )

            recall = recall_score(
                y_test,
                predictions
            )

            f1 = f1_score(
                y_test,
                predictions
            )

            print(
                f"{model_name} Accuracy: "
                f"{accuracy:.4f}"
            )

            print(
                f"{model_name} Precision: "
                f"{precision:.4f}"
            )

            print(
                f"{model_name} Recall: "
                f"{recall:.4f}"
            )

            print(
                f"{model_name} F1 Score: "
                f"{f1:.4f}"
            )

            print("\nClassification Report:")

            print(
                classification_report(
                    y_test,
                    predictions
                )
            )

            logger.info(
                f"{model_name} - "
                f"Accuracy: {accuracy:.4f}, "
                f"Precision: {precision:.4f}, "
                f"Recall: {recall:.4f}, "
                f"F1: {f1:.4f}"
            )

            # Store results
            evaluation_results.append({
                "Model": model_name,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1_Score": f1
            })

            # Select best model
            if accuracy > best_accuracy:

                best_accuracy = accuracy
                best_model = model
                best_model_name = model_name
                best_predictions = predictions

        # -----------------------------
        # Save best model
        # -----------------------------

        joblib.dump(
            best_model,
            self.model_path
        )

        logger.info(
            f"Best model selected: "
            f"{best_model_name}"
        )

        logger.info(
            f"Model saved at: "
            f"{self.model_path}"
        )

        # -----------------------------
        # Save metrics
        # -----------------------------

        results_df = pd.DataFrame(
            evaluation_results
        )

        results_df.to_csv(
            self.metrics_path,
            index=False
        )

        # -----------------------------
        # Confusion Matrix
        # -----------------------------

        cm = confusion_matrix(
            y_test,
            best_predictions
        )

        cm_df = pd.DataFrame(
            cm,
            index=[
                "Actual_0",
                "Actual_1"
            ],
            columns=[
                "Predicted_0",
                "Predicted_1"
            ]
        )

        cm_df.to_csv(
            self.confusion_matrix_path
        )

        logger.info(
            "Evaluation metrics saved successfully"
        )

        logger.info(
            "Confusion matrix saved successfully"
        )

        # -----------------------------
        # Final output
        # -----------------------------

        print("\n==============================")
        print("BEST MODEL")
        print("==============================")

        print(
            "Model:",
            best_model_name
        )

        print(
            "Accuracy:",
            f"{best_accuracy:.4f}"
        )

        print(
            "Metrics saved:",
            self.metrics_path
        )

        print(
            "Confusion Matrix saved:",
            self.confusion_matrix_path
        )

        print(
            "Model saved:",
            self.model_path
        )

        print("==============================")

        return self.model_path


if __name__ == "__main__":

    trainer = ModelTrainer()

    trainer.initiate_model_training()
