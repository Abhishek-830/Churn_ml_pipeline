from src.data_ingestion import DataIngestion
from src.data_transformation import DataTransformation
from src.model_trainer import ModelTrainer
from src.predict import PredictPipeline


def run_pipeline():

    print("\n========== PIPELINE STARTED ==========\n")

    # -----------------------------
    # 1. Data Ingestion
    # -----------------------------

    ingestion = DataIngestion()

    raw_data_path = (
        ingestion.initiate_data_ingestion()
    )

    # -----------------------------
    # 2. Data Transformation
    # -----------------------------

    transformation = DataTransformation()

    train_path, test_path = (
        transformation.initiate_data_transformation()
    )

    # -----------------------------
    # 3. Model Training
    # -----------------------------

    trainer = ModelTrainer()

    model_path = (
        trainer.initiate_model_training()
    )

    # -----------------------------
    # 4. Prediction
    # -----------------------------

    customer = {
        "CreditScore": 600,
        "Geography": "France",
        "Gender": "Male",
        "Age": 40,
        "Tenure": 3,
        "Balance": 60000,
        "NumOfProducts": 2,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 50000
    }

    pipeline = PredictPipeline()

    prediction, probability = (
        pipeline.predict(customer)
    )

    print("\n========== PREDICTION ==========")

    print(
        "Prediction:",
        prediction
    )

    print(
        "Churn Probability:",
        f"{probability:.2%}"
    )

    if prediction == 1:
        print("Customer will likely churn.")
    else:
        print("Customer will likely stay.")

    # -----------------------------
    # Final Output
    # -----------------------------

    print("\n========== PIPELINE COMPLETED ==========\n")

    print("Raw Data:", raw_data_path)
    print("Train Data:", train_path)
    print("Test Data:", test_path)
    print("Model:", model_path)
    print("Prediction: artifacts\\prediction.csv")
    print("Logs: logs\\pipeline.log")


if __name__ == "__main__":

    run_pipeline()
