import os
import joblib
import pandas as pd

from src.logger import logger


class PredictPipeline:

    def __init__(self):

        self.model_path = "artifacts/model.pkl"
        self.preprocessor_path = "artifacts/preprocessor.pkl"
        self.prediction_path = "artifacts/prediction.csv"

    def predict(self, input_data):

        logger.info("Prediction started")

        model = joblib.load(self.model_path)
        preprocessor = joblib.load(self.preprocessor_path)

        input_df = pd.DataFrame([input_data])

        transformed_data = preprocessor.transform(input_df)

        prediction = model.predict(transformed_data)

        probability = model.predict_proba(
            transformed_data
        )[0][1]

        # Save prediction result
        os.makedirs("artifacts", exist_ok=True)

        prediction_result = pd.DataFrame([{
            "Prediction": prediction[0],
            "Churn_Probability": probability
        }])

        prediction_result.to_csv(
            self.prediction_path,
            index=False
        )

        logger.info(
            f"Prediction: {prediction[0]}, "
            f"Probability: {probability:.4f}"
        )

        logger.info(
            f"Prediction result saved at: "
            f"{self.prediction_path}"
        )

        return prediction[0], probability


if __name__ == "__main__":

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

    prediction, probability = pipeline.predict(customer)

    print("Prediction:", prediction)
    print("Churn Probability:", probability)

    print(
        "Prediction saved:",
        pipeline.prediction_path
    )

    if prediction == 1:
        print("Customer will likely churn.")
    else:
        print("Customer will likely stay.")
