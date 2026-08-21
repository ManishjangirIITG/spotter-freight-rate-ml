from freight_rate.logging_config import setup_logging
from freight_rate.models.evaluate import evaluate_model_performance

if __name__ == "__main__":
    setup_logging("scripts_evaluate.log")
    evaluate_model_performance()