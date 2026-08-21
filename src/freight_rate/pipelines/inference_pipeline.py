import logging
from freight_rate.logging_config import setup_logging
from freight_rate.models.predict import generate_validation_predictions

logger = logging.getLogger(__name__)

def run_inference():
    setup_logging("pipeline_inference.log")
    logger.info("Starting batch inference pipeline...")
    df_preds = generate_validation_predictions()
    logger.info(f"Inference completed. Generated {len(df_preds)} predictions.")

if __name__ == "__main__":
    run_inference()