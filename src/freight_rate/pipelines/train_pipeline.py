import logging
from freight_rate.logging_config import setup_logging
from freight_rate.models.train import train_pipeline

logger = logging.getLogger(__name__)

def run_training():
    setup_logging("pipeline_train.log")
    logger.info("Starting automated end-to-end training pipeline...")
    model, metrics = train_pipeline()
    logger.info(f"Training completed successfully. Final MAE: ${metrics['mae']:.2f}")

if __name__ == "__main__":
    run_training()