import logging
from pathlib import Path
import pandas as pd
from freight_rate import config
from freight_rate.data.validation import validate_schema, DataValidationError

logger = logging.getLogger(__name__)

def load_raw_dataset(path: Path, is_training: bool = True) -> pd.DataFrame:
    """
    Reads a CSV dataset from disk and executes schema validation.
    """
    logger.info(f"Ingesting raw dataset from {path}")
    if not path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {path}")
        
    df = pd.read_csv(path)
    is_valid, issues = validate_schema(df, is_training=is_training)
    
    if issues:
        logger.warning(f"Data quality report for {path.name}: {issues}")
        
    if not is_valid:
        raise DataValidationError(f"Schema validation failed for {path.name}: {issues}")

    logger.info(f"Ingestion of raw dataset from {path} completed")
        
    return df

def load_train_data() -> pd.DataFrame:
    """Convenience loader for raw training data."""
    return load_raw_dataset(config.TRAIN_PATH, is_training=True)

def load_validation_data() -> pd.DataFrame:
    """Convenience loader for external validation target data."""
    return load_raw_dataset(config.VALIDATION_PATH, is_training=False)

def load_december_data() -> pd.DataFrame:
    """Convenience loader for December chart input scenario."""
    return load_raw_dataset(config.DECEMBER_PATH, is_training=False)