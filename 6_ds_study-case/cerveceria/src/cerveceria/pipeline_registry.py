from .pipelines.data_ingestion import create_data_ingestion_pipeline
from .pipelines.data_entry_validation import create_validation_pipeline
from kedro.pipeline import Pipeline

def register_pipelines() -> dict[str, Pipeline]:
    data_ingestion_pipeline = create_data_ingestion_pipeline()
    validation_pipeline = create_validation_pipeline()

    return {
        "__default__": data_ingestion_pipeline + validation_pipeline,
        "data_ingestion": data_ingestion_pipeline,
        "data_entry_validation": validation_pipeline,
    }
