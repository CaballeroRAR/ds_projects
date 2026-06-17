from kedro.pipeline import Pipeline, node, pipeline
from .nodes import fetch_sheet_data

def create_data_ingestion_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=fetch_sheet_data,
                inputs=None,
                outputs="raw_sheet_data",
                name="fetch_sheet_data_node",
            )
        ]
    )
