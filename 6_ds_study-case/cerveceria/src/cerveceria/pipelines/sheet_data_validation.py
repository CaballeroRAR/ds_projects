from kedro.pipeline import Pipeline, node, pipeline
from .nodes import validate_sheet_data

def create_validation_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=validate_sheet_data,
                inputs="raw_sheet_data",
                outputs="quiero_chela_bq",
                name="validate_sheet_data_node",
            )
        ]
    )
