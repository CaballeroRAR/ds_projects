from kedro.pipeline import Pipeline, node, pipeline
from .nodes import validate_data_entry

def create_validation_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=validate_data_entry,
                inputs="raw_sheet_data",
                outputs="quiero_chela_bq",
                name="validate_data_entry_node",
            )
        ]
    )
