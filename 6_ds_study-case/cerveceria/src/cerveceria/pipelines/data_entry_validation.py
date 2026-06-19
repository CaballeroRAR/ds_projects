from kedro.pipeline import Pipeline, node, pipeline
from .nodes import validate_data_entry, db_compare

def create_validation_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=validate_data_entry,
                inputs="raw_sheet_data",
                outputs="validated_sheet_data",
                name="validate_data_entry_node",
            ),
            node(
                func=db_compare,
                inputs=["validated_sheet_data", "quiero_chela_bq"],
                outputs="quiero_chela_bq",
                name="db_compare_node",
            ),
        ]
    )
