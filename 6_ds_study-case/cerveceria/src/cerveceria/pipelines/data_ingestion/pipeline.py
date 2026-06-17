from kedro.pipeline import Pipeline, node, pipeline
from .nodes import fetch_sheet_data

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=fetch_sheet_data,
                inputs=None,
                outputs="quiero_chela_bq",
                name="fetch_sheet_data_node",
            )
        ]
    )
