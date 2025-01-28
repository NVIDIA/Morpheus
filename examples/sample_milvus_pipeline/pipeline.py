import csv
import logging
import random

from morpheus.pipeline.linear_pipeline import LinearPipeline
from morpheus.config import Config
from morpheus.stages.postprocess.serialize_stage import SerializeStage
from morpheus.io.deserializers import read_file_to_df
from morpheus.utils.type_utils import exec_mode_to_df_type_str

from morpheus.stages.input.in_memory_data_generation_stage import InMemoryDataGenStage

from morpheus.stages.preprocess.drop_null_stage import DropNullStage
from morpheus_llm.stages.output.write_to_vector_db_stage import WriteToVectorDBStage
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage

# from morpheus.utils.logging import configure_logging
# from morpheus.utils.type_support import numpy_to_cudf

# Import Milvus services from Morpheus
from morpheus_llm.service.vdb.milvus_vector_db_service import MilvusVectorDBResourceService, MilvusVectorDBService

import numpy as np
import json

from morpheus.utils.logger import configure_logging

from morpheus.stages.input.file_source_stage import FileSourceStage

from morpheus.stages.output.write_to_file_stage import WriteToFileStage

from morpheus.stages.preprocess.deserialize_stage import DeserializeStage

from morpheus.config import ExecutionMode

logger = logging.getLogger(__name__)

def generate_random_vector(dim=3):
    """Generate a random vector of specified dimensions."""
    return [random.uniform(-1.0, 1.0) for _ in range(dim)]


def save_to_json_file(data, file_name="data.json"):
    """Save data to a JSON file."""
    with open(file_name, "w") as json_file:
        json.dump(data, json_file, indent=4)
    print(f"JSON data saved to {file_name}")


def generate_json_records(collection_name, output_file, num_records=100):
    """Generate a list of records to be saved in a JSON file."""
    data = []
    for pk in range(1, num_records + 1):
        record = {
            "id": pk,
            "vector": generate_random_vector(),
            "metadata": json.dumps({"description": f"Record {pk}", "category": random.choice(["A", "B", "C"])})
        }
        data.append(record)

    json_data = {
        "collectionName": collection_name,
        "data": data
    }

    return json_data


def generate_csv(file_name, num_rows):
    """
    Generates a CSV file with fields:
    - PK: Primary Key (integer)
    - metadata: Sample text metadata (string)
    - vector: A random vector of 3 dimensions (array of floats)

    :param file_name: Name of the output CSV file
    :param num_rows: Number of rows to generate
    """
    with open(file_name, mode='w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['id', 'vector', 'metadata']
#        fieldnames = ['vector', 'metadata']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write the header
#        writer.writeheader()

        for pk in range(1, num_rows + 1):
            # Generate random metadata
            metadata = f"Sample metadata for row {pk}"

            # Generate a random vector of 3 dimensions
            vector = [round(random.uniform(-1.0, 1.0), 4) for _ in range(3)]

            # Write the row
            writer.writerow({
                'id': pk,
                'vector': vector,
                'metadata': metadata
            })


def main(input_file_name: str):
    # Step 1: Configure logging

    # Step 2: Initialize Morpheus Config
    config = Config()
    config.execution_mode = ExecutionMode.GPU

    # Step 3: Setup Milvus services
    milvus_db_service = MilvusVectorDBService("https://in03-c87c25d216da0ac.serverless.gcp-us-west1.cloud.zilliz.com", user="db_c87c25d216da0ac", password="Cv3;^~HaY.>~>!)H", token="1c80242758bbfc207773c9a731421d9d96e269ac3ef41d87b40725f53795e1305489827dd310f0e55fb886ba0ea15898244de182")
    milvus_resource_service = milvus_db_service.load_resource("test_collection")
    collection_name = "test_collection"
    vector_dim = 3  # Example: 3-dimensional vector embeddings

    source_df = read_file_to_df(input_file_name, df_type=exec_mode_to_df_type_str(config.execution_mode))
    print(source_df.shape[0])    
# Step 5: Create a pipeline
    pipeline = LinearPipeline(config)

    # Step 6: Define source stage
    def data_generator():
        for i in range(5):
            embedding = np.random.random(vector_dim).tolist()
            metadata = {"id": i, "label": f"example_{i}"}
            yield {"embedding": embedding, "metadata": metadata}

#    pipeline.set_source(FileSourceStage(config, filename=input_file_name))
    pipeline.set_source(InMemorySourceStage(config, dataframes=[source_df]))

#    pipeline.add_stage(
#        DropNullStage(config, "vector")
#    )

    # pipeline.add_stage(WriteToFileStage(config, filename="output_file.csv", overwrite=True))
#    pipeline.add_stage(SerializeStage(config))
    pipeline.add_stage(DeserializeStage(config))

    # Step 9: Add WriteToVectorDBStage for Milvus
    pipeline.add_stage(
        WriteToVectorDBStage(
            config,
            milvus_db_service,
            "test_collection"
        )
    )

    pipeline.build()

    # Step 10: Execute the pipeline
    pipeline.run()

if __name__ == "__main__":
#    file_name = "test.json"
    file_name = "test.csv"
    collection_name = "test_collection"
    generate_csv(file_name, 1000)
#    data = generate_json_records(collection_name, file_name)
#    save_to_json_file(data, file_name)
    main(file_name)
