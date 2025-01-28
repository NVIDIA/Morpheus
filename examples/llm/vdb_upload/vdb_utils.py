# Copyright (c) 2024-2025, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import typing

import yaml

from morpheus.config import Config
from morpheus.config import PipelineModes

logger = logging.getLogger(__name__)

DEFAULT_RSS_URLS = [
    "https://www.theregister.com/security/headlines.atom",
    "https://isc.sans.edu/dailypodcast.xml",
    "https://threatpost.com/feed/",
    "http://feeds.feedburner.com/TheHackersNews?format=xml",
    "https://www.bleepingcomputer.com/feed/",
    "https://therecord.media/feed/",
    "https://blog.badsectorlabs.com/feeds/all.atom.xml",
    "https://krebsonsecurity.com/feed/",
    "https://www.darkreading.com/rss_simple.asp",
    "https://blog.malwarebytes.com/feed/",
    "https://msrc.microsoft.com/blog/feed",
    "https://securelist.com/feed",
    "https://www.crowdstrike.com/blog/feed/",
    "https://threatconnect.com/blog/rss/",
    "https://news.sophos.com/en-us/feed/",
    "https://www.us-cert.gov/ncas/current-activity.xml",
    "https://www.csoonline.com/feed",
    "https://www.cyberscoop.com/feed",
    "https://research.checkpoint.com/feed",
    "https://feeds.fortinet.com/fortinet/blog/threat-research",
    "https://www.mcafee.com/blogs/rss",
    "https://www.digitalshadows.com/blog-and-research/rss.xml",
    "https://www.nist.gov/news-events/cybersecurity/rss.xml",
    "https://www.sentinelone.com/blog/rss/",
    "https://www.bitdefender.com/blog/api/rss/labs/",
    "https://www.welivesecurity.com/feed/",
    "https://unit42.paloaltonetworks.com/feed/",
    "https://mandiant.com/resources/blog/rss.xml",
    "https://www.wired.com/feed/category/security/latest/rss",
    "https://www.wired.com/feed/tag/ai/latest/rss",
    "https://blog.google/threat-analysis-group/rss/",
    "https://intezer.com/feed/",
]

DEFAULT_RSS_CONFIG = {
    # RSS feeds can take a while to pull, smaller batch sizes allows the pipeline to feel more responsive
    "batch_size": 32,
    "output_batch_size": 2048,
    "cache_dir": "./.cache/http",
    "stop_after_rec": 0,
    "feed_input": DEFAULT_RSS_URLS.copy(),
    "strip_markup": True,
    "web_scraper_config": {}
}

DEFAULT_EMBEDDINGS_MODEL_KWARGS = {"force_convert_inputs": True, "use_shared_memory": False}

DEFAULT_PIPELINE_CONFIG = {"edge_buffer_size": 128, "max_batch_size": 256}

DEFAULT_TOKENIZER_CONFIG = {
    "model_name": "bert-base-uncased-hash",
    "model_kwargs": {
        "add_special_tokens": False,
        "column": "content",
        "do_lower_case": True,
        "truncation": True,
        "vocab_hash_file": "data/bert-base-uncased-hash.txt",
    }
}

DEFAULT_VDB_CONFIG = {
    # Vector db upload has some significant transaction overhead, batch size here should be as large as possible
    'batch_size': 16384,
    'recreate': True,
    'truncate_long_strings': True
}

DEFAULT_MILVUS_CONFIG = {
    "index_conf": {
        "field_name": "embedding",
        "metric_type": "L2",
        "index_type": "HNSW",
        "params": {
            "M": 8,
            "efConstruction": 64,
        },
    },
    "schema_conf": {
        "enable_dynamic_field": True,
        "schema_fields": [
            {
                "name": "id",
                "dtype": "INT64",
                "description": "Primary key for the collection",
                "is_primary": True,
                "auto_id": True
            },
            {
                "name": "title", "dtype": "VARCHAR", "description": "The title of the RSS Page", "max_length": 65_535
            },
            {
                "name": "source", "dtype": "VARCHAR", "description": "The URL of the RSS Page", "max_length": 65_535
            },
            {
                "name": "summary",
                "dtype": "VARCHAR",
                "description": "The summary of the RSS Page",
                "max_length": 65_535
            },
            {
                "name": "content",
                "dtype": "VARCHAR",
                "description": "A chunk of text from the RSS Page",
                "max_length": 65_535
            },
            {
                "name": "embedding", "dtype": "FLOAT_VECTOR", "description": "Embedding vectors", "dim": 384
            },
        ],
        "description": "RSS collection schema"
    }
}

YAML_TO_CONFIG_MAPPING = {
    'embeddings': 'embeddings_config',
    'pipeline': 'pipeline_config',
    'tokenizer': 'tokenizer_config',
    'vdb': 'vdb_config'
}


def build_milvus_config(resource_schema_config: dict):
    import pymilvus

    from morpheus_llm.service.vdb.milvus_client import DATA_TYPE_MAP

    schema_fields = []
    for field_data in resource_schema_config["schema_conf"]["schema_fields"]:
        field_data["dtype"] = DATA_TYPE_MAP.get(field_data["dtype"])
        field_schema = pymilvus.FieldSchema(**field_data)
        schema_fields.append(field_schema.to_dict())

    resource_schema_config["schema_conf"]["schema_fields"] = schema_fields

    return resource_schema_config


def is_valid_service(ctx, param, value):  # pylint: disable=unused-argument
    """
    Validate the provided vector database service name.

    Checks if the given vector database service name is supported and valid. This is used as a callback function
    for a CLI option to ensure that the user inputs a supported service name.

    Parameters
    ----------
    ctx : click.Context
        The context within which the command is being invoked.
    param : click.Parameter
        The parameter object that this function serves as a callback for.
    value : str
        The value of the parameter to validate.

    Returns
    -------
    str
        The validated and lowercased service name.

    Raises
    ------
    click.BadParameter
        If the provided service name is not supported or invalid.
    """
    from morpheus_llm.service.vdb.utils import validate_service
    value = value.lower()
    return validate_service(service_name=value)


def merge_dicts(dict_1, dict_2):
    """
    Recursively merge two dictionaries.

    Nested dictionaries are merged instead of being replaced.
    Non-dict items in the second dictionary will override those in the first.

    Parameters
    ----------
    dict_1 : dict
        The first dictionary.
    dict_2 : dict
        The second dictionary, whose items will take precedence.

    Returns
    -------
    dict
        The merged dictionary.
    """
    result = dict_1.copy()
    for key, value in dict_2.items():
        dict_1_value = dict_1.get(key)
        if isinstance(dict_1_value, dict) and isinstance(value, dict):
            result[key] = merge_dicts(dict_1_value, value)
        else:
            result[key] = value
    return result


def build_pipeline_config(pipeline_config: dict):
    """
    Construct a pipeline configuration object from a dictionary.

    Parameters
    ----------
    pipeline_config : dict
        A dictionary containing pipeline configuration parameters.

    Returns
    -------
    Config
        A pipeline configuration object populated with values from the input dictionary.

    Notes
    -----
    This function is responsible for mapping a dictionary of configuration parameters
    into a structured configuration object used by the pipeline.
    """

    config = Config()
    config.mode = PipelineModes.NLP

    embedding_size = pipeline_config.get('embedding_size')

    config.num_threads = pipeline_config.get('num_threads')
    config.pipeline_batch_size = pipeline_config.get('pipeline_batch_size')
    config.model_max_batch_size = pipeline_config.get('max_batch_size')
    config.feature_length = pipeline_config.get('feature_length')
    config.edge_buffer_size = pipeline_config.get('edge_buffer_size')
    config.class_labels = [str(i) for i in range(embedding_size)]

    return config


def _set_values_if_exists(dest_dict: dict[str, typing.Any], src_dict: dict[str, typing.Any], mapping: dict[str, str]):
    """
    Set values in dest_dict if they exist in src_dict
    Since a single source key can map to multiple destination keys, the mapping dictionary is in the form of:
        {dest_key: src_key}
    """
    for dest_key, src_key in mapping.items():
        # Explicitly using an `in` test here since `None` is a valid value
        if src_key in src_dict:
            dest_dict[dest_key] = src_dict[src_key]


def _cli_args_to_config(cli_args: dict[str, typing.Any], include_defaults: bool = False) -> dict:
    """
    CLI arguments are in a flat structure, this function converts them to the nested structure used byt the yaml congig
    allowing for easy merging of the two.
    """
    config = {}
    source_config = {}
    source_type = cli_args.get('source_type', [])
    if 'rss' in source_type:
        if include_defaults:
            rss_config = DEFAULT_RSS_CONFIG.copy()
        else:
            rss_config = {"web_scraper_config": {}}

        _set_values_if_exists(rss_config['web_scraper_config'],
                              cli_args, {
                                  "chunk_size": "content_chunking_size", "enable_cache": "enable_cache"
                              })
        _set_values_if_exists(
            rss_config,
            cli_args,
            {
                "cooldown_interval_sec": "interval_secs",
                "stop_after_rec": "stop_after",
                "enable_cache": "enable_cache",
                "enable_monitor": "enable_monitors",
                "interval_sec": "interval_secs",
                "request_timeout_sec": "rss_request_timeout_sec",
                "run_indefinitely": "run_indefinitely",
                "vdb_resource_name": "vector_db_resource_name",
            })

        # Handle feed inputs separately
        if len(cli_args.get('feed_inputs', [])) > 0:
            rss_config['feed_input'] = cli_args['feed_inputs']

        source_config['rss'] = {'type': 'rss', 'name': 'rss', 'config': rss_config}

    if 'filesystem' in source_type:
        fs_config = {"extractor_config": {}}
        _set_values_if_exists(fs_config["extractor_config"],
                              cli_args, {
                                  "chunk_size": "content_chunking_size", "num_threads": "num_threads"
                              })

        _set_values_if_exists(
            fs_config,
            cli_args,
            {
                "batch_size": "pipeline_batch_size",
                "enable_monitor": "enable_monitors",
                "filenames": "file_source",
                "vdb_resource_name": "vector_db_resource_name",
                "watch": "run_indefinitely"
            })

        source_config['filesystem'] = {'type': 'filesystem', 'name': 'filesystem-cli', 'config': fs_config}

    config['source_config'] = source_config

    embeddings_model_kwargs = {}
    if include_defaults:
        embeddings_model_kwargs.update(DEFAULT_EMBEDDINGS_MODEL_KWARGS.copy())

    _set_values_if_exists(embeddings_model_kwargs,
                          cli_args, {
                              "model_name": "embedding_model_name", "server_url": "triton_server_url"
                          })

    embeddings_config = {"model_kwargs": embeddings_model_kwargs}

    _set_values_if_exists(embeddings_config,
                          cli_args,
                          {
                              "feature_length": "model_fea_length",
                              "max_batch_size": "model_max_batch_size",
                              "num_threads": "num_threads"
                          })

    config['embeddings_config'] = embeddings_config

    # These values will be replaced later with a morpheus.config.Config object
    pipeline_config = {}
    if include_defaults:
        pipeline_config.update(DEFAULT_PIPELINE_CONFIG.copy())

    _set_values_if_exists(
        pipeline_config,
        cli_args,
        {
            "embedding_size": "embedding_size",
            "feature_length": "model_fea_length",
            "isolate_embeddings": "isolate_embeddings",
            "num_threads": "num_threads",
            "pipeline_batch_size": "pipeline_batch_size",
        })

    config['pipeline_config'] = pipeline_config

    if include_defaults:
        config['tokenizer_config'] = DEFAULT_TOKENIZER_CONFIG.copy()

    vdb_config = {}
    if include_defaults:
        vdb_config.update(DEFAULT_VDB_CONFIG.copy())

    _set_values_if_exists(
        vdb_config,
        cli_args,
        {
            'embedding_size': 'embedding_size',
            'resource_name': 'vector_db_resource_name',
            'service': 'vector_db_service',
            'uri': 'vector_db_uri'
        })

    # Milvus configs to be built later if needed. The reason here is that we could use the default embedding size, but
    # override the service type or name in other levels, we need the final resolved values of all three, for now just
    # stub in hte resource name
    resource_name = vdb_config.get('resource_name')
    if resource_name is not None:
        vdb_config["resource_schemas"] = {resource_name: None}

    config['vdb_config'] = vdb_config

    return config


def build_config(vdb_conf_path: str | None,
                 explicit_cli_args: dict[str, typing.Any],
                 implicit_cli_args: dict[str, typing.Any]):
    """
    Load and merge configurations from the CLI and YAML file.

    This function combines the configurations provided via the CLI with those specified in a YAML file.
    If a YAML configuration file is specified and exists, it will merge its settings with the CLI settings.

    The order of precedence is as follows: Explict CLI args set by user > YAML settings > default values of CLI args.

    Parameters
    ----------
    vdb_conf_path : str
        Path to the YAML configuration file.
    explicit_cli_args : dict[str, typing.Any]
        CLI args explicitly set by the user.
    implicit_cli_args : dict[str, typing.Any]
        CLI args including default values not explicitly set by the user.

    Returns
    -------
    dict
        A dictionary containing the final merged configuration for the pipeline.
    """

    # Load and merge configurations from the YAML file if it exists
    if vdb_conf_path:
        with open(vdb_conf_path, 'r', encoding='utf-8') as file:
            yaml_config = yaml.safe_load(file).get('vdb_pipeline', {})

        # Yaml specific transforms
        for (yaml_key, config_key) in YAML_TO_CONFIG_MAPPING.items():
            yaml_config[config_key] = yaml_config.pop(yaml_key, {})

        sources = yaml_config.pop('sources', [])
        yaml_config['source_config'] = {src['name']: src for src in sources}

    else:
        yaml_config = {}

    implicit_cli_config = _cli_args_to_config(implicit_cli_args, include_defaults=True)
    explicit_cli_config = _cli_args_to_config(explicit_cli_args, include_defaults=False)

    final_config = merge_dicts(implicit_cli_config, yaml_config)
    final_config = merge_dicts(final_config, explicit_cli_config)

    # Flatten the source configs into a list
    final_config['source_config'] = list(final_config.pop('source_config').values())

    # Handle the resource schema separately, the reason is we need both the service type, resource name and the
    # embedding size to all be defined, some or all of these values could be defined at any level.
    vdb_config = final_config['vdb_config']
    if vdb_config.get('service') == 'milvus':
        # Replace the resource schema configs with Milvus config objects
        resource_schema_configs = vdb_config.pop("resource_schemas", {})
        resource_schemas = {}
        for (resource_name, resource_schema) in resource_schema_configs.items():
            if resource_schema is None:
                resource_schema = DEFAULT_MILVUS_CONFIG.copy()
                # Update the embedding_size
                resource_schema['schema_conf']['schema_fields'][-1]['dim'] = vdb_config['embedding_size']

            resource_schemas[resource_name] = build_milvus_config(resource_schema)

        vdb_config['resource_schemas'] = resource_schemas

    # convert the pipeline config to a morpheus.config.Config object
    final_config['pipeline_config'] = build_pipeline_config(final_config['pipeline_config'])

    return final_config
