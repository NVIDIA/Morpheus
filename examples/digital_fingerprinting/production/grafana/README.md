<!--
# Copyright (c) 2023-2025, NVIDIA CORPORATION.
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
-->

# Using Grafana with Morpheus DFP Pipeline

This example builds on the [Azure DFP pipeline example](../production/README.md) to demonstrate how [Grafana](https://grafana.com/grafana/) can be used for log monitoring, error alerting, and inference results visualization.

## Grafana Configuration

The data sources and dashboards in this example are managed using configuration files. [Grafana's provisioning system](https://grafana.com/docs/grafana/latest/administration/provisioning/) then uses these files to add the data sources and dashboards to Grafana upon startup.

### Data Sources

Grafana includes built-in support for many data sources. There are also several data sources available that can be installed as plugins. More information about how to manage Grafana data sources can be found [here](https://grafana.com/docs/grafana/latest/datasources/).

The following data sources for this example are configured in [datasources.yaml](./datasources/datasources.yaml):

#### Loki data source

[Loki](https://grafana.com/docs/loki/latest/) is Grafana's log aggregation system. The Loki service is started automatically when the Grafana service starts up. The [Python script for running the DFP pipeline](./run.py) has been updated to configure a logging handler that sends the Morpheus logs to the Loki service.

#### CSV data source plugin

The [CSV data source plugin](https://grafana.com/grafana/plugins/marcusolsson-csv-datasource/) is installed to Grafana to read the Azure inference results CSV file. This example assumes we are using the CSV file generated from running the Python script for [Azure DFP pipeline example](../production/README.md).

Please note that the use of the CSV plugin is for demonstration purposes only. Grafana includes support for many data sources more suitable for production deployments. See [here](https://grafana.com/docs/grafana/latest/datasources/) for more information.

#### Updates to `grafana.ini`

The following is added to the default `grafana.ini` to enable local mode for CSV data source plugin. This allows the CSV data source plugin to access files on local file system.

```
[plugin.marcusolsson-csv-datasource]
allow_local_mode = true
```

## Add Loki logging handler to DFP pipeline

The [pipeline run script](./run.py) for the Azure DFP example has been updated with the following to add the Loki logging handler which will publish the Morpheus logs to our Loki service:

```
loki_handler = logging_loki.LokiHandler(
    url=f"{loki_url}/loki/api/v1/push",
    tags={"app": "morpheus"},
    version="1",
)

configure_logging(loki_handler, log_level=log_level)
```

More information about Loki Python logging can be found [here](https://pypi.org/project/python-logging-loki/).

## Start Grafana and Loki services:

To start Grafana and Loki, run the following command on host in `examples/digital_fingerprinting/production`:
```bash
docker compose up grafana
```

## Run Azure DFP Training

Start `bash` shell in `morpheus_pipeline` container, run the following command on host in `examples/digital_fingerprinting/production`:

```bash
docker compose run --rm morpheus_pipeline bash
```

Set `PYTHONPATH` environment variable to allow import of production DFP Morpheus stages:
```
export PYTHONPATH=/workspace/examples/digital_fingerprinting/production/morpheus
```

Run the following in the container to train the Azure models.
```bash
cd /workspace/examples/digital_fingerprinting/production/grafana
python run.py --log_level DEBUG --train_users generic --start_time "2022-08-01" --input_file="../../../data/dfp/azure-training-data/AZUREAD_2022*.json"
```

## View DFP Logs Dashboard in Grafana

While the training pipeline is running, you can view Morpheus logs live in a Grafana dashboard at http://localhost:3000/dashboards.

Click on `DFP Logs` in the `General` folder. You may need to expand the `General` folder to see the link.

<img src="./img/dfp_logs_dashboard.png">

This dashboard was provisioned using configuration files but can also be manually created with the following steps:
1. Click `Dashboards` in the left-side menu.
2. Click `New` and select `New Dashboard`.
3. On the empty dashboard, click `+ Add visualization`.
4. In the dialog box that opens, Select the `Loki` data source.
5. In the `Edit Panel` view, change from `Time Series` visualization to `Logs`.
6. Add label filter: `app = morpheus`.
7. Change Order to `Oldest first`.
8. Click `Apply` to see your changes applied to the dashboard. Then click the save icon in the dashboard header.

## Set up Error Alerting

We demonstrate here with a simple example how we can use Grafana Alerting to notify us of a pipeline error moments after it occurs. This is especially useful with long-running pipelines.

1. Click `Alert Rules` under `Alerting` in the left-side menu.
2. Click `New Alert Rule`
3. Enter alert rule name: `DFP Error Alert Rule`
4. In `Define query and alert condition` section, select `Loki` data source.
5. Switch to `Code` view by clicking the `Code` button on the right.
6. Enter the following Loki Query which counts the number of log lines in last minute that have an error label (`severity=error`):
```
count_over_time({severity="error"}[1m])
```
7. Under `Expressions`, keep default configurations for `Reduce` and `Threshold`. The alert condition threshold will be error counts > 0.
7. In `Set evaluation behavior` section, click `+ New folder`, enter `morpheus` then click `Create` button.
8. Click `+ New evaluation group`, enter `dfp` for `Evaluation group name` and `1m` for `Evaluation interval`, then click `Create` button.
9. Enter `0s` for `Pending period`. This configures alerts to be fired instantly when alert condition is met.
10. Test your alert rule, by running the following in your `morpheus_pipeline` container. This will cause an error because `--input-file` glob will no longer match any of our training data files.
```
python run.py --log_level DEBUG --train_users generic --start_time "2022-08-01" --input_file="../../../data/dfp/azure-training-data/AZUREAD_2022*.json"
```
11. Click the `Preview` button to test run the alert rule. You should now see how our alert query picks up the error log, processes it through our reduce/threshold expressions and satisfies our alert condition. This is indicated by the `Firing` label in the `Threshold` section.

<img src="./img/dfp_error_alert_setup.png">

12. Finally, click `Save rule and exit` at top right of the page.

By default, all alerts will be sent through the `grafana-default-email` contact point. You can add email addresses to this contact point by clicking on `Contact points` under `Alerting` in the left-side menu. You would also have to configure SMTP in the `[smtp]` section of your `grafana.ini`. More information about Grafana Alerting contact points can found [here](https://grafana.com/docs/grafana/latest/alerting/fundamentals/contact-points/).

## Run Azure DFP Inference:

Run the inference pipeline with `filter_threshold=0.0`. This will disable the filtering of the inference results.

```bash
python run.py --log_level DEBUG --train_users none  --start_time "2022-08-30" --input_file="../../../data/dfp/azure-inference-data/*.json" --filter_threshold=0.0
```

The inference results will be saved to `dfp_detection_azure.csv` in the directory where script was run.

## View DFP Detections Dashboard in Grafana

When the inference pipeline completes, you can view visualizations of the inference results at http://localhost:3000/dashboards.

Click on `DFP Detections` in the `General` folder. You may need to expand the `General` folder to see the link.

<img src="./img/dfp_detections_dashboard.png">

The dashboard has the following visualization panels:

- Time series of absolute `mean_abs_z` across all rows. We can observe the higher `mean_abs_z` scores for `attacktarget@domain.com` as expected.
- Time series of z-loss per feature across all rows. We can observe the higher z-loss scores for the `appincrement` and `logcount` features.
- Bar gauge of maximum `mean_abs_z` per user.
- Bar gauge of maximum z-loss per feature.
- Table view of all rows/columns in CSV file.
