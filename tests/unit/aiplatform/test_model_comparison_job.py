# -*- coding: utf-8 -*-

# Copyright 2022 Google LLC
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
#

import pytest
import yaml
import json
from google.protobuf import json_format, struct_pb2

from unittest import mock
from unittest.mock import patch, MagicMock
from datetime import datetime
from google.auth import credentials as auth_credentials

import test_metadata
from google.cloud import storage

from google.cloud import aiplatform
from google.cloud.aiplatform import base
from google.cloud.aiplatform import models
from google.cloud.aiplatform.utils import gcs_utils

from google.cloud.aiplatform.compat.services import (
    model_service_client,
    metadata_service_client_v1 as metadata_service_client,
    job_service_client_v1 as job_service_client,
)
from google.cloud.aiplatform.model_comparison import model_comparison_job

from google.cloud.aiplatform_v1.services.pipeline_service import (
    client as pipeline_service_client_v1,
)

from google.cloud.aiplatform.compat.types import model as gca_model

from google.cloud.aiplatform.compat.types import (
    pipeline_job as gca_pipeline_job,
    pipeline_state as gca_pipeline_state,
    model_evaluation as gca_model_evaluation,
    context as gca_context,
    artifact as gca_artifact,
    batch_prediction_job as gca_batch_prediction_job,
)

_TEST_PROJECT = "test-project"
_TEST_LOCATION = "us-central1"
_TEST_MODEL_NAME = "test-model"
_TEST_MODEL_ID = "1028944691210842416"
_TEST_EVAL_ID = "1028944691210842622"
_TEST_EXPERIMENT = "test-experiment"
_TEST_BATCH_PREDICTION_JOB_ID = "614161631630327111"

_TEST_MODEL_RESOURCE_NAME = model_service_client.ModelServiceClient.model_path(
    _TEST_PROJECT, _TEST_LOCATION, _TEST_MODEL_ID
)

_TEST_MODEL_EVAL_RESOURCE_NAME = (
    model_service_client.ModelServiceClient.model_evaluation_path(
        _TEST_PROJECT,
        _TEST_LOCATION,
        _TEST_MODEL_ID,
        _TEST_EVAL_ID,
    )
)

_TEST_BATCH_PREDICTION_RESOURCE_NAME = (
    job_service_client.JobServiceClient.batch_prediction_job_path(
        _TEST_PROJECT, _TEST_LOCATION, _TEST_BATCH_PREDICTION_JOB_ID
    )
)

_TEST_MODEL_EVAL_METRICS = {
    "auPrc": 0.80592036,
    "auRoc": 0.8100363,
    "logLoss": 0.53061414,
    "confidenceMetrics": [
        {
            "confidenceThreshold": -0.01,
            "recall": 1.0,
            "precision": 0.5,
            "falsePositiveRate": 1.0,
            "f1Score": 0.6666667,
            "recallAt1": 1.0,
            "precisionAt1": 0.5,
            "falsePositiveRateAt1": 1.0,
            "f1ScoreAt1": 0.6666667,
            "truePositiveCount": "415",
            "falsePositiveCount": "415",
        },
        {
            "recall": 1.0,
            "precision": 0.5,
            "falsePositiveRate": 1.0,
            "f1Score": 0.6666667,
            "recallAt1": 0.74216866,
            "precisionAt1": 0.74216866,
            "falsePositiveRateAt1": 0.25783134,
            "f1ScoreAt1": 0.74216866,
            "truePositiveCount": "415",
            "falsePositiveCount": "415",
        },
    ],
}

_TEST_INVALID_MODEL_RESOURCE_NAME = (
    f"prj/{_TEST_PROJECT}/locations/{_TEST_LOCATION}/models/{_TEST_MODEL_ID}"
)

# pipeline job
_TEST_ID = "1028944691210842416"
_TEST_PIPELINE_JOB_DISPLAY_NAME = "sample-pipeline-job-display-name"
_TEST_PIPELINE_JOB_ID = "sample-test-pipeline-202111111"
_TEST_GCS_BUCKET_NAME = "my-bucket"
_TEST_CREDENTIALS = auth_credentials.AnonymousCredentials()
_TEST_SERVICE_ACCOUNT = "abcde@my-project.iam.gserviceaccount.com"
_TEST_PIPELINE_ROOT = f"gs://{_TEST_GCS_BUCKET_NAME}/pipeline_root"
_TEST_PIPELINE_CREATE_TIME = datetime.now()

_TEST_TEMPLATE_PATH = f"gs://{_TEST_GCS_BUCKET_NAME}/evaluation_default_pipeline.json"

_TEST_TEMPLATE_REF = {
    "automl_tabular_without_feature_attribution": _TEST_TEMPLATE_PATH,
    "automl_tabular_with_feature_attribution": f"gs://{_TEST_GCS_BUCKET_NAME}/evaluation_feature_attribution_pipeline.json",
}
_TEST_PARENT = f"projects/{_TEST_PROJECT}/locations/{_TEST_LOCATION}"
_TEST_NETWORK = f"projects/{_TEST_PROJECT}/global/networks/{_TEST_PIPELINE_JOB_ID}"

_TEST_PIPELINE_JOB_NAME = f"projects/{_TEST_PROJECT}/locations/{_TEST_LOCATION}/pipelineJobs/{_TEST_PIPELINE_JOB_ID}"
_TEST_INVALID_PIPELINE_JOB_NAME = (
    f"prj/{_TEST_PROJECT}/locations/{_TEST_LOCATION}/{_TEST_PIPELINE_JOB_ID}"
)
_TEST_MODEL_EVAL_PIPELINE_JOB_DISPLAY_NAME = "test-eval-job"
_TEST_EVAL_RESOURCE_DISPLAY_NAME = "my-eval-resource-display-name"

_TEST_MODEL_EVAL_METADATA = {"pipeline_job_resource_name": _TEST_PIPELINE_JOB_NAME}

_TEST_MODEL_EVAL_PIPELINE_PARAMETER_VALUES = {
    "batch_predict_gcs_source_uris": ["gs://my-bucket/my-prediction-data.csv"],
    "dataflow_service_account": _TEST_SERVICE_ACCOUNT,
    "batch_predict_instances_format": "csv",
    "model_name": _TEST_MODEL_RESOURCE_NAME,
    "evaluation_display_name": _TEST_EVAL_RESOURCE_DISPLAY_NAME,
    "prediction_type": "classification",
    "project": _TEST_PROJECT,
    "location": _TEST_LOCATION,
    "root_dir": _TEST_GCS_BUCKET_NAME,
    "target_column_name": "predict_class",
}

_TEST_JSON_FORMATTED_MODEL_EVAL_PIPELINE_PARAMETER_VALUES = {
    "batch_predict_gcs_source_uris": '["gs://sdk-model-eval/batch-pred-heart.csv"]',
    "dataflow_service_account": _TEST_SERVICE_ACCOUNT,
    "batch_predict_instances_format": "csv",
    "model_name": _TEST_MODEL_RESOURCE_NAME,
    "prediction_type": "classification",
    "project": _TEST_PROJECT,
    "location": _TEST_LOCATION,
    "root_dir": _TEST_GCS_BUCKET_NAME,
    "target_column_name": "predict_class",
}

_TEST_MODEL_EVAL_PIPELINE_SPEC = {
    "pipelineInfo": {"name": "evaluation-default-pipeline"},
    "root": {
        "dag": {"tasks": {}},
        "inputDefinitions": {
            "parameters": {
                "batch_predict_gcs_source_uris": {"type": "STRING"},
                "dataflow_service_account": _TEST_SERVICE_ACCOUNT,
                "batch_predict_instances_format": {"type": "STRING"},
                "batch_predict_machine_type": {"type": "STRING"},
                "location": {"type": "STRING"},
                "model_name": {"type": "STRING"},
                "prediction_type": {"type": "STRING"},
                "project": {"type": "STRING"},
                "root_dir": {"type": "STRING"},
                "target_column_name": {"type": "STRING"},
            }
        },
    },
    "schemaVersion": "2.0.0",
    "sdkVersion": "kfp-1.8.12",
    "components": {},
}

_TEST_MODEL_EVAL_PIPELINE_SPEC_JSON = json.dumps(
    {
        "pipelineInfo": {"name": "evaluation-default-pipeline"},
        "root": {
            "dag": {"tasks": {}},
            "inputDefinitions": {
                "parameters": {
                    "batch_predict_gcs_source_uris": {"type": "STRING"},
                    "dataflow_service_account": {"type": "STRING"},
                    "batch_predict_instances_format": {"type": "STRING"},
                    "batch_predict_machine_type": {"type": "STRING"},
                    "location": {"type": "STRING"},
                    "model_name": {"type": "STRING"},
                    "prediction_type": {"type": "STRING"},
                    "project": {"type": "STRING"},
                    "root_dir": {"type": "STRING"},
                    "target_column_name": {"type": "STRING"},
                }
            },
        },
        "schemaVersion": "2.0.0",
        "sdkVersion": "kfp-1.8.12",
        "components": {},
    }
)

_TEST_INVALID_MODEL_EVAL_PIPELINE_SPEC = json.dumps(
    {
        "pipelineInfo": {"name": "my-pipeline"},
        "root": {
            "dag": {"tasks": {}},
            "inputDefinitions": {
                "parameters": {
                    "batch_predict_gcs_source_uris": {"type": "STRING"},
                    "dataflow_service_account": {"type": "STRING"},
                    "batch_predict_instances_format": {"type": "STRING"},
                    "model_name": {"type": "STRING"},
                    "prediction_type": {"type": "STRING"},
                    "project": {"type": "STRING"},
                    "location": {"type": "STRING"},
                    "root_dir": {"type": "STRING"},
                    "target_column_name": {"type": "STRING"},
                }
            },
        },
        "schemaVersion": "2.0.0",
        "sdkVersion": "kfp-1.8.12",
        "components": {"test_component": {}},
    }
)

_TEST_MODEL_EVAL_PIPELINE_JOB = json.dumps(
    {
        "runtimeConfig": {"parameters": _TEST_MODEL_EVAL_PIPELINE_PARAMETER_VALUES},
        "pipelineInfo": {"name": "evaluation-default-pipeline"},
        "root": {
            "dag": {"tasks": {}},
            "inputDefinitions": {
                "parameters": {
                    "batch_predict_gcs_source_uris": {"type": "STRING"},
                    "dataflow_service_account": {"type": "STRING"},
                    "batch_predict_instances_format": {"type": "STRING"},
                    "batch_predict_machine_type": {"type": "STRING"},
                    "location": {"type": "STRING"},
                    "model_name": {"type": "STRING"},
                    "prediction_type": {"type": "STRING"},
                    "project": {"type": "STRING"},
                    "root_dir": {"type": "STRING"},
                    "target_column_name": {"type": "STRING"},
                }
            },
        },
        "schemaVersion": "2.0.0",
        "sdkVersion": "kfp-1.8.12",
        "components": {},
    }
)

_TEST_INVALID_MODEL_EVAL_PIPELINE_JOB = json.dumps(
    {
        "runtimeConfig": {
            "parameterValues": _TEST_MODEL_EVAL_PIPELINE_PARAMETER_VALUES
        },
        "pipelineInfo": {"name": "my-pipeline"},
        "root": {
            "dag": {"tasks": {}},
            "inputDefinitions": {
                "parameters": {
                    "batch_predict_gcs_source_uris": {"type": "STRING"},
                    "batch_predict_instances_format": {"type": "STRING"},
                    "model_name": {"type": "STRING"},
                    "prediction_type": {"type": "STRING"},
                    "project": {"type": "STRING"},
                    "location": {"type": "STRING"},
                    "root_dir": {"type": "STRING"},
                    "target_column_name": {"type": "STRING"},
                }
            },
        },
        "schemaVersion": "2.0.0",
        "sdkVersion": "kfp-1.8.12",
        "components": {"test_component": {}},
    }
)

_EVAL_GCP_RESOURCES_STR = (
    '{\n  "resources": [\n    {\n      "resourceType": "ModelEvaluation",\n      "resourceUri": "https://us-central1-aiplatform.googleapis.com/v1/'
    + _TEST_MODEL_EVAL_RESOURCE_NAME
    + '"\n    }\n  ]\n}'
)

_BP_JOB_GCP_RESOURCES_STR = (
    '{\n  "resources": [\n    {\n      "resourceType": "BatchPredictionJob",\n      "resourceUri": "https://us-central1-aiplatform.googleapis.com/v1/'
    + _TEST_BATCH_PREDICTION_RESOURCE_NAME
    + '"\n    }\n  ]\n}'
)

_TEST_PIPELINE_JOB_DETAIL_EVAL = {
    "output:gcp_resources": _EVAL_GCP_RESOURCES_STR,
}

_TEST_PIPELINE_JOB_DETAIL_BP = {
    "output:gcp_resources": _BP_JOB_GCP_RESOURCES_STR,
}

_TEST_EVAL_METRICS_ARTIFACT_NAME = (
    "projects/123/locations/us-central1/metadataStores/default/artifacts/456"
)
_TEST_EVAL_METRICS_ARTIFACT_URI = "gs://test-bucket/eval_pipeline_root/123/evaluation-default-pipeline-20220615135923/model-evaluation-2_-789/evaluation_metrics"

_TEST_EXPERIMENT = "test-experiment"
_TEST_METADATASTORE = f"{_TEST_PARENT}/metadataStores/default"
_TEST_CONTEXT_NAME = f"{_TEST_METADATASTORE}/contexts/{_TEST_EXPERIMENT}"

# experiment mocks
get_experiment_mock = test_metadata.get_experiment_mock
get_metadata_store_mock = test_metadata.get_metadata_store_mock
get_context_mock = test_metadata.get_context_mock
add_context_children_mock = test_metadata.add_context_children_mock

@pytest.fixture
def mock_pipeline_service_create():
    with mock.patch.object(
        pipeline_service_client_v1.PipelineServiceClient, "create_pipeline_job"
    ) as mock_create_pipeline_job:
        mock_create_pipeline_job.return_value = gca_pipeline_job.PipelineJob(
            name=_TEST_PIPELINE_JOB_NAME,
            state=gca_pipeline_state.PipelineState.PIPELINE_STATE_SUCCEEDED,
            create_time=_TEST_PIPELINE_CREATE_TIME,
            service_account=_TEST_SERVICE_ACCOUNT,
            network=_TEST_NETWORK,
        )
        yield mock_create_pipeline_job

@pytest.fixture
def mock_model_comparison_job_get():
    with mock.patch.object(
        pipeline_service_client_v1.PipelineServiceClient, "get_pipeline_job"
    ) as mock_get_model_comparison_job:
        mock_get_model_comparison_job.return_value = gca_pipeline_job.PipelineJob(
            name=_TEST_PIPELINE_JOB_NAME,
            state=gca_pipeline_state.PipelineState.PIPELINE_STATE_SUCCEEDED,
            create_time=_TEST_PIPELINE_CREATE_TIME,
            service_account=_TEST_SERVICE_ACCOUNT,
            network=_TEST_NETWORK,
            pipeline_spec=_TEST_MODEL_EVAL_PIPELINE_SPEC,
        )
        yield mock_get_model_comparison_job

@pytest.fixture
def mock_load_yaml_and_json(job_spec):
    with patch.object(storage.Blob, "download_as_bytes") as mock_load_yaml_and_json:
        mock_load_yaml_and_json.return_value = job_spec.encode()
        yield mock_load_yaml_and_json



@pytest.mark.usefixtures("google_auth_mock")
class TestModelComparisonJob:
    @pytest.mark.parametrize(
        "job_spec",
        [_TEST_MODEL_EVAL_PIPELINE_JOB],
    )
    def test_init_model_comparison_job(
        self,
        mock_pipeline_service_create,
        job_spec,
        mock_load_yaml_and_json,
        mock_model_comparison_job_get,
    ):
        aiplatform.init(project=_TEST_PROJECT)

        model_comparison_job.ModelComparisonJob(
            comparison_pipeline_run_name=_TEST_PIPELINE_JOB_NAME
        )

        mock_model_comparison_job_get.assert_called_once_with(
            name=_TEST_PIPELINE_JOB_NAME, retry=base._DEFAULT_RETRY
        )