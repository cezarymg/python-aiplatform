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


import datetime
import yaml
import json
import pytest
from unittest import mock

from google.auth import credentials as auth_credentials
from google.protobuf import json_format
from google.cloud import storage
from urllib import request


from google.cloud import aiplatform
from google.cloud.aiplatform import base
from google.cloud.aiplatform.metadata import constants
from google.cloud.aiplatform.utils import gcs_utils

from google.cloud.aiplatform_v1.services.pipeline_service import (
    client as pipeline_service_client_v1,
)
from google.cloud.aiplatform_v1.types import (
    pipeline_job as gca_pipeline_job_v1,
)
from google.cloud.aiplatform_v1.types import (
    pipeline_state as gca_pipeline_state_v1,
)

from google.cloud.aiplatform._pipeline_based_service import (
    pipeline_based_service,
)

from google.cloud.aiplatform_v1 import Execution as GapicExecution
from google.cloud.aiplatform_v1 import MetadataServiceClient

from google.cloud.aiplatform.model_comparison import model_comparison_job


_TEST_PROJECT = "test-project"
_TEST_LOCATION = "us-central1"
_TEST_PIPELINE_JOB_DISPLAY_NAME = "sample-pipeline-job-display-name"
_TEST_PIPELINE_JOB_ID = "sample-test-pipeline-202111111"
_TEST_GCS_BUCKET_NAME = "my-bucket"
_TEST_BQ_DATASET = "bq://test-data.train"
_TEST_CREDENTIALS = auth_credentials.AnonymousCredentials()
_TEST_SERVICE_ACCOUNT = "abcde@my-project.iam.gserviceaccount.com"
_TEST_COMPONENT_IDENTIFIER = "fpc-structured-data"
_TEST_PIPELINE_NAME_IDENTIFIER = "model-comparison"
_TEST_INVALID_PIPELINE_NAME_IDENTIFIER = "not-a-valid-pipeline-name"
_TEST_PIPELINE_CREATE_TIME = datetime.datetime.now()


_TEST_TEMPLATE_PATH = f"gs://{_TEST_GCS_BUCKET_NAME}/job_spec.json"
_TEST_TEMPLATE_REF = {"test_pipeline_type": _TEST_TEMPLATE_PATH}
_TEST_PIPELINE_ROOT = f"gs://{_TEST_GCS_BUCKET_NAME}/pipeline_root"
_TEST_PARENT = f"projects/{_TEST_PROJECT}/locations/{_TEST_LOCATION}"
_TEST_NETWORK = f"projects/{_TEST_PROJECT}/global/networks/{_TEST_PIPELINE_JOB_ID}"

_TEST_PIPELINE_JOB_NAME = f"projects/{_TEST_PROJECT}/locations/{_TEST_LOCATION}/pipelineJobs/{_TEST_PIPELINE_JOB_ID}"
_TEST_INVALID_PIPELINE_JOB_NAME = (
    f"prj/{_TEST_PROJECT}/locations/{_TEST_LOCATION}/{_TEST_PIPELINE_JOB_ID}"
)

# executions: this is used in test_list_pipeline_based_service
_TEST_EXECUTION_PARENT = (
    f"projects/{_TEST_PROJECT}/locations/{_TEST_LOCATION}/metadataStores/default"
)

_TEST_RUN = "run-1"
_TEST_OTHER_RUN = "run-2"
_TEST_EXPERIMENT = "test-experiment"
_TEST_EXECUTION_ID = f"{_TEST_EXPERIMENT}-{_TEST_RUN}"
_TEST_EXECUTION_NAME = f"{_TEST_EXECUTION_PARENT}/executions/{_TEST_EXECUTION_ID}"


_TEST_OTHER_EXECUTION_ID = f"{_TEST_EXPERIMENT}-{_TEST_OTHER_RUN}"
_TEST_OTHER_EXECUTION_NAME = (
    f"{_TEST_EXECUTION_PARENT}/executions/{_TEST_OTHER_EXECUTION_ID}"
)

# execution metadata parameters: used in test_list_pipeline_based_service
_TEST_PARAM_KEY_1 = "learning_rate"
_TEST_PARAM_KEY_2 = "dropout"
_TEST_PIPELINE_PARAM_KEY = "pipeline_job_resource_name"
_TEST_PARAMS = {
    _TEST_PARAM_KEY_1: 0.01,
    _TEST_PARAM_KEY_2: 0.2,
    _TEST_PIPELINE_PARAM_KEY: _TEST_PIPELINE_JOB_NAME,
}
_TEST_OTHER_PARAMS = {_TEST_PARAM_KEY_1: 0.02, _TEST_PARAM_KEY_2: 0.3}


_TEST_PIPELINE_PARAMETER_VALUES = {
    "data_source_bigquery_table_path": _TEST_BQ_DATASET,
    "data_source_csv_filenames": "",
    "experiment": _TEST_EXPERIMENT,
    "location": _TEST_LOCATION,
    "root_dir": _TEST_GCS_BUCKET_NAME,
    "problem_type": "forecasting",
    "project": _TEST_PROJECT,
    "training_jobs": {},
}

_TEST_PIPELINE_SPEC_JSON = json.dumps(
    {
        "pipelineInfo": {"name": _TEST_PIPELINE_NAME_IDENTIFIER},
        "root": {
            "dag": {"tasks": {}},
            "inputDefinitions": {
                "parameters": {
                    "data_source_bigquery_table_path": {"type": "STRING"},
                    "data_source_csv_filenames": {"type": "STRING"},
                    "experiment": {"type": "STRING"},
                    "location": {"type": "STRING"},
                    "problem_type": {"type": "STRING"},
                    "project": {"type": "STRING"},
                    "root_dir": {"type": "STRING"},
                    "training_jobs": {"type": "STRING"},
                }
            },
        },
        "schemaVersion": "2.0.0",
        "sdkVersion": "kfp-1.8.12",
        "components": {},
    }
)

_TEST_PIPELINE_JOB = json.dumps(
    {
        "runtimeConfig": {"parameterValues": _TEST_PIPELINE_PARAMETER_VALUES},
        "pipelineSpec": json.loads(_TEST_PIPELINE_SPEC_JSON),
    }
)


@pytest.fixture
def mock_model_comparison_job_create():
    with mock.patch.object(
        pipeline_service_client_v1.PipelineServiceClient, "create_pipeline_job"
    ) as mock_create_pipeline_job:
        mock_create_pipeline_job.return_value = make_pipeline_job(
            gca_pipeline_state_v1.PipelineState.PIPELINE_STATE_SUCCEEDED
        )
        yield mock_create_pipeline_job


@pytest.fixture
def mock_model_comparison_job_get():
    with mock.patch.object(
        pipeline_service_client_v1.PipelineServiceClient, "get_pipeline_job"
    ) as mock_get_pipeline_job:
        mock_get_pipeline_job.side_effect = [
            make_pipeline_job(
                gca_pipeline_state_v1.PipelineState.PIPELINE_STATE_RUNNING
            ),
            make_pipeline_job(
                gca_pipeline_state_v1.PipelineState.PIPELINE_STATE_SUCCEEDED
            ),
            make_pipeline_job(
                gca_pipeline_state_v1.PipelineState.PIPELINE_STATE_SUCCEEDED
            ),
            make_pipeline_job(
                gca_pipeline_state_v1.PipelineState.PIPELINE_STATE_SUCCEEDED
            ),
            make_pipeline_job(
                gca_pipeline_state_v1.PipelineState.PIPELINE_STATE_SUCCEEDED
            ),
            make_pipeline_job(
                gca_pipeline_state_v1.PipelineState.PIPELINE_STATE_SUCCEEDED
            ),
            make_pipeline_job(
                gca_pipeline_state_v1.PipelineState.PIPELINE_STATE_SUCCEEDED
            ),
            make_pipeline_job(
                gca_pipeline_state_v1.PipelineState.PIPELINE_STATE_SUCCEEDED
            ),
            make_pipeline_job(
                gca_pipeline_state_v1.PipelineState.PIPELINE_STATE_SUCCEEDED
            ),
        ]

        yield mock_get_pipeline_job


@pytest.fixture
def mock_load_yaml_and_json(job_spec):
    with mock.patch.object(request, "urlopen") as mock_load_yaml_and_json:
        mock_load_yaml_and_json.return_value.read.return_value.decode.return_value = (
            job_spec.encode()
        )
        yield mock_load_yaml_and_json


@pytest.fixture
def get_execution_mock():
    with mock.patch.object(
        MetadataServiceClient, "get_execution"
    ) as get_execution_mock:
        get_execution_mock.return_value = GapicExecution(
            name=_TEST_EXECUTION_NAME,
            display_name=_TEST_RUN,
            schema_title=constants.SYSTEM_RUN,
            schema_version=constants.SCHEMA_VERSIONS[constants.SYSTEM_RUN],
            metadata={"component_type": _TEST_COMPONENT_IDENTIFIER},
        )
        yield get_execution_mock


@pytest.fixture
def mock_pipeline_bucket_exists():
    def mock_create_gcs_bucket_for_pipeline_artifacts_if_it_does_not_exist(
        output_artifacts_gcs_dir=None,
        service_account=None,
        project=None,
        location=None,
        credentials=None,
    ):
        output_artifacts_gcs_dir = (
            output_artifacts_gcs_dir
            or gcs_utils.generate_gcs_directory_for_pipeline_artifacts(
                project=project,
                location=location,
            )
        )
        return output_artifacts_gcs_dir

    with mock.patch(
        "google.cloud.aiplatform.utils.gcs_utils.create_gcs_bucket_for_pipeline_artifacts_if_it_does_not_exist",
        wraps=mock_create_gcs_bucket_for_pipeline_artifacts_if_it_does_not_exist,
    ) as mock_context:
        yield mock_context


def make_pipeline_job(state):
    return gca_pipeline_job_v1.PipelineJob(
        name=_TEST_PIPELINE_JOB_NAME,
        state=state,
        create_time=_TEST_PIPELINE_CREATE_TIME,
        pipeline_spec=json.loads(_TEST_PIPELINE_SPEC_JSON),
        job_detail=gca_pipeline_job_v1.PipelineJobDetail(
            task_details=[
                gca_pipeline_job_v1.PipelineTaskDetail(
                    task_id=123,
                    execution=GapicExecution(
                        name=_TEST_EXECUTION_NAME,
                        display_name=_TEST_RUN,
                        schema_title=constants.SYSTEM_RUN,
                        schema_version=constants.SCHEMA_VERSIONS[constants.SYSTEM_RUN],
                        metadata={"component_type": _TEST_COMPONENT_IDENTIFIER},
                    ),
                ),
            ],
        ),
    )


@pytest.mark.usefixtures("google_auth_mock")
class TestModelComparisonJob:
    def test_init_model_comparison_job(
        self,
        get_execution_mock,
        mock_model_comparison_job_get,
    ):
        aiplatform.init(project=_TEST_PROJECT)

        model_comparison_job.ModelComparisonJob(
            comparison_pipeline_run_name=_TEST_PIPELINE_JOB_NAME
        )

        mock_model_comparison_job_get.assert_called_with(
            name=_TEST_PIPELINE_JOB_NAME, retry=base._DEFAULT_RETRY
        )

        assert mock_model_comparison_job_get.call_count == 2

    def test_init_model_evaluation_job_with_invalid_pipeline_job_name_raises(self):
        aiplatform.init(
            project=_TEST_PROJECT,
            location=_TEST_LOCATION,
            credentials=_TEST_CREDENTIALS,
        )

        with pytest.raises(ValueError):
            model_comparison_job.ModelComparisonJob(
                comparison_pipeline_run_name=_TEST_INVALID_PIPELINE_JOB_NAME,
            )

    @pytest.mark.parametrize(
        "job_spec",
        [_TEST_PIPELINE_SPEC_JSON],
    )
    def test_model_comparison_job_submit(
        self,
        job_spec,
        mock_load_yaml_and_json,
        mock_model_comparison_job_get,
        mock_model_comparison_job_create,
        mock_pipeline_bucket_exists,
    ):
        aiplatform.init(
            project=_TEST_PROJECT,
            location=_TEST_LOCATION,
            credentials=_TEST_CREDENTIALS,
            staging_bucket=_TEST_GCS_BUCKET_NAME,
        )

        test_model_comparison_job = model_comparison_job.ModelComparisonJob.submit(
            data_source_bigquery_table_path=_TEST_BQ_DATASET,
            data_source_csv_filenames="",
            experiment=_TEST_EXPERIMENT,
            location=_TEST_LOCATION,
            pipeline_root=_TEST_GCS_BUCKET_NAME,
            problem_type="forecasting",
            project=_TEST_PROJECT,
            training_jobs={},
            job_id=_TEST_PIPELINE_JOB_ID,
            comparison_pipeline_display_name=_TEST_PIPELINE_JOB_DISPLAY_NAME
        )

        test_model_comparison_job.wait()

        expected_runtime_config_dict = {
            "gcsOutputDirectory": _TEST_GCS_BUCKET_NAME,
            "parameters": {
                "problem_type": {"stringValue": "forecasting"},
                "project": {"stringValue": _TEST_PROJECT},
                "location": {"stringValue": _TEST_LOCATION},
                "root_dir": {"stringValue": _TEST_GCS_BUCKET_NAME},
                "data_source_bigquery_table_path": {"stringValue": _TEST_BQ_DATASET},
                "data_source_csv_filenames": {"stringValue": ""},
                "experiment": {"stringValue": _TEST_EXPERIMENT},
            },
        }

        runtime_config = gca_pipeline_job_v1.PipelineJob.RuntimeConfig()._pb
        json_format.ParseDict(expected_runtime_config_dict, runtime_config)

        job_spec = yaml.safe_load(job_spec)
        pipeline_spec = job_spec.get("pipelineSpec") or job_spec

        # Construct expected request
        expected_gapic_pipeline_job = gca_pipeline_job_v1.PipelineJob(
            display_name=_TEST_PIPELINE_JOB_DISPLAY_NAME,
            pipeline_spec={
                "components": {},
                "pipelineInfo": pipeline_spec["pipelineInfo"],
                "root": pipeline_spec["root"],
                "schemaVersion": "2.0.0",
                "sdkVersion": "kfp-1.8.12",
            },
            runtime_config=runtime_config,
        )

        mock_model_comparison_job_create.assert_called_with(
            parent=_TEST_PARENT,
            pipeline_job=expected_gapic_pipeline_job,
            pipeline_job_id=_TEST_PIPELINE_JOB_ID,
            timeout=None,
        )

        assert mock_model_comparison_job_get.called_once
