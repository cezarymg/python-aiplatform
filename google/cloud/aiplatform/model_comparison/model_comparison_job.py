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

from re import template
from typing import Optional, List, Union, Dict

from google.auth import credentials as auth_credentials

from google.cloud import aiplatform
from google.cloud.aiplatform import base
from google.cloud.aiplatform import initializer
from google.cloud.aiplatform._pipeline_based_service import pipeline_based_service
from google.cloud.aiplatform import model_evaluation
from google.cloud.aiplatform import pipeline_jobs

from google.cloud.aiplatform.compat.types import (
    pipeline_state_v1 as gca_pipeline_state_v1,
)

import json

_LOGGER = base.Logger(__name__)

MODEL_COMPARISON_PIPELINE = 'model_comparison'
BQML_ARIMA_TRAIN_PIPELINE = 'bqml_arima_train'
AUTOML_TABULAR_PIPELINE = 'automl_tabular'

_PIPELINE_TEMPLATES = {
    MODEL_COMPARISON_PIPELINE: "gs://cezarym-staging/model_comparison_pipeline.json",
    BQML_ARIMA_TRAIN_PIPELINE: "gs://cezarym-staging/bqml_arima_train_pipeline.json",
    AUTOML_TABULAR_PIPELINE: "gs://cezarym-staging/automl_tabular_pipeline.json",
}

class ModelComparisonJob(pipeline_based_service._VertexAiPipelineBasedService):

    _template_ref = _PIPELINE_TEMPLATES

    _creation_log_message = "Created PipelineJob for your Model Comparison."

    @property
    def _metadata_output_artifact(self) -> Optional[str]:
        """The resource uri for the ML Metadata output artifact from the evaluation component of the Model Evaluation pipeline"""
        if self.state == gca_pipeline_state_v1.PipelineState.PIPELINE_STATE_SUCCEEDED:
            for task in self.backing_pipeline_job._gca_resource.job_detail.task_details:
                if (
                    task.task_name.startswith("model-evaluation")
                    and "evaluation_metrics" in task.outputs
                ):
                    return task.outputs["evaluation_metrics"].artifacts[0].name

    def __init__(
        self,
        comparison_pipeline_run_name: str,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ):
        """Retrieves a ModelComparisonJob and instantiates its representation.
        Example Usage:
            my_comparison = aiplatform.ModelComparisonJob(
                comparison_pipeline_run_name = "projects/123/locations/us-central1/pipelineJobs/456"
            )
        Args:
            comparison_pipeline_run_name (str):
                Required. A fully-qualified pipeline job run ID.
                Example: "projects/123/locations/us-central1/pipelineJobs/456" or
                "456" when project and location are initialized or passed.
            project (str):
                Optional. Project to retrieve pipeline job from. If not set, project
                set in aiplatform.init will be used.
            location (str):
                Optional. Location to retrieve pipeline job from. If not set, location
                set in aiplatform.init will be used.
            credentials (auth_credentials.Credentials):
                Optional. Custom credentials to use to retrieve this pipeline job. Overrides
                credentials set in aiplatform.init.
        """
        super().__init__(
            pipeline_job_name=comparison_pipeline_run_name,
            project=project,
            location=location,
            credentials=credentials,
        )

    @staticmethod
    def get_template_url(
        pipeline: str,
    ) -> str:
        """Gets the pipeline template URL for a given pipeline.

        Args:
            pipeline (str):
                Required. Pipeline name.

        Returns:
            (str): The pipeline template URL.
        """

        return ModelComparisonJob._template_ref.get(pipeline)

    @classmethod
    def submit(
        cls,
        problem_type: str,
        training_jobs: Dict[str, Dict[str, Any]],
        data_source_csv_filenames: List[str],
        data_source_bigquery_table_path: str,
        pipeline_root: str,
        job_id: Optional[str] = None,
        service_account: Optional[str] = None,
        network: Optional[str] = None,
        encryption_spec_key_name: Optional[str] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
        experiment: Optional[Union[str, "aiplatform.Experiment"]] = None,
    ) -> "ModelComparisonJob":
        """Submits a Model Comparison Job using aiplatform.PipelineJob and returns
        the ModelComparisonJob resource.

        Example usage:
        my_evaluation = _ModelEvaluationJob.submit(
            model="projects/123/locations/us-central1/models/456",
            prediction_type="classification",
            pipeline_root="gs://my-pipeline-bucket/runpath",
            gcs_source_uris=["gs://test-prediction-data"],
            target_column_name=["prediction_class"],
            instances_format="jsonl",
        )

        my_evaluation = _ModelEvaluationJob.submit(
            model="projects/123/locations/us-central1/models/456",
            prediction_type="regression",
            pipeline_root="gs://my-pipeline-bucket/runpath",
            gcs_source_uris=["gs://test-prediction-data"],
            target_column_name=["price"],
            instances_format="jsonl",
        )
        Args:
            problem_type: The type of problem being solved. Can be one of: regression,
                binary_classification, multiclass_classification, or forecasting
                training_jobs: A dict mapping name to a dict of training job inputs.
            data_source_csv_filenames: Paths to CSVs stored in GCS to use as the dataset
                for all training pipelines. This should be None if
                `data_source_bigquery_table_path` is not None.
            data_source_bigquery_table_path: Path to BigQuery Table to use as the
                dataset for all training pipelines. This should be None if
                `data_source_csv_filenames` is not None.
            pipeline_root (str):
                Required. The GCS directory to store output from the model evaluation PipelineJob.
            job_id (str):
                Optional. The unique ID of the job run.
                If not specified, pipeline name + timestamp will be used.
            service_account (str):
                Specifies the service account for workload run-as account for this Model Evaluation PipelineJob.
                Users submitting jobs must have act-as permission on this run-as account. The service account running
                this Model Evaluation job needs the following permissions: Dataflow Worker, Storage Admin, Vertex AI User.
            network (str):
                The full name of the Compute Engine network to which the job
                should be peered. For example, projects/12345/global/networks/myVPC.
                Private services access must already be configured for the network.
                If left unspecified, the job is not peered with any network.
            encryption_spec_key_name (str):
                Optional. The Cloud KMS resource identifier of the customer managed encryption key used to protect the job. Has the
                form: ``projects/my-project/locations/my-region/keyRings/my-kr/cryptoKeys/my-key``. The key needs to be in the same
                region as where the compute resource is created. If this is set, then all
                resources created by the PipelineJob for this Model Evaluation will be encrypted with the provided encryption key.
                If not specified, encryption_spec of original PipelineJob will be used.
            project (str):
                Optional. The project to run this PipelineJob in. If not set,
                the project set in aiplatform.init will be used.
            location (str):
                Optional. Location to create PipelineJob. If not set,
                location set in aiplatform.init will be used.
            credentials (auth_credentials.Credentials):
                Optional. Custom credentials to use to create the PipelineJob.
                Overrides credentials set in aiplatform.init.
            experiment (Union[str, experiments_resource.Experiment]):
                Optional. The Vertex AI experiment name or instance to associate to the PipelineJob executing
                this model evaluation job.
        Returns:
            (ModelComparisonJob): Instantiated represnetation of the model comparison job.
        """
        if not evaluation_pipeline_display_name:
            evaluation_pipeline_display_name = cls._generate_display_name()

        template_params = {
            "batch_predict_instances_format": instances_format,
            "evaluation_join_keys": key_columns,
            "model_name": model_resource_name,
            "prediction_type": prediction_type,
            "evaluation_display_name": evaluation_metrics_display_name,
            "project": project or initializer.global_config.project,
            "location": location or initializer.global_config.location,
            "root_dir": pipeline_root,
            "target_column_name": target_column_name,
            "encryption_spec_key_name": encryption_spec_key_name,
        }

        # If the user provides a SA, use it for the Dataflow job as well
        if service_account is not None:
            template_params["dataflow_service_account"] = service_account

        template_url = cls._get_template_url(
                model_type, generate_feature_attributions, use_experimental_templates
        )

        comparison_pipeline_run = cls._create_and_submit_pipeline_job(
            template_params=template_params,
            template_path=template_url,
            pipeline_root=pipeline_root,
            display_name=evaluation_pipeline_display_name,
            job_id=job_id,
            service_account=service_account,
            network=network,
            encryption_spec_key_name=encryption_spec_key_name,
            project=project,
            location=location,
            credentials=credentials,
            experiment=experiment,
        )

        _LOGGER.info(
            f"{ModelComparisonJob._creation_log_message} View it in the console: {comparison_pipeline_run.pipeline_console_uri}"
        )

        return comparison_pipeline_run

    # def get_model_evaluation(
    #     self,
    # ) -> Optional["model_evaluation.ModelEvaluation"]:
    #     """Gets the ModelEvaluation created by this ModelEvlauationJob.

    #     Returns:
    #         aiplatform.ModelEvaluation: Instantiated representation of the ModelEvaluation resource.
    #     Raises:
    #         RuntimeError: If the ModelEvaluationJob pipeline failed.
    #     """
    #     eval_job_state = self.backing_pipeline_job.state

    #     if eval_job_state in pipeline_jobs._PIPELINE_ERROR_STATES:
    #         raise RuntimeError(
    #             f"Evaluation job failed. For more details see the logs: {self.pipeline_console_uri}"
    #         )
    #     elif eval_job_state not in pipeline_jobs._PIPELINE_COMPLETE_STATES:
    #         _LOGGER.info(
    #             f"Your evaluation job is still in progress. For more details see the logs {self.pipeline_console_uri}"
    #         )
    #     else:
    #         for component in self.backing_pipeline_job.task_details:
    #             for metadata_key in component.execution.metadata:
    #                 if (
    #                     metadata_key == "output:gcp_resources"
    #                     and json.loads(component.execution.metadata[metadata_key])[
    #                         "resources"
    #                     ][0]["resourceType"]
    #                     == "ModelEvaluation"
    #                 ):
    #                     eval_resource_uri = json.loads(
    #                         component.execution.metadata[metadata_key]
    #                     )["resources"][0]["resourceUri"]
    #                     eval_resource_name = eval_resource_uri.split("v1/")[1]

    #                     eval_resource = model_evaluation.ModelEvaluation(
    #                         evaluation_name=eval_resource_name
    #                     )

    #                     eval_resource._gca_resource = eval_resource._get_gca_resource(
    #                         resource_name=eval_resource_name
    #                     )

    #                     return eval_resource

    def wait(self):
        """Wait for thie PipelineJob to complete."""
        pipeline_run = super().backing_pipeline_job

        if pipeline_run._latest_future is None:
            pipeline_run._block_until_complete()
        else:
            pipeline_run.wait()