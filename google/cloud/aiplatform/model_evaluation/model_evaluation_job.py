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
from typing import Optional, List, Union

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

# TODO: update this with the final gcs pipeline template urls
# First 2 are for automl tabular models, the others are for everything else
_MODEL_EVAL_PIPELINE_TEMPLATES = {
    "automl_tabular_without_feature_attribution": "gs://vertex-evaluation-templates/20220831_1916_test/evaluation_automl_tabular_pipeline.json",
    "automl_tabular_with_feature_attribution": "gs://vertex-evaluation-templates/20220831_1916_test/evaluation_automl_tabular_feature_attribution_pipeline.json",
    "other_without_feature_attribution": "gs://vertex-evaluation-templates/20220831_1916_test/evaluation_pipeline.json",
    "other_with_feature_attribution": "gs://vertex-evaluation-templates/20220831_1916_test/evaluation_feature_attribution_pipeline.json",
}

_EXPERIMENTAL_EVAL_PIPELINE_TEMPLATES = {
    "automl_tabular_without_feature_attribution": "gs://vertex-evaluation-templates/experimental/evaluation_automl_tabular_pipeline.json",
    "automl_tabular_with_feature_attribution": "gs://vertex-evaluation-templates/experimental/evaluation_automl_tabular_feature_attribution_pipeline.json",
    "other_without_feature_attribution": "gs://vertex-evaluation-templates/experimental/evaluation_pipeline.json",
    "other_with_feature_attribution": "gs://vertex-evaluation-templates/experimental/evaluation_feature_attribution_pipeline.json",
}

class _ModelEvaluationJob(pipeline_based_service._VertexAiPipelineBasedService):

    _template_ref = _MODEL_EVAL_PIPELINE_TEMPLATES

    _creation_log_message = "Created PipelineJob for your Model Evaluation."

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
        evaluation_pipeline_run_name: str,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ):
        """Retrieves a ModelEvaluationJob and instantiates its representation.
        Example Usage:
            my_evaluation = aiplatform.ModelEvaluationJob(
                pipeline_job_name = "projects/123/locations/us-central1/pipelineJobs/456"
            )
            my_evaluation = aiplatform.ModelEvaluationJob(
                pipeline_job_name = "456"
            )
        Args:
            evaluation_pipeline_run_name (str):
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
            pipeline_job_name=evaluation_pipeline_run_name,
            project=project,
            location=location,
            credentials=credentials,
        )

    @staticmethod
    def _get_template_url(
        model_type: str,
        feature_attributions: bool,
        use_experimental_templates: bool,
    ) -> str:
        """Gets the pipeline template URL for this model evaluation job given the type of data
        used to train the model and whether feature attributions should be generated.

        Args:
            data_type (str):
                Required. The type of data used to train the model.
            feature_attributions (bool):
                Required. Whether this evaluation job should generate feature attributions.

        Returns:
            (str): The pipeline template URL to use for this model evaluation job.
        """
        template_type = model_type

        if feature_attributions:
            template_type += "_with_feature_attribution"
        else:
            template_type += "_without_feature_attribution"

        if use_experimental_templates:
            return _EXPERIMENTAL_EVAL_PIPELINE_TEMPLATES[template_type]
        else:
            return _ModelEvaluationJob._template_ref[template_type]

    @classmethod
    def submit(
        cls,
        model_name: Union[str, "aiplatform.Model"],
        prediction_type: str,
        target_column_name: str,
        pipeline_root: str,
        model_type: str,
        gcs_source_uris: Optional[List[str]] = None,
        bigquery_source_uri: Optional[str] = None,
        class_names: Optional[List[str]] = None,
        key_columns: Optional[List[str]] = None,
        generate_feature_attributions: Optional[bool] = False,
        instances_format: Optional[str] = "jsonl",
        pipeline_display_name: Optional[str] = None,
        evaluation_metrics_display_name: Optional[str] = None,
        job_id: Optional[str] = None,
        service_account: Optional[str] = None,
        network: Optional[str] = None,
        encryption_spec_key_name: Optional[str] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
        experiment: Optional[Union[str, "aiplatform.Experiment"]] = None,
        use_experimental_templates: Optional[bool] = False,
    ) -> "_ModelEvaluationJob":
        """Submits a Model Evaluation Job using aiplatform.PipelineJob and returns
        the ModelEvaluationJob resource.

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
            model_name (Union[str, "aiplatform.Model"]):
                Required. An instance of aiplatform.Model or a fully-qualified model resource name or model ID to run the evaluation
                job on. Example: "projects/123/locations/us-central1/models/456" or
                "456" when project and location are initialized or passed.
            prediction_type (str):
                Required. The type of prediction performed by the Model. One of "classification" or "regression".
            target_column_name (str):
                Required. The name of your prediction column.
            data_source_uris (List[str]):
                Required. A list of data files containing the ground truth data to use for this evaluation job.
                These files should contain your model's prediction column. Currently only GCS urls are supported,
                for example: "gs://path/to/your/data.csv".
            pipeline_root (str):
                Required. The GCS directory to store output from the model evaluation PipelineJob.
            model_type (str):
                Required. One of "automl_tabular" or "other". This determines the Model Evaluation template used by this PipelineJob.
            gcs_source_uris (List[str]):
                Optional. A list of Cloud Storage data files containing the ground truth data to use for this
                evaluation job. These files should contain your model's prediction column. Currently only Google Cloud Storage
                urls are supported, for example: "gs://path/to/your/data.csv". The provided data files must be
                either CSV or JSONL. One of `gcs_source_uris` or `bigquery_source_uri` is required.
            bigquery_source_uri (str):
                Optional. A bigquery table URI containing the ground truth data to use for this evaluation job. This uri should
                be in the format 'bq://my-project-id.dataset.table'. One of `gcs_source_uris` or `bigquery_source_uri` is
                required.
            class_names (List[str]):
                Optional. For custom (non-AutoML) classification models, a list of possible class names, in the
                same order that predictions are generated. This argument is required when prediction_type is 'classification'.
                For example, in a classification model with 3 possible classes that are outputted in the format: [0.97, 0.02, 0.01]
                with the class names "cat", "dog", and "fish", the value of `class_names` should be `["cat", "dog", "fish"]` where
                the class "cat" corresponds with 0.97 in the example above.
            key_columns (str):
                Optional. The column headers in the data files provided to gcs_source_uris, in the order the columns
                appear in the file. This argument is required for custom models and AutoML Vision, Text, and Video models.
            generate_feature_attributions (boolean):
                Optional. Whether the model evaluation job should generate feature attributions. Defaults to False if not specified.
            instances_format (str):
                The format in which instances are given, must be one of the Model's supportedInputStorageFormats. If not set, defaults to "jsonl".
            pipeline_display_name (str)
                Optional. The user-defined name of the PipelineJob created by this Pipeline Based Service.
            evaluation_metrics_display_name (str)
                Optional. The user-defined name of the evaluation metrics resource uploaded to Vertex in the evaluation pipeline job.
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
            (ModelEvaluationJob): Instantiated represnetation of the model evaluation job.
        """

        if isinstance(model_name, aiplatform.Model):
            model_resource_name = model_name.resource_name
        else:
            model_resource_name = model_name

        if not pipeline_display_name:
            pipeline_display_name = cls._generate_display_name()

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

        if bigquery_source_uri:
            template_params["batch_predict_predictions_format"] = "bigquery"
            template_params["batch_predict_bigquery_source_uri"] = bigquery_source_uri
        elif gcs_source_uris:
            template_params["batch_predict_gcs_source_uris"] = gcs_source_uris

        if prediction_type == "classification" and model_type == "other" and class_names is not None:
            template_params["evaluation_class_names"] = class_names
        
        # If the user provides a SA, use it for the Dataflow job as well
        if service_account is not None:
            template_params["dataflow_service_account"] = service_account

        template_url = cls._get_template_url(
                model_type, generate_feature_attributions, use_experimental_templates
        )

        if use_experimental_templates:
            _LOGGER.info(f"Using experimental pipeline template: {template_url}")

        eval_pipeline_run = cls._create_and_submit_pipeline_job(
            template_params=template_params,
            template_path=template_url,
            pipeline_root=pipeline_root,
            display_name=pipeline_display_name,
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
            f"{_ModelEvaluationJob._creation_log_message} View it in the console: {eval_pipeline_run.pipeline_console_uri}"
        )

        return eval_pipeline_run

    def get_model_evaluation(
        self,
    ) -> Optional["model_evaluation.ModelEvaluation"]:
        """Gets the ModelEvaluation created by this ModelEvlauationJob.

        Returns:
            aiplatform.ModelEvaluation: Instantiated representation of the ModelEvaluation resource.
        Raises:
            RuntimeError: If the ModelEvaluationJob pipeline failed.
        """
        eval_job_state = self.backing_pipeline_job.state

        if eval_job_state in pipeline_jobs._PIPELINE_ERROR_STATES:
            raise RuntimeError(
                f"Evaluation job failed. For more details see the logs: {self.pipeline_console_uri}"
            )
        elif eval_job_state not in pipeline_jobs._PIPELINE_COMPLETE_STATES:
            _LOGGER.info(
                f"Your evaluation job is still in progress. For more details see the logs {self.pipeline_console_uri}"
            )
        else:
            for component in self.backing_pipeline_job.task_details:
                for metadata_key in component.execution.metadata:
                    if (
                        metadata_key == "output:gcp_resources"
                        and json.loads(component.execution.metadata[metadata_key])[
                            "resources"
                        ][0]["resourceType"]
                        == "ModelEvaluation"
                    ):
                        eval_resource_uri = json.loads(
                            component.execution.metadata[metadata_key]
                        )["resources"][0]["resourceUri"]
                        eval_resource_name = eval_resource_uri.split("v1/")[1]

                        eval_resource = model_evaluation.ModelEvaluation(
                            evaluation_name=eval_resource_name
                        )

                        eval_resource._gca_resource = eval_resource._get_gca_resource(
                            resource_name=eval_resource_name
                        )

                        return eval_resource

    def wait(self):
        """Wait for thie PipelineJob to complete."""
        pipeline_run = super().backing_pipeline_job

        if pipeline_run._latest_future is None:
            pipeline_run._block_until_complete()
        else:
            pipeline_run.wait()
