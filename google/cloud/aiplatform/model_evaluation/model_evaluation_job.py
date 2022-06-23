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

# TODO: update this with the final gcs pipeline template urls and add templates for unstructured data when they are available

_MODEL_EVAL_PIPELINE_TEMPLATES = {
    "tabular_without_feature_attribution": "gs://vertex-evaluation-templates/20220616_2055/evaluation_tabular_pipeline.json",
    "tabular_with_feature_attribution": "gs://vertex-evaluation-templates/20220616_2055/evaluation_tabular_feature_attribution_pipeline.json",
    # "unstructured_without_feature_attribution": "TODO",
    # "unstructured_with_feature_attribution": "TODO",
}


class ModelEvaluationJob(pipeline_based_service._VertexAiPipelineBasedService):

    _template_ref = _MODEL_EVAL_PIPELINE_TEMPLATES

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
        evaluation_pipeline_run: str,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ):
        """Retrieves a ModelEvaluationJob and instantiates its representation.
        Example Usage:
            my_evaluation = aiplatform.ModelEvaluationJob(
                pipeline_job_id = "projects/123/locations/us-central1/pipelineJobs/456"
            )
            my_evaluation = aiplatform.ModelEvaluationJob(
                pipeline_job_id = "456"
            )
        Args:
            evaluation_pipeline_run (str):
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
            pipeline_job_id=evaluation_pipeline_run,
            project=project,
            location=location,
            credentials=credentials,
        )

    @staticmethod
    def _get_template_url(data_type, feature_attributions) -> str:

        template_type = data_type

        if feature_attributions:
            template_type += "_with_feature_attribution"
        else:
            template_type += "_without_feature_attribution"

        return ModelEvaluationJob._template_ref[template_type]

    @classmethod
    def submit(
        cls,
        model_name: str,
        prediction_type: str,
        target_column_name: str,
        gcs_source_uris: List[str],
        pipeline_root: str,
        data_type: str,
        generate_feature_attributions: Optional[bool] = False,
        instances_format: Optional[str] = "jsonl",
        display_name: Optional[str] = None,
        job_id: Optional[str] = None,
        service_account: Optional[str] = None,
        network: Optional[str] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
        experiment: Optional[Union[str, "aiplatform.Experiment"]] = None,
    ) -> "ModelEvaluationJob":
        """Submits a Model Evaluation Job using aiplatform.PipelineJob and returns
        the ModelEvaluationJob resource.

        Example usage:
        my_evaluation = ModelEvaluationJob.submit(
            model="projects/123/locations/us-central1/models/456",
            prediction_type="classification",
            pipeline_root="gs://my-pipeline-bucket/runpath",
            gcs_source_uris=["gs://test-prediction-data"],
            target_column_name=["prediction_class"],
            instances_format="jsonl",
        )

        my_evaluation = ModelEvaluationJob.submit(
            model="projects/123/locations/us-central1/models/456",
            prediction_type="regression",
            pipeline_root="gs://my-pipeline-bucket/runpath",
            gcs_source_uris=["gs://test-prediction-data"],
            target_column_name=["price"],
            instances_format="jsonl",
        )
        Args:
            model_name (str):
                Required. A fully-qualified model resource name or model ID to run the evaluation
                job on. Example: "projects/123/locations/us-central1/models/456" or
                "456" when project and location are initialized or passed.
            prediction_type (str):
                Required. The type of prediction performed by the Model. One of "classification" or "regression".
            target_column_name (str):
                Required. The name of your prediction column.
            gcs_source_uris (List[str]):
                Required. A list of GCS URIs containing your input data for batch prediction. This is used to provide
                ground truth for each prediction instance, and should include a label column with the ground truth value.
            pipeline_root (str):
                Required. The GCS directory to store output from the model evaluation PipelineJob.
            instances_format (str):
                The format in which instances are given, must be one of the Model's supportedInputStorageFormats. If not set, defaults to "jsonl".
            display_name (str)
                Optional. The user-defined name of the PipelineJob created by this Pipeline Based Service.
            job_id (str):
                Optional. The unique ID of the job run.
                If not specified, pipeline name + timestamp will be used.
            service_account (str):
                Specifies the service account for workload run-as account.
                Users submitting jobs must have act-as permission on this run-as account.
            network (str):
                The full name of the Compute Engine network to which the job
                should be peered. For example, projects/12345/global/networks/myVPC.
                Private services access must already be configured for the network.
                If left unspecified, the job is not peered with any network.
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

        if not display_name:
            display_name = cls._generate_display_name()

        template_params = {
            "batch_predict_gcs_source_uris": gcs_source_uris,
            "batch_predict_instances_format": instances_format,
            "model_name": model_name,
            "prediction_type": prediction_type,
            "project": project or initializer.global_config.project,
            "location": location or initializer.global_config.location,
            "root_dir": pipeline_root,
            "target_column_name": target_column_name,
        }

        eval_pipeline_run = cls._create_and_submit_pipeline_job(
            template_params=template_params,
            template_path=cls._get_template_url(
                data_type, generate_feature_attributions
            ),
            pipeline_root=pipeline_root,
            display_name=display_name,
            job_id=job_id,
            service_account=service_account,
            network=network,
            project=project,
            location=location,
            credentials=credentials,
            experiment=experiment,
        )

        return eval_pipeline_run

    # TODO: not sure if this is the best way to get the eval resource uri
    def get_model_evaluation(
        self,
    ) -> Optional["model_evaluation.ModelEvaluation"]:
        """Creates a ModelEvaluation resource and instantiates its representation.

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
            _LOGGER.info(
                "Your evaluation job ran successfully. Creating Model Evaluation resource..."
            )

            # TODO: is there a better way to get the eval resource name?
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

    @classmethod
    def list(
        cls,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ) -> List["model_evaluation.ModelEvaluationJob"]:
        """Returns a list of all ModelEvaluationJob resources associated with this project.
        Args:
            project (str):
                Optional. The project to retrieve the ModelEvaluationJob resources from. If not set,
                the project set in aiplatform.init will be used.
            location (str):
                Optional. Location to retrieve the ModelEvaluationJob resources from. If not set,
                location set in aiplatform.init will be used.
            credentials (auth_credentials.Credentials):
                Optional. Custom credentials to use to retrieve the ModelEvaluationJob resources from.
                Overrides credentials set in aiplatform.init.
        Returns:
            (List[ModelEvaluationJob]):
                A list of ModelEvaluationJob resource objects.
        """
        return super().list(
            project=project,
            location=location,
            credentials=credentials,
        )
