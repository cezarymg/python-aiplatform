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
from google.auth import credentials as auth_credentials
from google.protobuf import field_mask_pb2

from google.cloud.aiplatform import base
from google.cloud.aiplatform import initializer
from google.cloud.aiplatform import utils
from google.cloud.aiplatform import pipeline_jobs
from google.cloud.aiplatform import pipeline_service

from typing import (
    Any,
    Callable,
    Dict,
    List,
    Iterable,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

_LOGGER = base.Logger(__name__)

_MODEL_EVAL_PIPELINE_TEMPLATE = "gs://sara-vertex-demos-bucket/test_template.json"

class ModelEvaluationJob(pipeline_service.VertexAiPipelineBasedService):

    @property
    def _template_ref(self) -> str:
        return _MODEL_EVAL_PIPELINE_TEMPLATE

    @property
    def backing_pipeline_job(self) -> pipeline_jobs.PipelineJob:
        return super().backing_pipeline_job

    @property
    def pipeline_console_uri(self) -> str:
        return super().pipeline_console_uri

    @property
    def metadata_output_artifact(self) -> Optional[str]:
        return super().metadata_output_artifact

    def __init__(
        self,
        evaluation_pipeline_run: str,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ):
        """Retrieves a ModelEvaluationJob and instantiates its representation."""

        super().__init__(
            pipeline_job_id=evaluation_pipeline_run,
        )

    @classmethod
    def run(
        cls,
        # model,
        # gcs_source_uris,
        # prediction_type,
        # prediction_format,
    ) -> "ModelEvaluationJob":

        # TODO: format template params from gcs_source_uris, prediction_type, prediction_format
        template_params = {"text": "Hello", "emoji_str": "sparkles"}
        
        eval_pipeline_run = cls._create_and_submit_pipeline_job(
            cls,
            template_ref=_MODEL_EVAL_PIPELINE_TEMPLATE,
            template_params=template_params,
            project = initializer.global_config.project,
            location = initializer.global_config.location,
            credentials = initializer.global_config.credentials,
        )

        # eval_pipeline_run.submit()

        model_eval_job_resource = cls.__new__(cls)
        
        model_eval_job_resource.backing_pipeline_job = eval_pipeline_run
        model_eval_job_resource.pipeline_console_uri = eval_pipeline_run._dashboard_uri

        return model_eval_job_resource