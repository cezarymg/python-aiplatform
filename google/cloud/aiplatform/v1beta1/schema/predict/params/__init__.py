# -*- coding: utf-8 -*-

# Copyright 2020 Google LLC
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
from google.cloud.aiplatform_helpers import add_methods_to_classes_in_package
import google.cloud.aiplatform.v1beta1.schema.predict.params_v1beta1.types as pkg

from google.cloud.aiplatform.v1beta1.schema.predict.params_v1beta1.types.image_classification import (
    ImageClassificationPredictionParams,
)
from google.cloud.aiplatform.v1beta1.schema.predict.params_v1beta1.types.image_object_detection import (
    ImageObjectDetectionPredictionParams,
)
from google.cloud.aiplatform.v1beta1.schema.predict.params_v1beta1.types.image_segmentation import (
    ImageSegmentationPredictionParams,
)
from google.cloud.aiplatform.v1beta1.schema.predict.params_v1beta1.types.video_action_recognition import (
    VideoActionRecognitionPredictionParams,
)
from google.cloud.aiplatform.v1beta1.schema.predict.params_v1beta1.types.video_classification import (
    VideoClassificationPredictionParams,
)
from google.cloud.aiplatform.v1beta1.schema.predict.params_v1beta1.types.video_object_tracking import (
    VideoObjectTrackingPredictionParams,
)

__all__ = (
    "ImageClassificationPredictionParams",
    "ImageObjectDetectionPredictionParams",
    "ImageSegmentationPredictionParams",
    "VideoActionRecognitionPredictionParams",
    "VideoClassificationPredictionParams",
    "VideoObjectTrackingPredictionParams",
)
add_methods_to_classes_in_package(pkg)
