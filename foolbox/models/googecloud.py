from __future__ import absolute_import
import numpy as np
import logging
from PIL import Image
import os
import io
from .base import Model

from google.cloud import visieon
from google.cloud.vision import types

class GoogleCloudModel(Model):
    """Creates a :class:`Model` instance from the `google cloud vision` API.

    """
    def __init__(
            self,
            bounds,
            channel_axis=3,
            preprocessing=(0, 1),
            predicts='probabilities'):

        super(GoogleCloudModel, self).__init__(bounds=bounds,
                                               channel_axis=channel_axis,
                                               preprocessing=preprocessing)

        self._temp_dir = '/var/run'
        self._client = vision.ImageAnnotatorClient()
        self._num_classes = 3

    def batch_predictions(self, images):
        predictions = []
        for image in images:
            predictions.append(self.predictions(image))
        return np.array(predictions)

    def predictions(self, image):
        # save image to disk and read as byte
        image_pil = Image.fromarray(image)
        file_name = os.path.join(self._temp_dir, 'temp.png')
        image_pil.save(file_name)
        with io.open(file_name, 'rb') as image_file:
            content = image_file.read()
        image_pb = types.Image(content=content)
        # Performs label detection on the image file
        response = self._client.label_detection(image=image_pb, max_results=10)
        gcp_labels = response.label_annotations
        return self.process_labels(gcp_labels)

    def process_labels(self, gcp_labels):
        # TODO(tong): implement this method.
        # return logits of three classes
        # the first one is the highest score of
        return np.zeros(self._num_classes)

    def num_classes(self):
        return self._num_classes

