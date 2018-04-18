from __future__ import absolute_import
import numpy as np
import logging
from PIL import Image
import os
import io
from .base import Model

from google.cloud import vision
from google.cloud.vision import types

class GoogleCloudModel(Model):
    """Creates a :class:`Model` instance from the `google cloud vision` API.

    """
    def __init__(
            self,
            bounds,
            channel_axis=3,
            preprocessing=(0, 1)):

        super(GoogleCloudModel, self).__init__(bounds=bounds,
                                               channel_axis=channel_axis,
                                               preprocessing=preprocessing)

        self._temp_dir = '/var/run'
        self._client = vision.ImageAnnotatorClient()
        self._num_classes = 3

    def batch_predictions(self, images):
        predictions = np.empty(shape=(len(images), 1), dtype=object)
        for i, image in enumerate(images):
            predictions[i, 0] = self.predictions(image)[0]
        return predictions

    def predictions(self, image):
        """
        Use Google Cloud vision API to annotate an image.

        :param image: a np.ndarray representing an RGB image.
        :return: a numpy array with shape (1,) to comply with the assumptions that a typical prediction (logits or probabilities) are 1D.
        """

        image_pb = self.convert_nparray_pb(image)
        # Performs label detection on the image file
        response = self._client.label_detection(image=image_pb, max_results=10)
        gcp_labels = response.label_annotations
        pred = np.zeros((1), dtype=object)
        pred[0] = gcp_labels
        return pred

    def convert_nparray_pb(self, image):
        """
        Convert a numpy.ndarray (representing rgb image) into a protobuf.

        :param image: a np.ndarray representing an RGB image.
        :return: the input image in protobuf.
        """

        # save image to disk and read as byte
        image_pil = Image.fromarray(image)
        file_name = os.path.join(self._temp_dir, 'temp.png')
        image_pil.save(file_name)
        with io.open(file_name, 'rb') as image_file:
            content = image_file.read()
        image_pb = types.Image(content=content)
        return image_pb

    def num_classes(self):
        return self._num_classes

