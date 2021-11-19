from math import ceil

import cv2
import numpy as np
import tensorflow as tf

from .estimators import JansenEstimator
from .sampling import TFSobolSequence
from .tf_perturbations import inpainting
from .utils import resize


class SobolAttributionMethod:
    """
    Sobol' Attribution Method.

    Once the explainer is initialized, you can call it with an array of inputs and labels (int) 
    to get the STi.

    Parameters
    ----------
    grid_size: int, optional
        Cut the image in a grid of grid_size*grid_size to estimate an indice per cell.
    nb_design: int, optional
        Must be a power of two. Number of design, the number of forward will be nb_design(grid_size**2+2).
    sampler : Sampler, optional
        Sampler used to generate the (quasi-)monte carlo samples.
    estimator: Estimator, optional
        Estimator used to compute the total order sobol' indices.
    perturbation_function: function, optional
        Function to call to apply the perturbation on the input.
    batch_size: int, optional,
        Batch size to use for the forwards.
    """

    def __init__(
        self,
        model,
        grid_size=8,
        nb_design=64,
        sampler=TFSobolSequence(),
        estimator=JansenEstimator(),
        perturbation_function=inpainting,
        batch_size=256
    ):

        assert (nb_design & (nb_design-1) == 0) and nb_design != 0,\
            "The number of design must be a power of two."

        self.model = model

        self.grid_size = grid_size
        self.nb_design = nb_design
        self.perturbation_function = perturbation_function

        self.sampler = sampler
        self.estimator = estimator

        self.batch_size = batch_size

        self.masks = sampler(grid_size**2, nb_design).reshape((-1, grid_size, grid_size, 1))

    def __call__(self, inputs, labels):
        """
        Explain a particular prediction

        Parameters
        ----------
        inputs: ndarray or tf.Tensor [Nb_samples, Width, Height, Channels]
            Images to explain.
        labels: list of int,
            Label of the class to explain.
        """
        input_shape = inputs.shape[1:-1]
        explanations = []

        for input, label in zip(inputs, labels):

            perturbator = self.perturbation_function(input)

            y = np.zeros((len(self.masks)))
            nb_batch = ceil(len(self.masks) / self.batch_size)

            for batch_index in range(nb_batch):
                # retrieve masks of the current batch
                start_index = batch_index * self.batch_size
                end_index = min(len(self.masks), (batch_index+1)*self.batch_size)
                batch_masks = self.masks[start_index:end_index]

                # apply perturbation to the input and forward
                batch_y = SobolAttributionMethod._batch_forward(self.model, input, batch_masks,
                                                                perturbator, input_shape)

                # store the results
                batch_y = batch_y[:, label].numpy()
                y[start_index:end_index] = batch_y

            # get the total sobol indices
            sti = self.estimator(self.masks, y, self.nb_design)
            sti = resize(sti, input_shape)

            explanations.append(sti)

        return np.array(explanations)

    @staticmethod
    @tf.function
    def _batch_forward(model, input, masks, perturbator, input_shape):
        upsampled_masks = tf.image.resize(masks, input_shape)
        perturbated_inputs = perturbator(upsampled_masks)
        outputs = model(perturbated_inputs)
        return outputs
