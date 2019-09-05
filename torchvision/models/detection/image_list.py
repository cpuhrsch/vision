# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from __future__ import division

import torch

import nestedtensor
torch = nestedtensor.nested.monkey_patch(torch)


class ImageList(object):
    """
    Structure that holds a list of images (of possibly
    varying sizes) as a single tensor.
    This works by padding the images to the same size,
    and storing in a field the original sizes of each image
    """

    def __init__(self, tensors, image_sizes, batched_images):
        """
        Arguments:
            tensors (tensor)
            image_sizes (list[tuple[int, int]])
        """
        self.tensors = torch.nested_tensor(tensors)
        if self.tensors.dim() - self.tensors.nested_dim() != 3:
            import pdb
            pdb.set_trace()
        self.batched_images = batched_images
        self.image_sizes = image_sizes

    def to(self, *args, **kwargs):
        cast_tensor = self.tensors.to(*args, **kwargs)
        return ImageList(cast_tensor, self.image_sizes, self.batched_images)
