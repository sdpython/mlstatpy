"""
@file
@brief shortcut to image
"""

from .detection_segment import detect_segments, plot_segments, compute_gradient, plot_gradient
from .detection_segment import convert_array2PIL, convert_PIL2array
from .geometrie import Point, Segment
from .queue_binom import tabule_queue_binom
from .random_image import random_noise_image, random_segment_image
