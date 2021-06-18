import copy as cp
import csv
import cv2 as cv
import numpy as np
import os
import re

import matplotlib.pyplot as plt

from enum import Enum
from typing import Dict, Tuple, Union



class ComputeMode(Enum):
	"""
	A way of processing the image and its histogram before calculating the persistance peaks.
	"""
	RAW = 'raw'  # supply the image as-is to the histogram method
	MAX_NORMALISED = 'max-normalised'  # normalise the image by dividing by the maximum value of the histogram
	CROPPED_RAW = 'cropped-raw'  # crop the image before computing the histogram
	CROPPED_MAX_NORMALISED = 'cropped-max-normalised'  # first crop, then max-normalise


class Statistic(Enum):
	"""
	A statistic which may be useful for analysing the persistence peak output data.
	"""
	MAE = 'mean-absolute-error'
	MSE = 'mean-squared-error'
	ME = 'average-error'
	SE = 'squared-error'
	E = 'error'


RELATIVE_SCROLL_DIR_PATH: str = os.path.join(os.pardir, os.pardir, 'resources', 'scrolls-jpg')
EXPECTED_LINE_CSV_FILE: str = 'lines.csv'  # file storing manual counts of number of lines per scroll
CROP_THRESHOLD: float = 9.92e-1  # minimal percentage full-intensity (white) pixels to keep cropping

# Maps from modes to `c`s, which in turn map to error-and-squared-error tuples.
AccuraciesDict = Dict[ComputeMode, Dict[float, Tuple[float, float]]]


# Maps from modes to `c`s, which in turn map to statistics, that in their turn map to the statistical values.
StatisticsDict = Dict[ComputeMode, Dict[float, Dict[Statistic, float]]]


def all_scrolls() -> Tuple[str, ...]:
	"""
	Yields the names of all scrolls present in the relevant subfolder of the `resources` directory.

	:returns: An ordered set of names.
	"""
	scrolls: Tuple[str, ...] = tuple()
	for scroll in os.listdir(RELATIVE_SCROLL_DIR_PATH):
		if re.search('(-binari)[sz](ed)', scroll):
			scrolls += scroll,
	return scrolls


def expected_number_of_lines(file_name: str) -> int:
	"""
	Obtains the number of lines we would expect to be detected by the persistance peak algorithm.

	:param file_name: The name of the scroll to obtain the expected number of lines from.
	:returns: The expectation.
	"""
	act_file_name: str = re.sub('(-binari)[sz](ed.jpg)', '', file_name)
	act_file_name = act_file_name.lower()
	with open('lines.csv') as handle:
		rd = csv.reader(handle)
		row: Tuple[str, str]  # file name, string version of a count
		for row in rd:
			if re.search('(%s)' % (act_file_name,), row[0]):
				return int(row[1])  # convert string version to integer
	raise ValueError('[expected_number_of_lines] Could not find number of rows for \'%s\'!' % (file_name,))


def cropped_image(img: np.ndarray, percent_white_thr: float = CROP_THRESHOLD) -> np.ndarray:
	"""
	Yields the image but with excessive white space in the margins (both horizontal and vertical) removed.

	The `percent_white_thr` is multiplied by the maximal pixel intensity (white, or 255) and compared to the
	average pixel intensity of a column (if cropping horizontally) or a row (vertically). If the column or row lies
	strictly below the percentage, we regard the column or row as being part of text, and thus do not crop further.

	:param img: The image to crop.
	:param percent_white_thr: A percentage below which we decide that we should not crop further.
	:returns: The cropped image.
	"""
	min_x: int = 0
	max_x: int = img.shape[1]  # itself exclusive
	min_y: int = 0
	max_y: int = img.shape[0]
	while min_x < img.shape[1] and (img[:, min_x].sum() / img.shape[0]) >= (255 * percent_white_thr):
		min_x += 1
	while max_x > 0 and (img[:, max_x - 1].sum() / img.shape[0]) >= (255 * percent_white_thr):
		max_x -= 1
	while min_y < img.shape[0] and (img[min_y, :].sum() / img.shape[1]) >= (255 * percent_white_thr):
		min_y += 1
	while max_y > 0 and (img[max_y - 1, :].sum() / img.shape[1]) >= (255 * percent_white_thr):
		max_y -= 1
	return img[min_y:max_y, min_x:max_x]


def histogram(img: np.ndarray, normalise: bool) -> np.ndarray:
	"""
	Computes the vertical projection histogram of `img`.

	:param img: The image to aggregate (sum) per row the entries of to yield the histogram.
	:param normalise: Whether to divide by the maximum intensity row of the histogram as a final step.
	:returns: The vertical projection histogram of the image.
	"""
	hist: np.ndarray = img.sum(axis=1)
	hist = (255 * img.shape[1]) - hist  # zero now signifies a fully empty row, and vice-versa
	if normalise:
		hist = hist / hist.max()
	return hist


def thresholded(hist: np.ndarray, thr: float) -> np.ndarray:
	"""
	Yields a vector which stores for every histogram position whether it is above the threshold.

	:param thr: The threshold to apply to `hist` to obtain the output.
	:returns: A vector storing `1` for values that were on or above `thr`, and `0` otherwise. Is of type `uint8`.
	"""
	out_hist = cp.deepcopy(hist)
	out_hist[np.where(hist >= thr)] = 1e0
	out_hist[np.where(hist < thr)] = 0e0
	return out_hist.astype('uint8')


def count_peaks(hist: np.ndarray) -> int:
	"""
	Counts the number of peaks in the supplied persistance histogram.

	:param hist: The persistence histogram to count peaks of.
	:returns: The number of peaks in `hist`.
	"""
	is_peak: bool = False
	ctr: int = 0
	for index in range(hist.shape[0]):
		if not is_peak and hist[index] == 1:
			# we entered a peak
			is_peak = True
			ctr += 1
		elif is_peak and hist[index] == 0:
			# we left a peak
			is_peak = False
	return ctr


def statistics_from_accuracies(acc: AccuraciesDict, num_scrolls: int) -> StatisticsDict:
	"""
	Given a mapping from `ComputeMode`s to accuracy metrics, yields a dictionary of statistics per mode.

	Included in the dictionary are all statistics listed in the `Statistic` enumeration, above.

	:param acc: The accuracy mapping to base the statistics dictionary on.
	:param num_scrolls: The number of scrolls that were available during build-up of `accuracies`.
	:returns: The dictionary.
	"""
	d: Dict[ComputeMode, Dict[float, Dict[str, Union[int, float]]]] = {}
	for mode in ComputeMode:
		d[mode] = {}
		for c in acc[mode]:
			d[mode][c] = {}
			for stat in Statistic:
				if stat == Statistic.MAE:
					d[mode][c][stat] = float(acc[mode][c][1] / num_scrolls)
				elif stat == Statistic.MSE:
					d[mode][c][stat] = float(acc[mode][c][2] / num_scrolls)
				elif stat == Statistic.ME:
					d[mode][c][stat] = float(acc[mode][c][0] / num_scrolls)
				elif stat == Statistic.SE:
					d[mode][c][stat] = acc[mode][c][1]
				elif stat == Statistic.E:
					d[mode][c][stat] = acc[mode][c][0]
				else:
					raise NotImplementedError(
						'[%s] Statistic \'%s\' not implemented!' %
						('statistics_from_accuracies', stat.value))
	return d


if __name__ == '__main__':
	# Set preferences here.
	#   - `values_considered` are the `c` scalars to test with to see how good peaks match with expected number of lines
	#   - `show_cropped` lets you see how cropped images look like
	values_considered: Tuple[float, ...] = tuple(np.arange(-0.8, (0.0 + 0.1), 0.02))  # Range: [-0.8, 0.08].
	show_cropped_images: bool = False

	# The best value scalar `c` for `MAX_NORMALISED` is `-0.66`, while for `CROPPED_MAX_NORMALISED` it is `-0.24`.
	# This applies when using `Statistic.MAE`.

	# Initialise the `accuracies` mapping, which will store our statistics.
	accuracies: AccuraciesDict = {}
	for mode in ComputeMode:
		accuracies[mode] = {}
		for c in values_considered:
			accuracies[mode][c] = (0.0, 0.0, 0.0)

	# Main loop. Determine absolute error between expected number of lines versus actual number from peaks.
	for scroll in all_scrolls():
		# print('Considering scroll \'%s\'.' % (scroll,), end=' ')
		# print('(expected: %d lines).' % (expected_number_of_lines(scroll),))
		img = cv.imread(os.path.join(RELATIVE_SCROLL_DIR_PATH, scroll), 0)  # read as black-and-white (grayscale)
		for mode in ComputeMode:
			# print('\tConsidering mode \'%s.\'' % (mode.value,))
			act_img: np.ndarray = cp.deepcopy(img)  # as we do not want to manipulate `img` directly

			# obtain the histogram
			if mode in (ComputeMode.CROPPED_RAW, ComputeMode.CROPPED_MAX_NORMALISED):
				act_img = cropped_image(act_img)
				if mode in (ComputeMode.CROPPED_RAW,) and show_cropped_images:
					fig, ax = plt.subplots(1, 1)
					ax.imshow(act_img, 'gist_gray')
					ax.set_title('Scroll \'%s\'' % (scroll,))
					plt.show()
			h = histogram(act_img, normalise=(mode in (ComputeMode.MAX_NORMALISED, ComputeMode.CROPPED_MAX_NORMALISED)))

			# determine how many peaks we'll get
			mu: float = h.mean()
			sd: float = h.std()
			for c in values_considered:
				t = thresholded(h, mu - (c * sd))  # See Surinta et al. (2014), p. 177. Normally, `c = 1.0`.
				peak_num: int = count_peaks(t)
				error: float = float(expected_number_of_lines(scroll) - peak_num)
				accuracies[mode][c] = \
					(
						accuracies[mode][c][0] + error,
						accuracies[mode][c][1] + float(np.abs(error)),
						accuracies[mode][c][2] + (error ** 2)
					)
				# print('\t\t(c=%s%.2lf) Got %2d peaks.' % (' ' if c >= 0.0 else '', c, peak_num))

	# Show statistics per `ComputeMode`.
	stats = statistics_from_accuracies(accuracies, len(all_scrolls()))
	for mode in ComputeMode:
		print('Mode: \'%s\'' % (mode.value,))
		for c in values_considered:
			print('\t(c=%s%.2lf) ' % (' ' if c >= 0.0 else '', c,), end=' ')
			print('%7.3lf' % (stats[mode][c][Statistic.MAE],))
