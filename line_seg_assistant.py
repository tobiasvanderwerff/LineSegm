import cv2 as cv
import numpy as np
import os
import re
import shutil as su
import sys

from enum import Enum
from typing import Optional, Tuple


SUCCESSFUL_EXIT_CODE: int = 0


class OpenCVVersion(Enum):
	"""
	The version of Open CV installed on the user's machine.
	"""
	VERSION_3 = 3
	VERSION_4 = 4


class LineSegmentationAssistant:
	"""
	A class that interfaces ('assists') with segmenting lines from the `LineSegm` submodule.
	"""

	BLACKLISTED_PLATFORMS: Tuple[str, ...] = ('win32',)
	CUSTOM_IMG_DIR: str = 'custom'  # pseudo scroll directory to segment own images in

	def __init__(self, relative_root: str) -> None:
		"""
		Constructs a line segmentation assistant.

		:param relative_root: The relative path to the `project` (project-level root) directory. E.g. `..`.
		"""
		if not self.user_has_right_os():
			raise Exception('[%s] Sorry, your OS cannot fully run this assistant.' % (self.__class__.__name__,))
		self.opencv_ver: OpenCVVersion = self.open_cv_version_from_input()
		self.root: str = relative_root
		self.data_path: str = os.path.join(
			self.root,
			'LineSegm',
			'c++',
			'linesegm%s' % ('' if self.opencv_ver == OpenCVVersion.VERSION_3 else '-opencv-v4',),
			'data')

	@staticmethod
	def user_has_right_os() -> bool:
		return sys.platform not in LineSegmentationAssistant.BLACKLISTED_PLATFORMS

	def open_cv_version_from_input(self) -> OpenCVVersion:
		selection: str = input(
			'[%s] What is your OpenCV version? (\'a\': Below V3, \'b\': V3, \'c\': V4) ' %
			(self.__class__.__name__,))
		while selection not in ('a', 'b', 'c'):
			selection: str = input('What is your OpenCV version? (\'a\': Below V3, \'b\': V3, \'c\': V4) ')
		if selection == 'a':
			raise ValueError(
				'[%s] This version is not supported, sorry.' %
				(self.__class__.__name__,))
		return OpenCVVersion.VERSION_3 if selection == 'a' else OpenCVVersion.VERSION_4

	def ensure_directory_exists(self, path: str) -> None:
		if not os.path.isdir(path):
			try:
				os.mkdir(path)
			except FileExistsError:
				raise FileExistsError('[%s] The directory \'%s\' exists. Aborting.' % (self.__class__.__name__, path,))

	def remove_file_if_existent(self, file: str) -> None:
		if os.path.exists(file):
			try:
				os.remove(file)
			except FileNotFoundError:
				raise FileNotFoundError(
					'[%s] Could not find to-be-deleted file \'%s\'. Aborting.' %
					(self.__class__.__name__, file,))

	def ensure_symbolic_link_exists(self) -> None:
		"""
		Ensures there exists a symbolic link from `LineSegm` to the right `data` directory.
		"""
		try:
			os.symlink(self.data_path, os.path.join(self.root, 'LineSegm', 'data-link'), target_is_directory=True)
		except OSError:
			raise OSError('[%s] Could not create a symbolic link to the data!' % (self.__class__.__name__,))

	def copy_binarised_scroll_to_data_directory(self, containing_dir: str, file: str) -> None:
		"""
		Copies a single binarised JPG scroll to the `data` directory.

		:param containing_dir: The directory in which the target file is contained. E.g. `resources/scrolls-jpg`.
		:param file: The name of the binarised scroll to copy. Should include the file extension, `.jpg`.
		"""
		file_as_dir: str = re.sub('(-binari)[sz](ed\\.jpg)', '', file).lower()
		path_to_scroll: str = os.path.join(self.data_path, file_as_dir)
		self.ensure_directory_exists(path_to_scroll)
		su.copy(os.path.join(containing_dir, file), os.path.join(path_to_scroll, file_as_dir + os.extsep + 'jpg'))

	def copy_binarised_scrolls_to_data_directory(self) -> None:
		"""
		Copies all binarised JPG scrolls present in `resources/scrolls-jpg` to the `data` directory.
		"""
		dir_containing_file: str = os.path.join(self.root, 'resources', 'scrolls-jpg')
		for file in os.listdir(dir_containing_file):
			# go over the scrolls and select only the binarised ones
			if re.search('(binari)[sz](ed)', file) is not None:
				self.copy_binarised_scroll_to_data_directory(dir_containing_file, file=file)

	def prepare_resources_directory(self) -> None:
		"""
		Copies all binarised JPG scrolls from `resources/scrolls-jpg` to a newly-created `data` directory.
		"""
		self.ensure_directory_exists(self.data_path)
		self.copy_binarised_scrolls_to_data_directory()

	def ensure_project_is_built(self, inform: bool = False) -> None:
		"""
		Checks whether the main C++ line segmentation project has been built, and if not, builds it.

		:param inform: Whether to inform of non-error progress. Defaults to `False`.
		"""
		path: str = os.path.join(self.data_path, os.pardir)  # get to the main line segmentation directory
		if inform:
			print('\n--- LINE SEGMENTER BUILD IN PROGRESS ---')
		ret: int = os.system(
			'cd %s; ./makefile.sh%s; cd %s' %
			(
				path,
				'' if not inform else ' 1> /dev/null',
				os.path.join(os.pardir, os.pardir)  # TODO: Can possibly omit.
			))
		if inform:
			print('-- LINE SEGMENTER BUILD FINISHED %sSUCCESSFULLY ---' % ('UN' if ret > SUCCESSFUL_EXIT_CODE else '',))
		if ret > SUCCESSFUL_EXIT_CODE:
			raise RuntimeError('Project could not be built successfully. Aborting.')

	def clean_up_data_dir(self, scroll_dir: Optional[str]) -> None:
		"""
		Ensures the `data` directory only contains scroll directories and e.g. no left-over `map` and `bw` images.

		Note that the `scroll_dir` should consist only of the name of the directory, and should not be a full or
		relative path. An example of a correct `scroll_dir` is `p21-fg006-r-c01-r01`.

		:param scroll_dir: Optional. If supplied, will move the `line_*.jpg` entries to `scroll_dir`.
		"""
		for file in ('bw' + os.extsep + 'jpg', 'map' + os.extsep + 'jpg'):
			self.remove_file_if_existent(os.path.join(self.data_path, file))
		if scroll_dir is None:
			return  # we are done if no `scroll_dir` moves need to be performed
		for file in os.listdir(self.data_path):
			if re.search('(line_)[0-9]+(%sjpg)' % (os.extsep,), file) is not None:
				try:
					su.move(os.path.join(self.data_path, file), os.path.join(self.data_path, scroll_dir))
				except su.Error:
					print(
						'\tFile \'%s\' already exists. Skipping...' %
						(os.path.join(self.data_path, scroll_dir, file),))
					self.remove_file_if_existent(os.path.join(self.data_path, file))  # no longer needed

	def segment_single_scroll(self, scroll_dir: str, inform: bool = False) -> None:
		"""
		Sets in motion the `linesegm` executable to segment a single scroll.

		:param scroll_dir: The scroll directory to segment, e.g. `p21-fg006-r-c01-r01`.
		:param inform: Whether to inform of non-error progress. Defaults to `False`.
		"""
		path: str = os.path.join(self.data_path, os.pardir)  # main directory of line segmentation project
		file_path: str = os.path.join('data', scroll_dir, scroll_dir + os.extsep + 'jpg')
		ret: int = os.system(
			'cd %s; ./bin/linesegm %s%s' %
			(
				path,
				file_path,
				' 1> /dev/null' if not inform else ''
			))  # note that `cd` doesn't affect position
		if ret > SUCCESSFUL_EXIT_CODE:
			raise Exception('Could not segment scroll \'%s\'. Aborting.' % (scroll_dir,))

	def segment_all_scrolls(self, inform: bool = True) -> None:
		"""
		Instructs the `linesegm` executable to segment all scrolls.

		:param inform: Whether to inform of non-error progress. Defaults to `False`.
		"""
		if inform:
			print('\n\n\tSit back, grab a drink. This will take around 1h, 20m.', end='\n' * 2)
			print('\n--- SEGMENTING ALL SCROLLS ---')
			print('Considering the following:')
		for index, drc in enumerate(os.listdir(self.data_path)):
			if inform:
				print('\t(%2d) \'%s\'' % (index + 1, drc,))
			if not os.path.isdir(os.path.join(self.data_path, drc)):
				if inform:
					print('\t\t(to be deleted)')
				self.remove_file_if_existent(os.path.join(self.data_path, drc))
		for index, drc in enumerate(os.listdir(self.data_path)):
			if inform:
				print('--- WORKING ON SCROLL \'%s\' (%3d/%3d) ---' % (drc, index + 1, len(os.listdir(self.data_path))))
			self.segment_single_scroll(drc, inform)
			self.clean_up_data_dir(drc)
		if inform:
			print('--- FINISHED SEGMENTING ALL SCROLLS ---')

	def segmented_custom_image(self, img: np.ndarray, inform: bool = False) -> Tuple[np.ndarray, ...]:
		"""
		Segments a self-specified image using the `linesegm` executable.

		This method affects the `data` directory only marginally: it uses it during the segmentation and leaves
		a pseudo scroll directory named `custom`, but nothing else is done.

		:param img: The image to segment into lines.
		:param inform: Whether to inform of non-error progress. Defaults to `False`.
		:returns: An ordered set of images. Each image is one line. The first image is the first line.
		"""
		path: str = os.path.join(self.data_path, LineSegmentationAssistant.CUSTOM_IMG_DIR)
		self.ensure_directory_exists(path)
		cv.imwrite(os.path.join(path, 'custom.jpg'), img)
		if inform:
			print('--- WORKING ON CUSTOM SCROLL \'%s\' ---' % (LineSegmentationAssistant.CUSTOM_IMG_DIR,))
		self.segment_single_scroll(LineSegmentationAssistant.CUSTOM_IMG_DIR)
		self.clean_up_data_dir(scroll_dir=LineSegmentationAssistant.CUSTOM_IMG_DIR)
		out: Tuple[np.ndarray, ...] = tuple()
		for idx in range(1, len(os.listdir(path))):
			entry: str = 'line_%d.jpg' % (idx,)
			if re.search('(line_)[0-9]+(\\.jpg)', entry) is not None:
				out += cv.imread(os.path.join(path, entry), 0),  # read as grayscale image
		os.system(
			'rm -rf %s%s' %
			(os.path.join(path, '*'), ' 1> /dev/null' if not inform else ''))
		os.rmdir(path)
		return out


if __name__ == '__main__':
	assistant = LineSegmentationAssistant(relative_root=os.pardir)  # will be different for other scripts, of course
	assistant.prepare_resources_directory()
	assistant.ensure_project_is_built(inform=False)
	assistant.segment_all_scrolls(inform=True)
	assistant.ensure_symbolic_link_exists()
