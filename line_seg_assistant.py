import os
import re
import shutil as su
import sys

from enum import Enum


CORRECT_EXIT_CODE: int = 0


class OpenCVVersion(Enum):
	VERSION_3 = 3,
	VERSION_4 = 4


def ensure_os_compatibility() -> None:
	if sys.platform in ('win32',):
		raise Exception('Sorry, your OS cannot fully run this assistant.')


def open_cv_version_from_input() -> OpenCVVersion:
	print('Which version of OpenCV do you currently use?')
	print('\t(a) Below V4')
	print('\t(b) V4')
	selection: str = input('(answer) ')
	while selection not in ('a', 'b'):
		selection = input('(answer) ')
	return OpenCVVersion.VERSION_3 if selection == 'a' else OpenCVVersion.VERSION_4


def ensure_directory_exists(path: str) -> None:
	if not os.path.isdir(path):
		try:
			os.mkdir(path)
		except FileExistsError:
			raise FileExistsError('The directory \'%s\' exists. Aborting.' % (path,))


def copy_binarised_scroll_to_data_directory(ver: OpenCVVersion, containing_dir: str, file: str) -> None:
	path: str = os.path.join('c++', 'linesegm%s' % ('-opencv-v4' if ver == OpenCVVersion.VERSION_4 else '',), 'data')
	file_as_dir: str = re.sub('(-binari)[sz](ed\\.jpg)', '', file).lower()
	path_to_scroll: str = os.path.join(path, file_as_dir)  # in the `data` directory, not the original directory
	ensure_directory_exists(path_to_scroll)
	su.copy(os.path.join(containing_dir, file), os.path.join(path_to_scroll, file_as_dir + os.extsep + 'jpg'))


def copy_binarised_scrolls_to_data_directory(ver: OpenCVVersion) -> None:
	path: str = os.path.join(os.pardir, 'resources', 'scrolls-jpg')
	for file in os.listdir(path):
		if re.search('(binari)[sz](ed)', file) is not None:
			copy_binarised_scroll_to_data_directory(ver, containing_dir=path, file=file)


def prepare_resources_directory(ver: OpenCVVersion) -> None:
	path: str = os.path.join('c++', 'linesegm%s' % ('-opencv-v4' if ver == OpenCVVersion.VERSION_4 else '',), 'data')
	ensure_directory_exists(path)
	copy_binarised_scrolls_to_data_directory(ver)


def ensure_project_is_built(ver: OpenCVVersion) -> None:
	path: str = os.path.join('c++', 'linesegm%s' % ('-opencv-v4' if ver == OpenCVVersion.VERSION_4 else '',))
	print('\n--- LINE SEGMENTER BUILD IN PROGRESS ---')
	ret: int = os.system('cd %s; ./makefile.sh; cd %s' % (path, os.path.join(os.pardir, os.pardir)))
	print('-- LINE SEGMENTER BUILD FINISHED %sSUCCESSFULLY ---' % ('UN' if ret > CORRECT_EXIT_CODE else '',))
	if ret > CORRECT_EXIT_CODE:
		raise Exception('Project could not be built successfully. Aborting.')


def remove_file_if_existent(file: str) -> None:
	if os.path.exists(file):
		try:
			os.remove(file)
		except FileNotFound:
			raise FileNotFound('Could not find to-be-deleted file \'%s\'. Aborting.' % (file,))


def segment_single_scroll(ver: OpenCVVersion, scroll_dir: str) -> None:
	# segment single scroll
	path: str = os.path.join('c++', 'linesegm%s' % ('-opencv-v4' if ver == OpenCVVersion.VERSION_4 else '',))
	file_path: str = os.path.join('data', scroll_dir, scroll_dir + os.extsep + 'jpg')
	ret: int = os.system('cd %s; ./bin/linesegm %s' % (path, file_path))  # Note: `cd` does not affect our position!
	if ret > CORRECT_EXIT_CODE:
		raise Exception('Could not segment scroll \'%s\'. Aborting.' % (scroll_dir,))
	# perform clean-up
	for file in ('bw' + os.extsep + 'jpg', 'map' + os.extsep + 'jpg'):
		remove_file_if_existent(os.path.join(path, 'data', file))
	for file in os.listdir(path=os.path.join(path, 'data')):
		if re.search('(line_)[0-9]+(%sjpg)' % (os.extsep,), file) is not None:
			try:
				su.move(os.path.join(path, 'data', file), os.path.join(path, 'data', scroll_dir))
			except su.Error:
				print('\tFile \'%s\' already exists. Skipping...' % (os.path.join(path, 'data', scroll_dir, file),))
				remove_file_if_existent(os.path.join(path, 'data', file))  # no longer needed


def segment_all_scrolls(ver: OpenCVVersion) -> None:
	path: str = os.path.join('c++', 'linesegm%s' % ('-opencv-v4' if ver == OpenCVVersion.VERSION_4 else '',), 'data')
	print('\n\n\tSit back, grab a drink. This will take around 1h, 20m.', end='\n' * 2)
	print('\n--- SEGMENTING ALL SCROLLS ---')
	print('Considering the following:')
	for index, dir in enumerate(os.listdir(path)):
		print('\t(%2d) \'%s\'' % (index + 1, dir,))
		if not os.path.isdir(os.path.join(path, dir)):
			print('\t\t(to be deleted)')
			remove_file_if_existent(os.path.join(path, dir))
	for dir in os.listdir(path):
		print('--- WORKING ON SCROLL \'%s\' ---' % (dir,))
		segment_single_scroll(ver, dir)
	print('--- FINISHED SEGMENTING ALL SCROLLS ---')


if __name__ == '__main__':
	ensure_os_compatibility()
	version: OpenCVVersion = open_cv_version_from_input()
	prepare_resources_directory(version)
	ensure_project_is_built(version)
	segment_all_scrolls(version)
	try:
		# `src` is the target (existed already), and `dst` is the newly-created symbolic link
		src: str = os.path.join('c++', 'linesegm%s' % ('-opencv-v4' if version == OpenCVVersion.VERSION_4 else '',), 'data')
		os.symlink(src, 'data-link', target_is_directory=True)
	except OSError:
		raise OSError('Could not create a symbolic link to the data. Aborting.')
