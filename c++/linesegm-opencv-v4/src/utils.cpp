/*
 * utils.cpp
 *
 *  Created on: Apr 22, 2016
 *      Author: saverio
 */


#include "opencv2/opencv.hpp"
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;


inline void print_help () {

	fprintf(stderr,
	            "Usage: bin/linesegm [FILES]... [OPTIONS]...\n"
	            "Line segmentation for handwritten documents.\n"
	            "\n"
	            "Options:\n"
	            "\t-s integer \t\tStep value (1 or 2).\n"
	            "             \t\t\tChange the step with which explore the map.\n"
	            "\t-mf integer   \t\tMultiplication factor (must be a positive integer).\n"
	            "             \t\t\tIncrease the multiplication factor to obtain a non-admissible heuristic.\n"
	            "\t--stats	\t\tCompute and show statistics about the line segmentation.\n"
	            "\t--help       \t\tShow this help information.\n"
	            "\n"
	            "Examples:\n"
	            "\tbin/linesegm image.jpg -s 2 -mf 5 --stats\n"
	            "\tbin/linesegm images/* -s 1 -mf 20 --stats\n"
			    "\tbin/linesegm data/saintgall/images/csg562-003.jpg --stats\n");

	    exit(0);

}


inline Mat distance_transform (Mat input) {

	Mat dmat = input.clone();
	for (int i = 0; i < input.cols; i++) {
		Mat column = input(Rect(i, 0, 1, input.rows));
		Mat dcol;
		distanceTransform(column, dcol, DIST_L2, 5);
		dcol.copyTo(dmat.col(i));
	}

	return dmat;
}

template<typename Node>
inline void draw_path (Mat& graph, vector<Node>& path) {

	for (auto node : path) {
		int row, col;
		tie (row, col) = node;
		graph.at<uchar>(row, col) = (uchar) 0;
		if (col < graph.cols) {
			graph.at<uchar>(row, col + 1) = (uchar) 0;
		}
	}
	imwrite("data/map.jpg", graph*255);
}

template<typename Node>
inline void segment_above_boundary (Mat& input, vector<Node> boundary) {
	int row, col;
	for (auto node : boundary) {
		tie (row, col) = node;
		for (int i = row; i < input.rows; i++) {
			input.at<uchar>(i, col) = (uchar) 255;
			if (col < input.cols) {
				input.at<uchar>(i, col + 1) = (uchar) 255;
			}
		}
	}
}

template<typename Node>
inline void segment_below_boundary (Mat& input, vector<Node> boundary) {
	int row, col;
	for (auto node : boundary) {
		tie (row, col) = node;
		for (int i = row; i >= 0; i--) {
			input.at<uchar>(i, col) = (uchar) 255;
			if (col < input.cols) {
				input.at<uchar>(i, col + 1) = (uchar) 255;
			}
		}
	}
}

template<typename Node>
inline int lowest_boundary_pos(vector<Node> boundary) {

	int lowest_pos, row, col;
	for (unsigned i = 0; i < boundary.size(); i++) {
		tie(row, col) = boundary[i];
		if (i == 0) {
			lowest_pos = row;
		}
		else if (row > lowest_pos) {
			lowest_pos = row;
		}
	}
	return lowest_pos;
}

template<typename Node>
inline int highest_boundary_pos(vector<Node> boundary) {

	int highest_pos, row, col;
	for (unsigned i = 0; i < boundary.size(); i++) {
		tie(row, col) = boundary[i];
		if (i == 0) {
			highest_pos = row;
		}
		else if (row < highest_pos) {
			highest_pos = row;
		}
	}
	return highest_pos;
}

inline int highest_pixel_row(Mat& input) {
	uchar pixel_val;
	for (int i = 0; i < input.rows; i++) {
		for (int j = 0; j < input.cols; j++) {
			pixel_val = input.at<uchar>(i, j);
			if (pixel_val == (uchar) 0) {
				return i;
			}
		}
	}
	return input.rows;
}

inline int lowest_pixel_row(Mat& input) {
	uchar pixel_val;
	for (int i = input.rows-1; i >= 0; i--) {
		for (int j = input.cols-1; j >= 0; j--) {
			pixel_val = input.at<uchar>(i, j);
			if (pixel_val == (uchar) 0) {
				return i;
			}
		}
	}
	return 0;
}

inline Mat extract_bounding_box(Mat& input, int x, int y, int width, int height) {
	Mat ROI(input, Rect(x, y, width, height));
	Mat output;
	ROI.copyTo(output);
	return output;
}

inline void ensure_directory_exists(string dir_path) {
	struct stat st = {0};

	if (stat(dir_path.c_str(), &st) == -1) {
			    mkdir(dir_path.c_str(), 0755);
			    cout << "\n- Created folder ";
			    cout << dir_path << endl;
			}	
}

template<typename Node>
inline void segment_text_line (Mat& input, string out_dir, int line_id, vector<Node> lower, vector<Node> upper) {
	Mat output = input.clone();

	int highest_pos = highest_boundary_pos(upper);
	int lowest_pos = lowest_boundary_pos(lower);

	segment_above_boundary(output, lower);
	segment_below_boundary(output, upper);

	output = extract_bounding_box(output, 0, highest_pos, input.cols, lowest_pos-highest_pos);
	imwrite(out_dir + "line_" + to_string(line_id) + ".jpg", output*255);
}

template<typename Node>
inline void segment_text_line (Mat& input, string out_dir, int line_id, bool boundary_is_lower, vector<Node> boundary) {
	Mat output = input.clone();

	if (boundary_is_lower) {
		int lowest_pos = lowest_boundary_pos(boundary);
		int upper_bound = highest_pixel_row(output);
		segment_above_boundary(output, boundary);
		output = extract_bounding_box(output, 0, upper_bound, input.cols, lowest_pos-upper_bound);
	} else {
		int highest_pos = highest_boundary_pos(boundary);
		int lower_bound = lowest_pixel_row(output);
		segment_below_boundary(output, boundary);
		output = extract_bounding_box(output, 0, highest_pos, input.cols, lower_bound-highest_pos);
	}

	imwrite(out_dir + "line_" + to_string(line_id) + ".jpg", output*255);
}

template<typename Node>
inline Mat segment_line (Mat& input, vector<Node> path){

	Mat output = input.clone();
	for (auto node: path) {
		int row, col;
		tie(row, col) = node;
		for (int i = row; i < input.rows; i++) {
			output.at<uchar>(i, col) = (uchar) 255;
			if (col < output.cols) {
				output.at<uchar>(i, col + 1) = (uchar) 255;
			}
		}
	}

	return output;
}

inline bool strreplace (string& str, string& rem, string& repl) {
	size_t start_pos = str.find(rem);
	if (start_pos == string::npos)
		return false;
	str.replace(start_pos, rem.length(), repl);
	return true;
}

inline string infer_dataset (string filename) {
	size_t mls = filename.find("mls");
	size_t sg = filename.find("saintgall");
	if (mls != string::npos) {
		return "mls";
	} else if (sg != string::npos) {
		return "saintgall";
	} else {
		return "NULL";
	}
}

inline vector<string> read_folder (const char* folder) {
	DIR *pdir = NULL;
	pdir = opendir (folder);
	struct dirent *pent = NULL;
	vector<string> files;

	if (pdir == NULL) {
		cout << "\nERROR! pdir could not be initialised correctly";
		exit (3);
	}

	while ((pent = readdir (pdir))) {
		if (pent == NULL) {
			cout << "\nERROR! pent could not be initialised correctly";
			exit (3);
		}
		if (!strcmp(pent->d_name, ".") == 0 and !strcmp(pent->d_name, "..") == 0) {
			files.push_back((string) pent->d_name);
		}
	}
	closedir (pdir);

	return files;
}

inline int count_occurences (Mat& input, int num) {
	int count = 0;
	for (int i = 0; i < input.rows; i++) {
		for (int j = 0; j < input.cols; j++) {
			if (input.at<uchar>(i, j) == (uchar) num) {
				count++;
			}
		}
	}

	return count;
}

inline vector<double> select_best_assignments (vector<double>& hitrate, vector<double>& line_detection_GT, vector<double>& line_detection_R,
									 vector<string> lines, string groudtruth) {

	auto max = max_element(hitrate.begin(), hitrate.end());
	int pos = distance(hitrate.begin(), max);

	double hit_rate = *max;
	double line_det_GT = line_detection_GT[pos];
	double line_det_R = line_detection_R[pos];

	vector<double> stats;
	stats.push_back(hit_rate);
	stats.push_back(line_det_GT);
	stats.push_back(line_det_R);

	cout << "\t## Groundtruth: " << groudtruth << " - Detected: " << lines[pos];
	cout << " - Hit rate: " << to_string(*max);
	cout << " - Line detection GT: " << to_string(line_det_GT);
	cout << " - Line detection R: " << to_string(line_det_R) << endl;

	return stats;

}

inline void compute_statistics (string filename) {

	string dataset = infer_dataset(filename);

	string rem1 = "data/" + dataset + "/images/";
	string rem2 = ".jpg";
	string repl = "";
	strreplace(filename, rem1, repl);
	strreplace(filename, rem2, repl);

	string folder_lines = "data/" + dataset + "/detected/" + filename + "/";
	string folder_groundtruth = "data/" + dataset + "/groundtruth/" + filename + "/";

	vector<string> lines = read_folder(folder_lines.c_str());
	vector<string> groundtruth = read_folder(folder_groundtruth.c_str());

	Mat line, ground, united, shared;

	int tot_correctly_detected = 0;
	double tot_hitrate, tot_line_detection_GT, tot_line_detection_R = 0;
	for (unsigned int i = 0; i < groundtruth.size(); i++) {

		vector<double> hitrate, line_detection_GT, line_detection_R;
		ground =  imread(folder_groundtruth + groundtruth[i], 0) / 255;

		for (unsigned int j = 0; j < lines.size(); j++){

			line = imread(folder_lines + lines[j], 0) / 255;

			bitwise_or(line, ground, shared);
			bitwise_and(line, ground, united);

			int black_pixels_line = countNonZero(line == 0);
			int black_pixels_ground = countNonZero(ground == 0);
			int black_pixels_shared = countNonZero(shared == 0);
			int black_pixels_united = countNonZero(united == 0);

			hitrate.push_back((((double) black_pixels_shared) / ((double) black_pixels_united)));
			line_detection_GT.push_back((((double) black_pixels_shared) / ((double) black_pixels_ground)));
			line_detection_R.push_back((((double) black_pixels_shared) / ((double) black_pixels_line)));
		}

		vector<double> stats = select_best_assignments(hitrate, line_detection_GT, line_detection_R, lines, groundtruth[i]);
		tot_hitrate = tot_hitrate + stats[0];
		tot_line_detection_GT = tot_line_detection_GT + stats[1];
		tot_line_detection_R = tot_line_detection_R + stats[2];

		if (stats[1] >= 0.9 && stats[2] >= 0.9) {
			tot_correctly_detected++;
		}

	}

	cout << "\n\t## Avg. stats ==> ";
	cout << " Hit rate: " << to_string(tot_hitrate / groundtruth.size());
	cout << " - Line detection GT: " << to_string(tot_line_detection_GT / groundtruth.size());
	cout << " - Line detection R: " << to_string(tot_line_detection_R / groundtruth.size());
	cout << " - Correctly detected: " << to_string(tot_correctly_detected) << "/" << to_string(groundtruth.size()) << endl;

	ofstream csvfile;
	csvfile.open("data/" + dataset + "/stats.csv", std::ios_base::app);
	csvfile << filename;
	csvfile << ",";
	csvfile << int(round((tot_hitrate / groundtruth.size()) * 100));
	csvfile << ",";
	csvfile << int(round((tot_line_detection_GT / groundtruth.size()) * 100));
	csvfile << ",";
	csvfile << int(round((tot_line_detection_R / groundtruth.size()) * 100));
	csvfile << ",";
	csvfile << tot_correctly_detected;
	csvfile << ",";
	csvfile << groundtruth.size();
	csvfile << "\n";

	csvfile.close();

}
