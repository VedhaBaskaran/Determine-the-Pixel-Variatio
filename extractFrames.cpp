#ifdef USE_PYTHON
#include <Python.h> // Must be included first
#endif
#include <opencv2/opencv.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/videoio/legacy/constants_c.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/io.h>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <assert.h>
#include <iostream>
#include <cstring>
#include <random>
#include <queue>
#include <fstream>
#include <filesystem>
#include <limits.h>
#include <random>

using namespace std;
using namespace cv;
namespace fs = filesystem;

#define DISPLAY(stream) if (verbose){stream;}
#define BET_FRAMES_DIRNAME "in_between"
#define SAME_FRAME_THRESHOLD 15

/*======== SUBFUNCTIONS IMPLEMENTATION ==========*/

inline double my_random(void){
	static std::random_device rd;
	return ((double) rd() - rd.min()) / (double) (rd.max() - rd.min());
}

/* String and directory tools */
bool is_supported_videofile(const fs::path path){
	if (!path.has_extension()){
		return false;
	}
	string ext = path.extension();
	//TODO: Find OpenCV supported file extension and add it here with a switch,
	//Then remove the commentary into 'get_video_files'
	return true;
}

queue<string> *get_video_files(const char *in_path, double file_proportion, bool verbose){
	queue<string> *vid_files = new queue<string>();
	fs::directory_entry f = fs::directory_entry(in_path);

	// File
	if (f.is_regular_file() && is_supported_videofile(f.path())){
		vid_files->push((string) f.path());
		DISPLAY(cout << "Added: " << f.path() << '\n');
		return vid_files;
	}

	// Directory
	if (f.is_directory()){
		double r;
		for(auto& p: fs::recursive_directory_iterator(in_path)){
			if (p.is_regular_file()){// && is_supported_videofile(p.path())){
				r = my_random();
				if (r < file_proportion){
					vid_files->push((string) p.path());
					DISPLAY(cout << "Added: " << p.path() << '\n');
				}
			}
		}
		//shuffle(vid_files); //TODO: Find a way to shuffle the queue
		return vid_files;
	}

	// Error case
	cerr << "Fatal Error: " << in_path << " is not a file nor a directory." << endl;
	return NULL;
}

string get_filename(const string &filepath){
	return fs::path(filepath).filename().replace_extension("");
}

void str_normalize(string &s){
	for (size_t i = 0; i < s.length(); i++){
		switch (s[i]){
			case ' ': s[i] = '_'; break;
			case '(': s[i] = '_'; break;
			case ')': s[i] = '_'; break;
			case '[': s[i] = '_'; break;
			case ']': s[i] = '_'; break;
		}
	}
}

int build_output_dir(const char *path){
	struct stat info;
	if(stat(path, &info) != 0){
		if (mkdir(path, S_IRWXU) != 0){
			cerr << "Fatal Error: Couldn't create " << path << endl;
			return EXIT_FAILURE;
		}
	}
	else if (!S_ISDIR(info.st_mode)){ 
		cerr << "Fatal Error: " << path << " is not a directory" << endl;
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}

/* Frame Management tools */
double difference(const Mat &prev, const Mat &next, const unsigned int area_side);

void write_frame(const Mat &m, string dir, string name, bool verbose){
	string tmp = dir+'/'+name;
	bool a = imwrite(tmp, m);
	if (a){
		DISPLAY(cout << "Wrote frame " << tmp << endl);
	}
	else{
		DISPLAY(cout << "Failed to write frame " << tmp << endl);
	}
}

bool are_identic_frames(const Mat &m1, const Mat &m2){
	assert(m1.dims       == m2.dims 
		&& m1.size()     == m2.size() 
		&& m1.channels() == m2.channels());
	unsigned char *m1_data = (unsigned char*)(m1.data);
	unsigned char *m2_data = (unsigned char*)(m2.data);
	for (int i = 0; i < m1.rows; i++){
		for (int j = 0; j < m1.cols; j++){
			for (int k = 0; k < m1.channels(); k++){
				if (abs((int) m1_data[m1.step*i + j*m1.channels() + k]
				 - m2_data[m2.step*i + j*m2.channels() + k]) > SAME_FRAME_THRESHOLD){
					return false;
				}
			}
		}
	}
	return true;
}

void get_next_frame(VideoCapture &video, Mat &prev, Mat &curr){
	prev.release();
	curr.copyTo(prev);
	curr.release();
	video >> curr;
}

void video_skip_frames(VideoCapture &video, const unsigned int nb_frames){
	bool success = true;
	for (unsigned int i = 1; success && i < nb_frames; i++){
		success = video.grab();
	}
}

void video_skip_frames_stock(VideoCapture &video, const unsigned int nb_frames, queue<Mat*> &in_btw_frm_stocked, bool remove_identic_frames){
	if (!remove_identic_frames){
		Mat *tmp_mat;
		for (unsigned int i = 1; i < nb_frames; i++){
			tmp_mat = new Mat();
			video >> (*tmp_mat);
			if (tmp_mat->empty()){ break;}
			in_btw_frm_stocked.push(tmp_mat);
		}
	} else {
		Mat *tmp_mat;
		Mat *tmp_mat_prev = new Mat();
		for (unsigned int i = 1; i < nb_frames; i++){
			tmp_mat = new Mat(); // New matrix to be stocked into queue
			video >> (*tmp_mat);
			if (tmp_mat->empty()){ break;}
			if (!tmp_mat_prev->empty() && !are_identic_frames(*tmp_mat_prev, *tmp_mat)){
				in_btw_frm_stocked.push(tmp_mat);
				tmp_mat_prev->release();
				*tmp_mat_prev = tmp_mat->clone();
			} else {
				tmp_mat_prev->release();
				*tmp_mat_prev = tmp_mat->clone();
				delete tmp_mat;
			}
		}
		delete tmp_mat_prev;
	}
}

void empty_frame_stock(queue<Mat*> &in_btw_frm_stocked){
	while(!in_btw_frm_stocked.empty()){
		delete in_btw_frm_stocked.front();
		in_btw_frm_stocked.pop();
	}
}

/**
 * @brief Write frames from a video file into the output directory with various adjustable parameters.\
 * Also allow the possibility to save the in-between frames. 
 * 
 * @param in_path The path to the video file or directory containing video files.
 * @param out_dir The path to the output directory. If it doesn't exist, a new directory will be created.
 * @param skip_frames The interval between each pair of frames that have to be saved/compared.
 * @param skip_seconds Same as skip_frames, but the unit is in seconds instead. You cannot skip by frames AND by seconds.
 * @param start_at_frame The starting frame index.
 * @param stop_at_frame The stoping frame index.
 * @param display_interval The interval for information display. This is not by frame, but by loop. Each 'skip_frame' frames count for 1 loop. 
 * @param min_mean_counted The number of difference that have to be computed in order to have a meaningful mean.
 * @param save_in_between_frames Whether or not in-between frames have to be saved.
 * @param stock_in_between_frames Whether or not in-between frames have to be stocked in memory or saved later while relooping the video.
 * @param remove_identic_frames Whether or not identic frames has to be removed or not.
 * @param compute_difference Whether or not the difference between each pair of frames has to be calculated or not, in order to save only pictures that aren't very different.
 * @param verbose Whether or not informations has to be displayed on the screen.
 * @param diff_threshhold The threshold idicating when frames are considered to too different (values are around 0~2).
 * @param pic_save_proba The probability to save a pair of frames or not (unselected pair won't compute difference or user-specified test functions)
 * @param file_proportion The probability compute a found video or not (if too many videos are in the same directory)
 * @param timeout A timeout in seconds. Note the if 'stock_in_between_frames' is set to 0, the last video in-between frames might not be all saved. 0 means 'no timeout'.
 * @param first_frame_func A user-specified function to test a property on the first frame. If this property if not verified, the pair of frames won't be saved.
 * @param second_frame_func A user-specified function to test a property on the second frame. If this property if not verified, the pair of frames won't be saved.
 * @param compare_frame_func A user-specified function to test a property on the pair of frames. If this property if not verified, the pair of frames won't be saved.
 * @return int Success : 0, Failure : 1
 */
extern int extractFrames(const char *in_path, const char *out_dir,
			unsigned int skip_frames             = 1,
			double       skip_seconds            = 0,
			unsigned int start_at_frame          = 0,
			unsigned int stop_at_frame           = 0,
			unsigned int display_interval        = 1,
			bool         save_in_between_frames  = true,
			bool         stock_in_between_frames = true,
			bool         remove_identic_frames   = false,
			bool         compute_difference      = false,
			unsigned int min_mean_counted        = 20,
			double       diff_threshhold         = 0.20,
			bool         verbose                 = true,
			double       pic_save_proba          = 1,
			double       file_proportion         = 1,
			double       timeout                 = 0,     //In seconds
			bool (*first_frame_func)(const Mat &)  = NULL,
			bool (*second_frame_func)(const Mat &) = NULL,
			bool (*compare_frame_func)(const Mat &, const Mat &) = NULL){

	// Parameters verification
	assert(skip_frames != 0 || skip_seconds != 0);
	assert(0 <= skip_seconds);
	assert(0 <= timeout);
	assert(0 <= diff_threshhold);
	assert(0 <= pic_save_proba && pic_save_proba <= 1);
	assert(0 <= file_proportion && file_proportion <= 1);
	assert(start_at_frame < stop_at_frame || stop_at_frame == 0);

	// Retrieving video file paths into vid_path_queue
	queue<string> *vid_path_queue = get_video_files(in_path, file_proportion, verbose);
	unsigned int init_queue_size = vid_path_queue->size();

	// Building output directory if doesn't exists
	if (vid_path_queue == NULL || build_output_dir(out_dir)){
		return EXIT_FAILURE;
	}
	string str_out_dir = out_dir;

	// Global parameter initialisation
	unsigned int global_counter = 0;
	//srand(time(NULL)+getpid());
	double t0 = time(NULL);

	// Parameters initialisation
	Mat prev_frame, curr_frame;
	unsigned int curr_frame_idx, prev_frame_idx, loop_idx; // loop_idx is also used for as mean counter
	double diff, diff_coef, mean; // The mean of all computed differences
	float r;
	bool fff_verified, sff_verified, cff_verified, do_usr_func;
	queue<unsigned int> in_btw_frm_indexes;
	queue<Mat*> in_btw_frm_stocked;
	string frame_path;
	double width, height, fps, nb_frames, _timeout;
	unsigned int _skip_frames, _stop_at_frame;

	while (!vid_path_queue->empty()){
		// Retrieving video path
		string filepath = vid_path_queue->front();
		vid_path_queue->pop();
		if (my_random() < 0.8){ // In order to have kind of a "shuffle" effect on the queue
			vid_path_queue->push(filepath);
			continue;
		}
		string filename = get_filename(filepath);
		//str_normalize(filename);

		// Opening Video
		VideoCapture video(filepath); // Open the argv[1] video file
		if(!video.isOpened()){
			cerr << "Couldn't open " << filepath << endl;
			continue;
		}

		// Building prefix for images writing
		string curr_file_out_dir = str_out_dir+"/"+filename;
		fs::directory_entry dir_entry = fs::directory_entry(curr_file_out_dir);
		
		// Video settings
		width     = video.get(CV_CAP_PROP_FRAME_WIDTH);
		height    = video.get(CV_CAP_PROP_FRAME_HEIGHT);
		fps       = video.get(CV_CAP_PROP_FPS);
		nb_frames = video.get(CV_CAP_PROP_FRAME_COUNT);
		DISPLAY(cout << "\nLoaded " << filename << format("(%.2f %%)", (double) 100. - (100*(double)vid_path_queue->size()/init_queue_size))
			<< "\nVideo Properties: " << width << "x" << height
			<< ", " << fps << " fps, " << nb_frames << " frames." << endl);

		// Arguments interpretation
		_skip_frames   = skip_seconds != 0 ? max(1., skip_seconds*fps) : skip_frames;
		_stop_at_frame = stop_at_frame == 0 ? nb_frames + 1 : stop_at_frame;
		_timeout = timeout == 0 ? DBL_MAX : timeout; 
		if (_skip_frames == 1 && save_in_between_frames){
			DISPLAY(cout << "Error: Cannot save in-between frame wile skiping only 1 frame.\n\
			No in-between frame will be saved.")
			sleep(5);
		}

		// Placing the video to the start point
		video_skip_frames(video, start_at_frame);

		// Parameters initialisation
		video >> prev_frame; // Getting first frame
		video >> curr_frame; // Getting second frame
		curr_frame_idx = video.get(CV_CAP_PROP_POS_FRAMES);
		prev_frame_idx = curr_frame_idx - 1;
		loop_idx = 0;
		mean = 0;

		// Main Loop
		while(curr_frame_idx < _stop_at_frame && !curr_frame.empty()){ // While frames remain
			// Getting next frame
			prev_frame_idx = curr_frame_idx;
			if (save_in_between_frames && stock_in_between_frames){
				video_skip_frames_stock(video, _skip_frames, in_btw_frm_stocked, remove_identic_frames);
			} else {
				video_skip_frames(video, _skip_frames);
			}
			get_next_frame(video, prev_frame, curr_frame);
			curr_frame_idx = video.get(CV_CAP_PROP_POS_FRAMES);
			if (curr_frame.empty()){
				break;
			}

			// Computing difference between frames
			if (compute_difference){
				diff = abs(difference(prev_frame, curr_frame, 10)); // Returns the difference matrix and the value
			}

			// Ajusting mean
			if (compute_difference){
				if (loop_idx == 0){mean = mean + diff;}
				else {mean += (diff - mean)/loop_idx;}

				// If difference is unusually high, write the picture into the given directory
				diff_coef = abs((diff - mean))/mean;
			}

			// Displaying script advancement
			if (display_interval != 0 && loop_idx % display_interval == 0){
				if (!compute_difference){
					DISPLAY(cout << "Frame: " << curr_frame_idx
					<< format(" (%.2f %%)", min(100., ((double) 100*(curr_frame_idx - start_at_frame))/(_stop_at_frame-start_at_frame))) << endl);
				} else {
					DISPLAY(cout << "Frame: " << curr_frame_idx
					<< format(" (%.2f %%)", min(100., ((double) 100*(curr_frame_idx - start_at_frame))/(_stop_at_frame-start_at_frame))) <<
					"\tMean: " << mean << "\tMean coef: " << diff_coef << endl);
				}
			}

			// Chance to save this frame
			r = my_random();

			// Deciding if user_functions need to be used or not
			do_usr_func = true;
			if (r <= pic_save_proba){
				// If identic frames need to be removed
				if (remove_identic_frames){
					if (are_identic_frames(prev_frame, curr_frame)){
						do_usr_func = false;
					}
				}
				// If difference is too high
				if (compute_difference){
					if (loop_idx > min_mean_counted
					 && diff_coef >= diff_threshhold){
						do_usr_func = false;
					}
				}
			// If the probability to save this picture is too low, don't save it
			} else {
				do_usr_func = false;
			}
			
			// Apply user custom restrictions only if necessary
			if (do_usr_func){
				fff_verified = first_frame_func == NULL   ? true : first_frame_func(prev_frame);
				sff_verified = second_frame_func == NULL  ? true : second_frame_func(curr_frame);
				cff_verified = compare_frame_func == NULL ? true : compare_frame_func(prev_frame, curr_frame);
			} else {
				fff_verified = false;
			}

			if (fff_verified
			&& sff_verified
			&& cff_verified
			&& (!compute_difference
			  || (loop_idx > min_mean_counted
				&& diff_coef < diff_threshhold))
			&& (!remove_identic_frames
			  || (!are_identic_frames(prev_frame, curr_frame)))){
				// Displaying information about difference if needed
				if (compute_difference){
					DISPLAY(cout << "Difference Ratio: " << diff_coef 
					<< " at frames " << curr_frame_idx-_skip_frames 
					<< " " << curr_frame_idx << endl);
				}

				// Creating directory if doesn't exist
				frame_path = out_dir+(string) "/"+to_string(global_counter);
				dir_entry = fs::directory_entry(frame_path);
				if (dir_entry.exists()){
					DISPLAY(cout << "Removing " << dir_entry.path() << endl);
					fs::remove_all(dir_entry.path());
				}
				if (!fs::create_directory(dir_entry.path())){
					DISPLAY(cerr << "Failed to create directory " << dir_entry.path() << endl);
				}

				// Writing frames into the directory
				write_frame(prev_frame, frame_path, to_string(global_counter)+"_frame_"+to_string(prev_frame_idx)+"_IN.jpg", verbose);
				write_frame(curr_frame, frame_path, to_string(global_counter)+"_frame_"+to_string(curr_frame_idx)+"_OUT.jpg", verbose);

				// Writing stocked in-between frames
				if (save_in_between_frames && stock_in_between_frames){
					// Creating in-between frames directory
					frame_path += (string) "/"
								+(string) BET_FRAMES_DIRNAME;
					dir_entry = fs::directory_entry(frame_path);
					if (dir_entry.exists()){
						DISPLAY(cout << "Removing " << dir_entry.path() << endl);
						fs::remove_all(dir_entry.path());
					}
					if (!fs::create_directory(dir_entry.path())){
						DISPLAY(cerr << "Failed to create directory " << dir_entry.path() << endl);
					}
					// Writing frames into the directory
					unsigned int tmp_idx = prev_frame_idx+1;
					Mat *tmp_mat;
					Mat *tmp_mat_prev = new Mat();
					while(!in_btw_frm_stocked.empty()){
						tmp_mat = in_btw_frm_stocked.front();
						in_btw_frm_stocked.pop();
						if (!remove_identic_frames
						|| (remove_identic_frames
							&& !tmp_mat_prev->empty()
							&& !are_identic_frames(*tmp_mat_prev, *tmp_mat))){
							write_frame(*tmp_mat, frame_path, to_string(global_counter)+"_frame_"+to_string(tmp_idx)+".jpg", false);
						}
						tmp_idx++;
						tmp_mat_prev->release();
						*tmp_mat_prev = tmp_mat->clone();
						delete tmp_mat;
					}
					delete tmp_mat_prev;
					DISPLAY(cout << format("Wrote frame %u to %u into ", prev_frame_idx+1, tmp_idx)
					 << frame_path << endl;)
				}

				// Saving indexes to save the in-between frames if needed
				if (save_in_between_frames && !stock_in_between_frames){
					in_btw_frm_indexes.push(global_counter);
					in_btw_frm_indexes.push(prev_frame_idx);
					in_btw_frm_indexes.push(curr_frame_idx);
				}

				global_counter++;
			}

			// Cleaning memory and exiting program if timeout is reached
			if (difftime(time(NULL), t0) > _timeout){
				prev_frame.release();
				curr_frame.release();
				video.release();
				empty_frame_stock(in_btw_frm_stocked);
				delete vid_path_queue;
				DISPLAY(cout << "Timeout reached.\n");
				return EXIT_FAILURE;
			}

			empty_frame_stock(in_btw_frm_stocked);
			loop_idx++;
		}

		// Cleaning memory
		prev_frame.release();
		curr_frame.release();
		video.release();

		// Relooping on the video to save in-between frames if necessary
		if (save_in_between_frames && !stock_in_between_frames){
			// Reload video
			video = VideoCapture(filepath);
			if(!video.isOpened()){
				cerr << "Couldn't open " << filepath << endl;
				continue;
			}

			// Moving the video to the start frame
			video_skip_frames(video, start_at_frame);

			// Parameters reinitialisation
			video >> curr_frame; // Getting second frame
			curr_frame_idx = video.get(CV_CAP_PROP_POS_FRAMES);
			prev_frame = curr_frame.clone(); // Just to avoid "uninitialized" warning

			unsigned int folder_id, frame_inf, frame_sup = 0;
			while(true){ // break is computed later (curr_frame_idx must be modified)
				// Getting next frame
				if (remove_identic_frames){
					prev_frame = curr_frame.clone();
				}
				curr_frame.release();
				video >> curr_frame;
				curr_frame_idx = video.get(CV_CAP_PROP_POS_FRAMES);
				if (curr_frame.empty()){
					break;
				}

				DISPLAY(cout << "Saving in-between frames... "
				<< format(" (%.2f %%) ", min(100., ((double) 100*(curr_frame_idx - start_at_frame))/(_stop_at_frame-start_at_frame)))
				 << "\r");

				if (curr_frame_idx > frame_sup){
					if (in_btw_frm_indexes.empty()){
						break;
					}
					// Retrieving path to current animation
					folder_id = in_btw_frm_indexes.front();
					frame_path = out_dir+(string) "/"
							+to_string(folder_id)
							+(string) "/"
							+(string) BET_FRAMES_DIRNAME;
					in_btw_frm_indexes.pop();

					// Retrieving frames indexes
					frame_inf = in_btw_frm_indexes.front();
					in_btw_frm_indexes.pop();
					frame_sup = in_btw_frm_indexes.front();
					in_btw_frm_indexes.pop();

					// Creating "in-between" directory 
					dir_entry = fs::directory_entry(frame_path);
					if (!dir_entry.exists()){
						if (!fs::create_directory(dir_entry.path())){
							DISPLAY(cerr << "Failed to create directory " << dir_entry.path() << endl);
						}
					}
				}

				// Saving in-between frames
				if (frame_inf < curr_frame_idx && curr_frame_idx < frame_sup){
					if (!remove_identic_frames
						|| (remove_identic_frames && !are_identic_frames(prev_frame, curr_frame))){
						write_frame(curr_frame, frame_path, to_string(folder_id)+"_frame_"+to_string(curr_frame_idx)+".jpg", false);
					}
				}

				// Cleaning memory and exiting program if timeout is reached
				if (difftime(time(NULL), t0) > _timeout){
					prev_frame.release();
					curr_frame.release();
					video.release();
					empty_frame_stock(in_btw_frm_stocked);
					delete vid_path_queue;
					DISPLAY(cout << "Timeout reached.\nLast complete in-between frame folder: " << folder_id-1 << endl);
					return EXIT_SUCCESS;
				}

			}
			empty_frame_stock(in_btw_frm_stocked);
			curr_frame.release();
			prev_frame.release();
			video.release();
		}
	}

	delete vid_path_queue;

	DISPLAY(cout << "Exited successfully." << endl);

	return EXIT_SUCCESS;
}


/*======== DIFFERENCE FUNCTION IMPLEMENTATION ==========*/

typedef cv::Point3_<uchar> Pixel;
class Operator{ // An operator used by Mat::forEach()
	private:
		Operator();
		const Mat *prev;
		const Mat *next;
		int halfside;
		unsigned int diff_len;
		double *diff; // Buffer for difference calculus
		Mat diff_mat;

	public:
		Operator(const Mat *prev, const Mat *next, const unsigned int side){
			assert(prev->dims == next->dims && prev->size() == next->size() && prev->channels() == next->channels());
			this->prev     = prev;
			this->next     = next;
			this->halfside = (int) side/2;
			this->diff_len = next->rows*next->cols;
			this->diff     = new double[this->diff_len]; // One case for each computed pixel to prevent parallel issue
		}

		// Local difference operation (called for each pixel)
		void operator()(Pixel &pixel, const int * pos) const{
			// Submatrix extraction around current pixel (O(1))
			Range r_x = Range(max(pos[0]-halfside, 0), min(pos[0]+halfside+1, this->prev->rows));
			Range r_y = Range(max(pos[1]-halfside, 0), min(pos[1]+halfside+1, this->prev->cols));
			const Mat sub_prev = this->prev->operator()(r_x, r_y);
			const Mat sub_next = this->next->operator()(r_x, r_y);

			// Submatrix difference calculation
			Scalar sum_diff = sum(sub_next) - sum(sub_prev); // Difference on each channel
			this->diff[pos[0]*prev->cols + pos[1]] 
				= abs(sum(sum_diff)[0]) / (sub_prev.rows * sub_prev.cols); // Total difference ratio
		}

		// Retrieve difference value
		// /!\ Call this after the forEach function, not before !!
		double getDiff(){
			double ret = 0;
			for (unsigned int i = 0; i < this->diff_len; i++){
				ret += this->diff[i];
			}
			delete[] this->diff; // For unknown segfault issue, this delete is here instead of the class destructor
			return ret;
		}
};

double difference(const Mat &prev, const Mat &next, const unsigned int area_side = 5){
	//Note: Buffer are used because of the "const" qualifier of Operator::operator() requiered by Mat::forEach
	Operator op = Operator(&prev, &next, area_side);
	prev.forEach<Pixel>(op);
	return op.getDiff();
}


/*========= BOOST =========*/
#ifdef BOOST_PYTHON /* Can't work because BOOST only accept 15 args/function maximum */
#include <boost/python.hpp>

BOOST_PYTHON_FUNCTION_OVERLOADS(ef_overloads, extractFrames, 2, 20) // For default arguments

BOOST_PYTHON_MODULE(extractFrames){
    using namespace boost::python;
    def("extractFrames", extractFrames, ef_overloads());
}
#endif


/*======== PYTHON ==========*/
#ifdef USE_PYTHON
// Argument retrieving function
static PyObject *
PyExtractFrames(PyObject *self, PyObject *args, PyObject *kwargs){
	//Arguments list with default arguments
	char *in_path;
	char *out_dir;
	unsigned int skip_frames             = 1;
	double       skip_seconds            = 0;
	unsigned int start_at_frame          = 0;
	unsigned int stop_at_frame           = 0;
	unsigned int display_interval        = 1;
	bool         save_in_between_frames  = true;
	bool         stock_in_between_frames = true;
	bool         remove_identic_frames   = false;
	bool         compute_difference      = false;
	unsigned int min_mean_counted        = 20;
	double       diff_threshhold         = 0.20;
	bool         verbose                 = true;
	double       pic_save_proba          = 1;
	double       file_proportion         = 1;
	double       timeout                 = 0;
	bool (*first_frame_func)(const Mat &)  = NULL;
	bool (*second_frame_func)(const Mat &) = NULL;
	bool (*compare_frame_func)(const Mat &, const Mat &) = NULL;

	// Keyword list for keyword arguments
	static char *kwlist[] = {
		 	"skip_frames",
			"skip_seconds",
			"start_at_frame",
			"stop_at_frame",
			"display_interval",
			"save_in_between_frames",
			"stock_in_between_frames",
			"remove_identic_frames",
			"compute_difference",
			"min_mean_counted",
			"diff_threshhold",
			"verbose",
			"pic_save_proba",
			"file_proportion",
			"timeout",
			"first_frame_func",
			"second_frame_func",
			"compare_frame_func",
			NULL};

	// Retrieving arguments into local C++ variables
    if (!PyArg_ParseTupleAndKeywords(args, kwargs,
			"ss|$IdIIIppppIdpddd000:", kwlist,
			&in_path, 
			&out_dir,
			&skip_frames,
			&skip_seconds,
			&start_at_frame,
			&stop_at_frame,
			&display_interval,
			&save_in_between_frames,
			&stock_in_between_frames,
			&remove_identic_frames,
			&compute_difference,
			&min_mean_counted,
			&diff_threshhold,
			&verbose,
			&pic_save_proba,
			&file_proportion,
			&timeout,
			&first_frame_func,
			&second_frame_func,
			&compare_frame_func)){
		PyErr_SetString(PyExc_TypeError, "Invalid parameters");
        return NULL;
	}

	// Executing extractFrames function
    int error = extractFrames(in_path, out_dir,
			skip_frames,
			skip_seconds,
			start_at_frame,
			stop_at_frame,
			display_interval,
			save_in_between_frames,
			stock_in_between_frames,
			remove_identic_frames,
			compute_difference,
			min_mean_counted,
			diff_threshhold,
			verbose,
			pic_save_proba,
			file_proportion,
			timeout,
			first_frame_func,
			second_frame_func,
			compare_frame_func);
    return PyLong_FromLong(error);
};

// Python extractFrame definition
static PyMethodDef extractFrameMethods[] = 
{
    {"extractFrames", (PyCFunctionWithKeywords) PyExtractFrames, METH_VARARGS | METH_KEYWORDS,
	 " Write frames from a video file into the output directory with various adjustable parameters.\n\
Also allow the possibility to save the in-between frames. \n\
\n\
-in_path: The path to a video file or a directory containing video files.\n\
-out_dir: The path to the output directory. If it doesn't exist, a new directory will be created.\n\
-skip_frames: The interval between each pair of frames that have to be compared and saved. For example, a *skip_frames* of 5 will compare and save frames 1-5, 6-11, 12-17...\n\
-skip_seconds: Same as skip_frames, but the unit is in seconds instead. If both *skip_frames* and *skip_seconds* are > 0, the priority is given to *skip_seconds*.\n\
-start_at_frame: The starting frame index.\n\
-stop_at_frame: The stoping frame index.\n\
-display_interval: The interval for information display. This is not by frame, but by loop. Each 'skip_frame' frames count for 1 loop. For example, having 'skip_frame' to 5 and *display_interval* to 3 will result in display at frames 1-5, 18-23...\n\
-save_in_between_frames: Whether or not in-between frames have to be saved.\n\
-stock_in_between_frames: Whether or not in-between frames have to be stocked in memory or saved later while relooping the video. It speeds up the function, but if too many frames are stored into the memory you might want to set this to *false*.\n\
-remove_identic_frames: Whether or not identic frames has to be removed or not. It is calculated for every successive frames so it slowers a bit the function.\n\
-compute_difference: Whether or not the difference between each pair of frames has to be calculated, in order to save only pictures that aren't very different. This slowers a lot the function.\n\
-min_mean_counted: The number of differences that have to be computed in order to have a meaningful mean.\n\
-diff_threshhold: The threshold idicating when frames are considered to too different (values are around 0~2).\n\
-verbose: Whether or not informations has to be displayed on the screen.\n\
-pic_save_proba: The probability to save a pair of frames or not (unselected pair won't compute difference or user-specified test functions)\n\
-file_proportion: The probability compute a found video or not (if too many videos are in the same directory).\n\
-timeout: A timeout in seconds. Note the if 'stock_in_between_frames' is set to 0, the last video's in-between frames might not be all saved.\n\
-first_frame_func: A user-specified function to test a property on the first frame. If this property if not verified, the pair of frames won't be saved.\n\
-second_frame_func: A user-specified function to test a property on the second frame. If this property if not verified, the pair of frames won't be saved.\n\
-compare_frame_func: A user-specified function to test a property on the pair of frames. If this property if not verified, the pair of frames won't be saved.\n\
"},
    {NULL, NULL, 0, NULL}
};
#endif


/*======== MAIN ==========*/
#ifndef NOMAIN

void usage(char* name){
	cout << "Usage: " << name << " <path to video or directory> <output directory>" << endl;
}

#define PARAM 2
int main(int argc, char** argv)
{
	if (argc != PARAM+1){
		usage(argv[0]);
		return -1;
	}
	extractFrames(argv[1], argv[2],
			100,    // skip_frames
			0,     // skip_seconds
			10000,    // start_at_frame
			17000,   // stop_at_frame
			10,     // display_interval
			false,  // save_in_between_frames
			false, // stock_in_between_frames
			false,  // remove_identic_frames
			false, // compute_difference
			20,    // min_mean_counted
			0.16,     // diff_threshhold
			true,  // verbose
			0.06,   // pic_save_proba
			0.2,  // file_proportion
			7200,     // timeout
			NULL,  // first_frame_func
			NULL,  // second_frame_func
			NULL); // compare_frame_func
    return 0;
}
#endif

