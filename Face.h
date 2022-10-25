#include <opencv2\opencv_modules.hpp>
#include <iostream>
#include <stdlib.h>
#include <opencv2\objdetect\objdetect.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\core\core.hpp>
#include <opencv2\core.hpp>
#include <opencv2/face.hpp>
#include <fstream>
#include <sstream>
#include <direct.h>
using namespace std;
using namespace cv;
using namespace cv::face;


CascadeClassifier faceDetect;

string file;
string name;
int NumOfFiles = 0;


void Recognize(Mat frame) {

	vector<Rect> faces;
	Mat grayFrame;
	Mat cut;
	Mat resolution;
	Mat gray;
	string text;
	stringstream sstm;

	cvtColor(frame, grayFrame, COLOR_BGR2GRAY);

	equalizeHist(grayFrame, grayFrame);

	faceDetect.detectMultiScale(grayFrame, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

	Rect roi_b;
	Rect roi_c;

	size_t ic = 0;
	int ac = 0;

	size_t ib = 0;
	int ab = 0;


	/*
		resizes the image and saves in the location as specified
		by the string file var.
	*/

	for (ic = 0; ic < faces.size(); ic++)

	{
		roi_c.x = faces[ic].x;
		roi_c.y = faces[ic].y;
		roi_c.width = (faces[ic].width);
		roi_c.height = (faces[ic].height);

		ac = roi_c.width * roi_c.height;

		roi_b.x = faces[ib].x;
		roi_b.y = faces[ib].y;
		roi_b.width = (faces[ib].width);
		roi_b.height = (faces[ib].height);


		cut = frame(roi_b);
		resize(cut, resolution, Size(128, 128), 0, 0, INTER_LINEAR);
		cvtColor(cut, gray, COLOR_BGR2GRAY);
		stringstream ssfn;
		file = "C:\\Users\\Reece K\\Desktop\\FaceApp\\";
		ssfn << file.c_str() << name << NumOfFiles << ".jpg";
		file = ssfn.str();
		imwrite(file, resolution);
		NumOfFiles++;


		Point pt1(faces[ic].x, faces[ic].y);
		Point pt2((faces[ic].x + faces[ic].height), (faces[ic].y + faces[ic].width));
		rectangle(frame, pt1, pt2, Scalar(0, 255, 0), 2, 8, 0);
	}


	sstm << "Crop area size: " << roi_b.width << "x" << roi_b.height << " Filename: " << file;
	text = sstm.str();

	if (!cut.empty())
	{
		imshow("detected", cut);
	}
	else
		destroyWindow("detected");
}

void addFace()
{
	cout << "\nEnter Your Name:  ";
	cin >> name;

	VideoCapture capture(0);

	if (!capture.isOpened())
		cout << "Cannot Open Camera";
		return;

		faceDetect.load("haarcascade_frontalface_alt.xml");

	Mat frame;
	cout << "Position your face infront of the camera\nPress 'C' 10 times to enter your face";
	char key;
	int i = 0;

	for (;;)
	{
		capture >> frame;
		Recognize(frame);
		i++;
		if (i == 10)
		{
			cout << "Capture Success Face Added";
			break;
		}
		int c = waitKey(10);

		if (27 == char(c))
		{
			break;
		}
	}
	return;
}


static void dbread(vector<Mat>& images, vector<int>& labels) {
	vector<cv::String> fn;
	file = "C:\\Users\\Reece K\\Desktop\\FaceApp\\";
	glob(file, fn, false);

	size_t count = fn.size();

	for (size_t i = 0; i < count; i++)
	{
		string itsname = "";
		char sep = '\\';
		size_t j = fn[i].rfind(sep, fn[i].length());
		if (j != string::npos)
		{
			itsname = (fn[i].substr(j + 1, fn[i].length() - j - 6));
		}
		images.push_back(imread(fn[i], 0));
		labels.push_back(atoi(itsname.c_str()));
	}
}

void eigenFaceTrainer() {
	vector<Mat> images;
	vector<int> labels;
	dbread(images, labels);
	cout << "size of the images is " << images.size() << endl;
	cout << "size of the labels is " << labels.size() << endl;
	cout << "Training begins...." << endl;

	//create algorithm eigenface recognizer
	Ptr<EigenFaceRecognizer> model = EigenFaceRecognizer::create();

	//train data
	model->train(images, labels);

	model->save("C:\\Users\\Reece K\\Desktop\\FaceApp\\eigenface.yml");

	cout << "Training finished...." << endl;
	waitKey(10000);
}


void  FaceRecognition() {

	cout << "Facial Recoginition has began..." << endl;

	//tested dataset
	Ptr<FaceRecognizer>  model = FisherFaceRecognizer::create();
	model->read("C:\\Users\\Reece K\\Desktop\\FaceApp\\eigenface.yml");

	Mat testSample = imread("C:\\Users\\Reece K\\Desktop\\FaceApp\\0.jpg", 0);

	int img_width = testSample.cols;
	int img_height = testSample.rows;


	//lbpcascades/lbpcascade_frontalface.xml

	string window = "Capture - face detection";

	if (!faceDetect.load("haarcascade_frontalface_alt.xml")) {
		cout << " An error has occurred" << endl;
		return;
	}

	VideoCapture cap(0);
	if (!cap.isOpened())
	{
		cout << "error has occurred please exit" << endl;
		return;
	}

	namedWindow(window, 1);
	long count = 0;
	string Pname = "";

	while (true)
	{
		vector<Rect> faces;
		Mat frame;
		Mat graySacleFrame;
		Mat original;

		cap >> frame;
		//counting frames
		count = count + 1; 

		if (!frame.empty()) {

			original = frame.clone();

			//image to gray
			cvtColor(original, graySacleFrame, COLOR_BGR2GRAY);

			//detect the face
			faceDetect.detectMultiScale(graySacleFrame, faces, 1.1, 3, 0, cv::Size(90, 90));

			//number of faces detected
			//cout << faces.size() << " faces detected" << endl;
			std::string frameset = std::to_string(count);
			std::string faceset = std::to_string(faces.size());

			int width = 0, height = 0;

			for (int i = 0; i < faces.size(); i++)
			{
				
				Rect face_i = faces[i];

				
				Mat face = graySacleFrame(face_i);
				Mat face_resized;
				cv::resize(face, face_resized, Size(img_width, img_height), 1.0, 1.0, INTER_CUBIC);
				int label = -1; double confidence = 0;
				model->predict(face_resized, label, confidence);

				cout << " confidence " << confidence << " Label: " << label << endl;

				Pname = to_string(label);

				//drawing blue rectagle in around the faces that are regoginezed
				rectangle(original, face_i, CV_RGB(0, 0, 255), 1);
				string text = Pname;

				int pos_x = std::max(face_i.tl().x - 10, 0);
				int pos_y = std::max(face_i.tl().y - 10, 0);

				//the name entered of the person whos face appears
				putText(original, text, Point(pos_x, pos_y), FONT_HERSHEY_COMPLEX_SMALL, 1.0, CV_RGB(0, 255, 0), 1.0);

			}


			putText(original, "Frames: " + frameset, Point(30, 60), FONT_HERSHEY_COMPLEX_SMALL, 1.0, CV_RGB(0, 255, 0), 1.0);
			putText(original, "No. of Persons detected: " + to_string(faces.size()), Point(30, 90), FONT_HERSHEY_COMPLEX_SMALL, 1.0, CV_RGB(0, 255, 0), 1.0);
			
			cv::imshow(window, original);

		}
		if (waitKey(30) >= 0) break;
	}
}

