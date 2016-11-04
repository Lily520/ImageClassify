#include<opencv2\opencv.hpp>
#include<iostream> 
#include<fstream>
#include<iomanip>
#include <stdio.h>  
#include <stdlib.h>  
#include<string>

using namespace std;
using namespace cv;

class HistogramND {
private:
	Mat image;//源图像
	int hisSize[1], hisWidth, hisHeight;//直方图的大小,宽度和高度
	float range[2];//直方图取值范围
	const float *ranges;
	Mat channelsRGB[3];//分离的BGR通道
	MatND outputRGB[3];//输出直方图分量
public:
	HistogramND() {
		hisSize[0] = 256;//颜色空间划分的区间
		hisWidth = 400;
		hisHeight = 400;
		range[0] = 0.0;
		range[1] = 255.0;
		ranges = &range[0];
	}

	//导入图片
	bool importImage(String path) {
		image = imread(path);
		if (!image.data)
			return false;
		return true;
	}

	//分离通道
	void splitChannels() {
		split(image, channelsRGB);
	}

	//计算直方图
	void getHistogram(int sum) {
		calcHist(&channelsRGB[0], 1, 0, Mat(), outputRGB[0], 1, hisSize, &ranges);
		calcHist(&channelsRGB[1], 1, 0, Mat(), outputRGB[1], 1, hisSize, &ranges);
		calcHist(&channelsRGB[2], 1, 0, Mat(), outputRGB[2], 1, hisSize, &ranges);

		//输出各个bin的值
		int count = 1;
		std::ofstream fout("C:\\Caltech\\color_all.txt", ios::app);
		fout <<sum<< " ";
		cout << "sum:" << sum << endl;
		for (int i = 0; i < hisSize[0]; ++i) {
			fout << (count++) << ":" << outputRGB[0].at<float>(i) << " ";
			fout << (count++) << ":" << outputRGB[1].at<float>(i) << " ";
			fout << (count++) << ":" << outputRGB[2].at<float>(i) << " ";
		}//for
		fout << endl;
		fout.close();
	}

	//显示直方图
	/*void displayHisttogram() {
	Mat rgbHist[3];
	for (int i = 0; i < 3; i++)
	{
	rgbHist[i] = Mat(hisWidth, hisHeight, CV_8UC3, Scalar::all(0));
	}
	normalize(outputRGB[0], outputRGB[0], 0, hisWidth - 20, NORM_MINMAX);
	normalize(outputRGB[1], outputRGB[1], 0, hisWidth - 20, NORM_MINMAX);
	normalize(outputRGB[2], outputRGB[2], 0, hisWidth - 20, NORM_MINMAX);
	for (int i = 0; i < hisSize[0]; i++)
	{
	int val = saturate_cast<int>(outputRGB[0].at<float>(i));
	rectangle(rgbHist[0], Point(i * 2 + 10, rgbHist[0].rows), Point((i + 1) * 2 + 10, rgbHist[0].rows - val), Scalar(0, 0, 255), 1, 8);
	val = saturate_cast<int>(outputRGB[1].at<float>(i));
	rectangle(rgbHist[1], Point(i * 2 + 10, rgbHist[1].rows), Point((i + 1) * 2 + 10, rgbHist[1].rows - val), Scalar(0, 255, 0), 1, 8);
	val = saturate_cast<int>(outputRGB[2].at<float>(i));
	rectangle(rgbHist[2], Point(i * 2 + 10, rgbHist[2].rows), Point((i + 1) * 2 + 10, rgbHist[2].rows - val), Scalar(255, 0, 0), 1, 8);
	}

	cv::imshow("R", rgbHist[0]);
	imshow("G", rgbHist[1]);
	imshow("B", rgbHist[2]);
	imshow("image", image);
	}*/
};


int main() {
	ifstream in_out("C:\\Users\\Administrator\\Desktop\\毕设\\datasets\\caltech\\101.txt");
	string s, path;
	string s_out;
	HistogramND hist;
	int sum = -1;
	while (getline(in_out, s_out)) {
		ifstream in("C:\\Users\\Administrator\\Desktop\\毕设\\datasets\\caltech\\" + s_out + ".txt");
		sum++;
		while (getline(in, s)) {
			path = "C:\\Users\\Administrator\\Desktop\\毕设\\datasets\\caltech\\101_ObjectCategories\\" + s_out + "\\" + s;
			if (!hist.importImage(path)) {
				cout << "Import Error!" << endl;
				return -1;
			}
			hist.splitChannels();
			hist.getHistogram(sum);
			//hist.displayHisttogram();
			waitKey(0);
		}

		in.close();
	}
	
	//string path = "C:\\1.jpg";
	

	return 0;
}
