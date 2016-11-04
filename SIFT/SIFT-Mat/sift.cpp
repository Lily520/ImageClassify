#include "opencv2/highgui/highgui.hpp"  
#include "opencv2/imgproc/imgproc.hpp"  
#include "opencv2/nonfree/nonfree.hpp"  
#include "opencv2/nonfree/features2d.hpp"  
#include <iostream>  
#include<fstream>
#include<iomanip>
#include <stdio.h>  
#include <stdlib.h>  
#include<string>

using namespace cv;
using namespace std;

int main()
{
	initModule_nonfree(); //初始化模块，使用SIFT或SURF时用到 
	Ptr<FeatureDetector> detector = FeatureDetector::create("SIFT");//创建SIFT特征检测器 
	Ptr<DescriptorExtractor> descriptor_extractor = DescriptorExtractor::create("SIFT");//创建特征向量生成器 
	Ptr<DescriptorMatcher> descriptor_matcher = DescriptorMatcher::create("BruteForce");//创建特征匹配器 
	if (detector.empty() || descriptor_extractor.empty())
		cout << "fail to create detector!";

	ifstream in("C:\\Users\\Administrator\\Desktop\\毕设\\datasets\\caltech\\101_ObjectCategories\\anchor.txt");
	string s;
	string ss, s1;
	int len;
	int flag;

	Mat descriptors1;
	vector<KeyPoint> keypoints1;
	Mat img1;
	//读入图像 
	while (getline(in, s)) {
		ss = "C:\\Users\\Administrator\\Desktop\\毕设\\datasets\\caltech\\101_ObjectCategories\\anchor\\" + s;
		//file_path = ss;
		img1 = imread(ss);
		//特征点检测 
		double t = (double)getTickCount();//当前滴答数 
		detector->detect(img1, keypoints1); //检测img1中的SIFT特征点，存储到keypoints1中 

		cout << "图像1特征点个数:" << keypoints1.size() << endl;

		//根据特征点计算特征描述子矩阵，即特征向量矩阵 
		descriptor_extractor->compute(img1, keypoints1, descriptors1);

		t = ((double)getTickCount() - t) / getTickFrequency();
		cout << "SIFT算法用时：" << t << "秒" << endl;


		cout << "图像1特征描述矩阵大小：" << descriptors1.size()
			<< "，特征向量个数：" << descriptors1.rows << "，维数：" << descriptors1.cols << endl;

		len = s.length();
		s1 = s.substr(0, len - 4);
		//std::ofstream fout("C:\\VOC2010\\" + s1 + ".txt");
		std::ofstream fout("C:\\VOC2010\\sift.txt",ios::app);
		if (!fout.is_open())
		{
			cout << "打开文件失败" << endl;
		}

		// 检查矩阵是否为空 
		if (descriptors1.empty())
		{
			cout << "矩阵为空" << endl;
			return 0;
		}
		fout << "0 ";
		flag = 1;
		for (int r = 0; r < descriptors1.rows; r++)
		{
			for (int c = 0; c < descriptors1.cols; c++)
			{
				float data = descriptors1.at<float >(r, c);  //读取数据，at<type> - type 是矩阵元素的具体数据格式 
				if (data != 0)
					fout << (flag++) << ":" << data << " ";   //每列数据用空格隔开 
				else
					flag++;
			}
			
		}
		fout << endl;  //换行 
		fout.close();
		
	}//while
	in.close();
	return 0;

}
