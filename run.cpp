#include <opencv2/opencv.hpp>
#include <iostream>
#include "fhog.h"

using namespace cv;
using namespace std;

int main()
{
	int scale = 96;
	//读取图片
	Mat img = imread('img path');
	if (img.empty())
	{
		break;
	}
	Mat img_fix;
	resize(img, img_fix, Size(scale, scale)); //标准化图片大小

	//提取fhog特征图
	int cell_size = 4;
	int size_patch[3];
	IplImage z_ipl = img_fix;
	Mat FeaturesMap;
	CvLSVMFeatureMapCaskade *map;
	getFeatureMaps(&z_ipl, cell_size, &map);
	normalizeAndTruncate(map, 0.2f);
	PCAFeatureMaps(map);
	size_patch[0] = map->sizeY;
	size_patch[1] = map->sizeX;
	size_patch[2] = map->numFeatures;
	FeaturesMap = cv::Mat(cv::Size(map->numFeatures, map->sizeX*map->sizeY), CV_32F, map->map);  // Procedure do deal with cv::Mat multichannel bug
	FeaturesMap = FeaturesMap.t();
	freeFeatureMapObject(&map);

	//构建特征向量(sizeY*sizeX*numFeatures)*1
	Mat test_line;
	test_line = FeaturesMap.reshape(1, 1); 


	//特征提取完毕，进行进一步的操作





	return 0;
}