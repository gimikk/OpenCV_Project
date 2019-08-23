#include <stdio.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include <vector>
#include <math.h>
#include <stdlib.h>
#include <random>


using namespace cv;
using namespace std;

#define Octave 2
#define Gaussian 6

// feature_P를 저장할 구조체, Point와 radius로 구성됨
struct feature_P
{
	int x;
	int y;
	int radius;
};

void* LBP(Mat& image, int x, int y, int *result)
{
	int cnt = 0;
	int standard = image.at<uchar>(y, x);
	for (int m = y - 7; m <= y + 7; m++)
		for (int n = x - 7; n <= x + 7; n++)
		{
			// 중간점은 캔슬
			if (n == x && m == y)
				continue;
			if (standard > image.at<uchar>(m, n))
				result[cnt++] = 0;
			else
				result[cnt++] = 1;
		}

	return result;
}


int main()
{
	//이미지 흑백으로 만듦
	Mat image0 = imread("41006.jpg");
	Mat image;
	cvtColor(image0, image, CV_BGR2GRAY);
	const int width = image.cols;
	const int height = image.rows; 
	vector<feature_P> vec;

	// vec에 저장된 feature Point를 txt 파일에 저장
	errno_t err;
	FILE *fp;
	err = fopen_s(&fp,"test_LBP.txt", "w");

	for (int i = 8; i < height - 8; i++)
	{
		for (int j = 8; j < width - 8; j++)
		{
			int result[224];
			LBP(image, j, i, result);
		
			fprintf(fp, "0 ");
			for (int n = 0; n < 224; n++)
				fprintf(fp, "%d:%d ", n + 1, result[n]);
			fprintf(fp, "\n");
		}
	}

	//파일 닫기
	fclose(fp);
	return 0;
}
