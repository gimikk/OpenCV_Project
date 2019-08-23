#include <stdio.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>

using namespace cv;
using namespace std;

int main()
{
	//�̹��� ������� ����
	Mat image0 = imread("41006.jpg");
	Mat image;
	cvtColor(image0, image, CV_BGR2GRAY);
	const int width = image.cols;
	const int height = image.rows;

	// vec�� ����� feature Point�� txt ���Ͽ� ����
	errno_t err;
	FILE *fp;
	err = fopen_s(&fp,"svm_predictions.txt", "r");
	
	char s[100];

	// �ܰ��� �κп� ���� ó��
	int cnt = 0;
	while (!feof(fp))
	{
		fgets(s, 80, fp);
		if (s[0] != '-') {
			int first = s[0] - 48;

			if (first >= 1)
			{
				//printf("%d\n ", first);
				int x = cnt % (width - 16) + 8;
				int y = cnt / (width - 16) + 8;
				circle(image0, Point(x, y), 1, Scalar(0, 255, 0), 1);
				
			}
		}
		cnt++;
	}

	imshow("image", image0);
	waitKey(0);

	imwrite("result.jpg", image0);

	//���� �ݱ�
	fclose(fp);
	return 0;
}
