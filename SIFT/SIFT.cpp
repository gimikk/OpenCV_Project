#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <math.h>

#define oSize 2 // ��Ÿ�� �ִ� ��ȣ

int main()
{
	//�̹��� �ε�
	cv::Mat image = cv::imread("img1.bmp");

	//�̹��� ����, ����
	const int width = image.cols;
	const int height = image.rows;

	cv::Mat G[oSize][6]; // ����þ� �̹���
	cv::Mat DoG[oSize][5]; // DoG �̹���
	cv::Size mask(3, 3);

	double sigma[6]; // ����þ� �̹��� ��ȣ�� ���� �ñ׸� �迭
	double k = pow(2.0, 1.0 / 3.0);

	//�ñ׸� �ʱ�ȭ
	sigma[0] = 1.6;
	for (int i = 1; i < 6; ++i)
	{
		sigma[i] = pow(k, i)*sigma[0];
	}

	//������ ���� ����
	double sig = sqrt(pow(sigma[0], 2.0) - pow(0.5, 2.0)); // �Է� ���� ��ü�� �̹� 0.5�� ������ �Ǿ��ٰ� ����
	cv::GaussianBlur(image, G[0][0], mask, sig, sig); // ������ ���� ������ ��� ����
	cv::cvtColor(G[0][0], G[0][0], CV_BGR2GRAY); //


	for (int j = 0; j < oSize; ++j) // ��Ÿ�� ��ȣ
	{
		for (int i = 1; i < 6; ++i) // ���� ��ȣ
		{
			double s = sqrt(pow(sigma[i], 2) - pow(sigma[i - 1], 2));
			cv::GaussianBlur(G[j][i - 1], G[j][i], cv::Size(2 * (i - 1) + 3, 2 * (i - 1) + 3), s, s); // ���� ���󿡼� �������� ���̸�ŭ ������ 

			cv::absdiff(G[j][i], G[j][i - 1], DoG[j][i - 1]); // ������ DoG ä�� ����
															  //DoG[j][i - 1] = G[j][i] - G[j][i - 1];

			if (i == 3 && j < oSize - 1)
			{
				cv::pyrDown(G[j][i], G[j + 1][0], cv::Size(G[j][i].cols / 2, G[j][i].rows / 2)); // �׹�° ������ ���� ��Ÿ���� ù��° �������� ����
			}
		}
	}
	printf("0");

	//Ư¡�� ã��
	for (int o = 0; o < oSize; ++o) // ��Ÿ�� ��ȣ
	{
		for (int d = 1; d < 4; ++d) // DoG ��ȣ
		{
			printf("\n1");
			for (int w = 1; w < width / pow(2, o) - 1; ++w) // width ��ǥ
			{
				for (int h = 1; h < height / pow(2, o) - 1; ++h) // height ��ǥ
				{
					// DoG �� ���� (���� �̿���)
					int map[26] = { DoG[o][d - 1].at<uchar>(h, w), DoG[o][d - 1].at<uchar>(h, w - 1), DoG[o][d - 1].at<uchar>(h, w + 1),
						DoG[o][d - 1].at<uchar>(h - 1, w), DoG[o][d - 1].at<uchar>(h - 1, w - 1), DoG[o][d - 1].at<uchar>(h - 1, w + 1),
						DoG[o][d - 1].at<uchar>(h + 1, w), DoG[o][d - 1].at<uchar>(h + 1, w - 1), DoG[o][d - 1].at<uchar>(h + 1, w + 1),
						DoG[o][d].at<uchar>(h, w - 1), DoG[o][d].at<uchar>(h, w + 1),
						DoG[o][d].at<uchar>(h - 1, w), DoG[o][d].at<uchar>(h - 1, w - 1), DoG[o][d].at<uchar>(h - 1, w + 1),
						DoG[o][d].at<uchar>(h + 1, w), DoG[o][d].at<uchar>(h + 1, w - 1), DoG[o][d].at<uchar>(h + 1, w + 1),
						DoG[o][d + 1].at<uchar>(h, w), DoG[o][d + 1].at<uchar>(h, w - 1), DoG[o][d + 1].at<uchar>(h, w + 1),
						DoG[o][d + 1].at<uchar>(h - 1, w), DoG[o][d + 1].at<uchar>(h - 1, w - 1), DoG[o][d + 1].at<uchar>(h - 1, w + 1),
						DoG[o][d + 1].at<uchar>(h + 1, w), DoG[o][d + 1].at<uchar>(h + 1, w - 1), DoG[o][d + 1].at<uchar>(h + 1, w + 1) };

					int max = map[0], min = map[0];

					for (int j = 1; j < 26; j++)
						if (map[j] > max) max = map[j]; // �̿��� �ִ밪

					for (int j = 1; j < 26; j++)
						if (map[j] < min) min = map[j]; // �̿��� �ּҰ�

					int y = h * pow(2, o), x = w * pow(2, o); // ��Ÿ�꿡 ���� ��ǥ ����

															  //���� 26�� �̿��� ���� ���� �Ǵ� �ִ��� �� ����
					if (DoG[o][d].at<uchar>(h, w) < min || DoG[o][d].at<uchar>(h, w) > max)
					{
						cv::circle(image, cv::Point(x, y), 1.6*pow(2, (o + d) / 3), cv::Scalar(0, 255, 0), 1); // Ư¡�� display
					}
				}
			}
		}
	}
	printf("\n0");

	//�̹��� ����
	cv::imwrite("resultIMG.bmp", image);

	return 0;
}