#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <math.h>

#define oSize 2 // 옥타브 최대 번호

int main()
{
	//이미지 로드
	cv::Mat image = cv::imread("img1.bmp");

	//이미지 가로, 세로
	const int width = image.cols;
	const int height = image.rows;

	cv::Mat G[oSize][6]; // 가우시안 이미지
	cv::Mat DoG[oSize][5]; // DoG 이미지
	cv::Size mask(3, 3);

	double sigma[6]; // 가우시안 이미지 번호에 따른 시그마 배열
	double k = pow(2.0, 1.0 / 3.0);

	//시그마 초기화
	sigma[0] = 1.6;
	for (int i = 1; i < 6; ++i)
	{
		sigma[i] = pow(k, i)*sigma[0];
	}

	//스케일 공간 생성
	double sig = sqrt(pow(sigma[0], 2.0) - pow(0.5, 2.0)); // 입력 영상 자체가 이미 0.5로 스무딩 되었다고 가정
	cv::GaussianBlur(image, G[0][0], mask, sig, sig); // 스케일 공간 구성의 토대 영상
	cv::cvtColor(G[0][0], G[0][0], CV_BGR2GRAY); //


	for (int j = 0; j < oSize; ++j) // 옥타브 번호
	{
		for (int i = 1; i < 6; ++i) // 영상 번호
		{
			double s = sqrt(pow(sigma[i], 2) - pow(sigma[i - 1], 2));
			cv::GaussianBlur(G[j][i - 1], G[j][i], cv::Size(2 * (i - 1) + 3, 2 * (i - 1) + 3), s, s); // 이전 영상에서 스케일의 차이만큼 스무딩 

			cv::absdiff(G[j][i], G[j][i - 1], DoG[j][i - 1]); // 차영상 DoG 채널 생성
															  //DoG[j][i - 1] = G[j][i] - G[j][i - 1];

			if (i == 3 && j < oSize - 1)
			{
				cv::pyrDown(G[j][i], G[j + 1][0], cv::Size(G[j][i].cols / 2, G[j][i].rows / 2)); // 네번째 영상을 다음 옥타브의 첫번째 영상으로 취함
			}
		}
	}
	printf("0");

	//특징점 찾기
	for (int o = 0; o < oSize; ++o) // 옥타브 번호
	{
		for (int d = 1; d < 4; ++d) // DoG 번호
		{
			printf("\n1");
			for (int w = 1; w < width / pow(2, o) - 1; ++w) // width 좌표
			{
				for (int h = 1; h < height / pow(2, o) - 1; ++h) // height 좌표
				{
					// DoG 맵 구성 (인접 이웃점)
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
						if (map[j] > max) max = map[j]; // 이웃의 최대값

					for (int j = 1; j < 26; j++)
						if (map[j] < min) min = map[j]; // 이웃의 최소값

					int y = h * pow(2, o), x = w * pow(2, o); // 옥타브에 대한 좌표 조정

															  //주위 26개 이웃에 대해 최저 또는 최대인 점 검출
					if (DoG[o][d].at<uchar>(h, w) < min || DoG[o][d].at<uchar>(h, w) > max)
					{
						cv::circle(image, cv::Point(x, y), 1.6*pow(2, (o + d) / 3), cv::Scalar(0, 255, 0), 1); // 특징점 display
					}
				}
			}
		}
	}
	printf("\n0");

	//이미지 저장
	cv::imwrite("resultIMG.bmp", image);

	return 0;
}