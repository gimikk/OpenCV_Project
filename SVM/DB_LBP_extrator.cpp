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
#include <io.h>
#include <string>


using namespace cv;
using namespace std;

#define Octave 2
#define Gaussian 6

struct OctaveGroup//한 옥타브 그룹당 가지고 있는 정보 가우시안 필터, DoG 채널, 키포인트
{
	Mat gaussianFliter[Gaussian];//한 옥타브 당 가지고 있는 Gaussian Filter
	Mat DoGChannel[Gaussian - 1];//한 옥타브 당 가지고 있는 DoG Channel은 Gaussian-1개.
};

// feature_P를 저장할 구조체, Point와 radius로 구성됨
struct feature_P
{
	int x;
	int y;
	int radius;
};

int CheckSigSize(double sig)//Sig를 이용하여 값을 계산해야 Size를 계산해야 되는데 빈번하게 쓰여서 따로 함수를 묶음.
{
	int size;//출력할 마스크 사이즈

	size = ceil(6 * sig);//6*sig에 올림수를 마스크 사이즈로 임시 설정

	if (size % 2 == 0)//만약 짝수일 경우
		size++;//1을 더함

	return size;//마스크 사이즈 return;
}


void SIFT(Mat& gray,vector<feature_P>& vec)
{
	OctaveGroup octave[Octave];

	double beforeSig; //이전 Sig와의 비교를 이용하여 sqrt(sig_n+1^2 - sig_n^2)를 구하기 위한 용도.

	for (int i = 0; i < Octave; ++i)
	{
		for (int j = 0; j < Gaussian; ++j)
		{
			double sig;
			int size;

			if (j == 0)//첫 가우시안 필터일 경우.
			{
				if (i != 0)//0옥타브가 아닐 경우
				{
					pyrDown(octave[i - 1].gaussianFliter[3], octave[i].gaussianFliter[j],
						Size(octave[i - 1].gaussianFliter[3].cols / 2, octave[i - 1].gaussianFliter[3].rows / 2));//이전 옥타브의 4번째 이미지를 축소하여 가져온다.
					beforeSig = 1.6;//Sig는 1.6이었던 것으로 가정.
				}
				else//0 옥타브일 경우, 원본 이미지에 sig 1.6 값을 이용하여 필터링.
				{
					sig = 1.6;//첫번째 이미지는 Sig가 1.6.
					size = CheckSigSize(sig);//Sig 값에 따른 마스크 사이즈 확인
					GaussianBlur(gray, octave[i].gaussianFliter[j], Size(size, size), sig, sig);//가우시안 블러.
					beforeSig = sig;//Sig 기록
				}
			}
			else
			{
				sig = beforeSig * pow(2.0, 1.0 / 3.0);//Sig_n+1 = Sig_n * k.   k = 2^(1/3)에 대한 코드
				double currentSig = sqrt(pow(sig, 2) - pow(beforeSig, 2));//sqrt(Sig_n+1^2 - Sig_n^2)에 대한 코드.
				size = CheckSigSize(currentSig);//Sig 값에 따른 마스크 사이즈 확인
				GaussianBlur(octave[i].gaussianFliter[j - 1], octave[i].gaussianFliter[j], Size(size, size), currentSig, currentSig);//이전 가우시안 필터를 기반으로 한 가우시안 블러.
				beforeSig = sig;//Sig 기록
			}
		}
	}

	for (int i = 0; i < Octave; ++i)//모든 옥타브에서 처리.
	{
		for (int j = 0; j < Gaussian - 1; ++j)//DoG 채널의 개수는 Gaussian - 1
		{
			//octave[i].DoGChannel[j] = octave[i].gaussianFliter[j + 1] - octave[i].gaussianFliter[j];
			absdiff(octave[i].gaussianFliter[j + 1], octave[i].gaussianFliter[j], octave[i].DoGChannel[j]);
		}
	}

	for (int o = 0; o < Octave; ++o)//모든 옥타브에서 실행.
	{
		for (int d = 1; d < Gaussian - 2; ++d)//DoG를 비교할 때, 0번째와 맨 마지막은 사용하지 않는다. DoG개수는 Gaussian-1이므로 Gaussian-2로 설정
		{
			for (int i = 1; i < octave[o].DoGChannel[d].rows - 1; ++i)//첫 열과 끝 열, 첫 행과 끝 행은 기준으로 삼지 않는다. 옥타브마다 rows, cols값이 다르다.
			{
				for (int j = 1; j < octave[o].DoGChannel[d].cols - 1; ++j)
				{
					int neighbor[26];//26개의 이웃값을 삽입할 배열
					int n = 0;//배열 번호.
					int value = octave[o].DoGChannel[d].at<uchar>(i, j);//현재 기준의 값.

					for (int t = -1; t <= 1; ++t)//26개는 각각 채널별로 3x3 공간의 값을 받아온다. 자신을 기준으로 -1부터 +1까지의 값을 받아오므로 다음과 같은 for문으로 대입.
					{
						for (int tt = -1; tt <= 1; ++tt)
						{
							neighbor[n++] = octave[o].DoGChannel[d - 1].at<uchar>(i + t, j + tt);//기준 위치보다 상위 DoG채널의 값 받아오기.
							neighbor[n++] = octave[o].DoGChannel[d + 1].at<uchar>(i + t, j + tt);//기준 위치보다 하위 DoG채널의 값 받아오기.
							if (t != 0 || tt != 0)//기준과 같은 위치의 값은 배열에 넣지 않는다.
							{
								neighbor[n++] = octave[o].DoGChannel[d].at<uchar>(i + t, j + tt);//기준 DoG채널의 값 받아오기
							}
						}
					}

					int max = neighbor[0];//max, min에 초기값으로 첫 neighbor값 삽입.
					int min = neighbor[0];

					for (int x = 1; x < 26; ++x)//min, max 비교
					{
						if (max < neighbor[x])
						{
							max = neighbor[x];
						}
						else if (min > neighbor[x])
						{
							min = neighbor[x];
						}
					}
					if (value > max || value < min)//극정일 경우
					{
						feature_P tmp;
						tmp.x = j * pow(2, o);
						tmp.y = i * pow(2, o);
						tmp.radius = 1.6*pow(2, (o + d) / 3.0);
						vec.push_back(tmp);
						//circle(image, Point(j*pow(2, o), i*pow(2, o)), 1.6*pow(2, (o + d) / 3.0), Scalar(0, 255, 0), 1);//원 그리기. 반지름은 scale 값이며 1.6*2^((Octave num + DoGlayer Num)/3.0)
					}
				}
			}
		}
	}
}

// 15*15 block
void* LBP(Mat& image, int x, int y, int *result)
{
	int cnt = 0;
	int standard = image.at<uchar>(y, x);
	for(int m = y - 7; m <= y + 7; m++)
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
	//저장할 txt 파일
	errno_t err;
	FILE *fp;
	err = fopen_s(&fp, "train_Data.txt", "w");

	// 디렉토리 내 파일 읽어오는 path
	string path = "C:\\Users\\user\\Desktop\\train\\*.jpg";
	
	struct _finddata_t fd;
	intptr_t handle;
	if ((handle = _findfirst(path.c_str(), &fd)) == -1L)
		cout << "No file in directory!" << endl;
	do
	{
		path = "C:\\Users\\user\\Desktop\\train";
		string fullFileName = path + "\\" + fd.name;
		cout << fullFileName << endl;
		//이미지 흑백으로 만듦
		Mat image0 = imread(fullFileName.c_str());
		if (image0.empty()) //제대로 안읽히면 예외처리
			continue;
		Mat image;
		cvtColor(image0, image, CV_BGR2GRAY);
		const int width = image.cols;
		const int height = image.rows;

		// feature point를 저장할 vec 선언
		vector<feature_P> vec;
		// SIFT 함수를 이용해 feature Point를 추출해준다.
		SIFT(image, vec);

		int count = 0;
		// 클래스가 1인 feature vector를 저장
		for (int i = 0; i < vec.size(); i++) {
			if (vec.at(i).x > 8 && vec.at(i).x < width - 8 && vec.at(i).y > 8 && vec.at(i).y < height - 8)
			{
				fprintf(fp, "+1 ");
				int result[224];
				LBP(image, vec.at(i).x, vec.at(i).y, result);
				for (int n = 0; n < 224; n++)
					fprintf(fp, "%d:%d ", n + 1, result[n]);
				fprintf(fp, "\n");
				count++;
			}
		}

		// 더미 데이터 생성
		int i = 0;
		while(i <= count)
		{
			fprintf(fp, "-1 ");
			// 랜덤 x,y좌표를 생성
			int x = rand() % (width - 16) + 8;
			int y = rand() % (height - 16) + 8;

			for (int it = 0; it < vec.size(); it++)
				if (vec.at(it).x == x && vec.at(it).y == y)
					continue;

			int fake[224];
			LBP(image, x, y, fake);
			for (int n = 0; n < 224; n++)
				fprintf(fp, "%d:%d ", n + 1, fake[n]);
			fprintf(fp, "\n");
			i++;
		}
	} while (_findnext(handle, &fd) == 0);
	_findclose(handle);

	//파일 닫기
	fclose(fp);
	return 0;
}


