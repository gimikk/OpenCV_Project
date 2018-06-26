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

struct OctaveGroup//�� ��Ÿ�� �׷�� ������ �ִ� ���� ����þ� ����, DoG ä��, Ű����Ʈ
{
	Mat gaussianFliter[Gaussian];//�� ��Ÿ�� �� ������ �ִ� Gaussian Filter
	Mat DoGChannel[Gaussian - 1];//�� ��Ÿ�� �� ������ �ִ� DoG Channel�� Gaussian-1��.
};

// feature_P�� ������ ����ü, Point�� radius�� ������
struct feature_P
{
	int x;
	int y;
	int radius;
};

int CheckSigSize(double sig)//Sig�� �̿��Ͽ� ���� ����ؾ� Size�� ����ؾ� �Ǵµ� ����ϰ� ������ ���� �Լ��� ����.
{
	int size;//����� ����ũ ������

	size = ceil(6 * sig);//6*sig�� �ø����� ����ũ ������� �ӽ� ����

	if (size % 2 == 0)//���� ¦���� ���
		size++;//1�� ����

	return size;//����ũ ������ return;
}


void SIFT(Mat& gray,vector<feature_P>& vec)
{
	OctaveGroup octave[Octave];

	double beforeSig; //���� Sig���� �񱳸� �̿��Ͽ� sqrt(sig_n+1^2 - sig_n^2)�� ���ϱ� ���� �뵵.

	for (int i = 0; i < Octave; ++i)
	{
		for (int j = 0; j < Gaussian; ++j)
		{
			double sig;
			int size;

			if (j == 0)//ù ����þ� ������ ���.
			{
				if (i != 0)//0��Ÿ�갡 �ƴ� ���
				{
					pyrDown(octave[i - 1].gaussianFliter[3], octave[i].gaussianFliter[j],
						Size(octave[i - 1].gaussianFliter[3].cols / 2, octave[i - 1].gaussianFliter[3].rows / 2));//���� ��Ÿ���� 4��° �̹����� ����Ͽ� �����´�.
					beforeSig = 1.6;//Sig�� 1.6�̾��� ������ ����.
				}
				else//0 ��Ÿ���� ���, ���� �̹����� sig 1.6 ���� �̿��Ͽ� ���͸�.
				{
					sig = 1.6;//ù��° �̹����� Sig�� 1.6.
					size = CheckSigSize(sig);//Sig ���� ���� ����ũ ������ Ȯ��
					GaussianBlur(gray, octave[i].gaussianFliter[j], Size(size, size), sig, sig);//����þ� ��.
					beforeSig = sig;//Sig ���
				}
			}
			else
			{
				sig = beforeSig * pow(2.0, 1.0 / 3.0);//Sig_n+1 = Sig_n * k.   k = 2^(1/3)�� ���� �ڵ�
				double currentSig = sqrt(pow(sig, 2) - pow(beforeSig, 2));//sqrt(Sig_n+1^2 - Sig_n^2)�� ���� �ڵ�.
				size = CheckSigSize(currentSig);//Sig ���� ���� ����ũ ������ Ȯ��
				GaussianBlur(octave[i].gaussianFliter[j - 1], octave[i].gaussianFliter[j], Size(size, size), currentSig, currentSig);//���� ����þ� ���͸� ������� �� ����þ� ��.
				beforeSig = sig;//Sig ���
			}
		}
	}

	for (int i = 0; i < Octave; ++i)//��� ��Ÿ�꿡�� ó��.
	{
		for (int j = 0; j < Gaussian - 1; ++j)//DoG ä���� ������ Gaussian - 1
		{
			//octave[i].DoGChannel[j] = octave[i].gaussianFliter[j + 1] - octave[i].gaussianFliter[j];
			absdiff(octave[i].gaussianFliter[j + 1], octave[i].gaussianFliter[j], octave[i].DoGChannel[j]);
		}
	}

	for (int o = 0; o < Octave; ++o)//��� ��Ÿ�꿡�� ����.
	{
		for (int d = 1; d < Gaussian - 2; ++d)//DoG�� ���� ��, 0��°�� �� �������� ������� �ʴ´�. DoG������ Gaussian-1�̹Ƿ� Gaussian-2�� ����
		{
			for (int i = 1; i < octave[o].DoGChannel[d].rows - 1; ++i)//ù ���� �� ��, ù ��� �� ���� �������� ���� �ʴ´�. ��Ÿ�긶�� rows, cols���� �ٸ���.
			{
				for (int j = 1; j < octave[o].DoGChannel[d].cols - 1; ++j)
				{
					int neighbor[26];//26���� �̿����� ������ �迭
					int n = 0;//�迭 ��ȣ.
					int value = octave[o].DoGChannel[d].at<uchar>(i, j);//���� ������ ��.

					for (int t = -1; t <= 1; ++t)//26���� ���� ä�κ��� 3x3 ������ ���� �޾ƿ´�. �ڽ��� �������� -1���� +1������ ���� �޾ƿ��Ƿ� ������ ���� for������ ����.
					{
						for (int tt = -1; tt <= 1; ++tt)
						{
							neighbor[n++] = octave[o].DoGChannel[d - 1].at<uchar>(i + t, j + tt);//���� ��ġ���� ���� DoGä���� �� �޾ƿ���.
							neighbor[n++] = octave[o].DoGChannel[d + 1].at<uchar>(i + t, j + tt);//���� ��ġ���� ���� DoGä���� �� �޾ƿ���.
							if (t != 0 || tt != 0)//���ذ� ���� ��ġ�� ���� �迭�� ���� �ʴ´�.
							{
								neighbor[n++] = octave[o].DoGChannel[d].at<uchar>(i + t, j + tt);//���� DoGä���� �� �޾ƿ���
							}
						}
					}

					int max = neighbor[0];//max, min�� �ʱⰪ���� ù neighbor�� ����.
					int min = neighbor[0];

					for (int x = 1; x < 26; ++x)//min, max ��
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
					if (value > max || value < min)//������ ���
					{
						feature_P tmp;
						tmp.x = j * pow(2, o);
						tmp.y = i * pow(2, o);
						tmp.radius = 1.6*pow(2, (o + d) / 3.0);
						vec.push_back(tmp);
						//circle(image, Point(j*pow(2, o), i*pow(2, o)), 1.6*pow(2, (o + d) / 3.0), Scalar(0, 255, 0), 1);//�� �׸���. �������� scale ���̸� 1.6*2^((Octave num + DoGlayer Num)/3.0)
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
			// �߰����� ĵ��
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
	//������ txt ����
	errno_t err;
	FILE *fp;
	err = fopen_s(&fp, "train_Data.txt", "w");

	// ���丮 �� ���� �о���� path
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
		//�̹��� ������� ����
		Mat image0 = imread(fullFileName.c_str());
		if (image0.empty()) //����� �������� ����ó��
			continue;
		Mat image;
		cvtColor(image0, image, CV_BGR2GRAY);
		const int width = image.cols;
		const int height = image.rows;

		// feature point�� ������ vec ����
		vector<feature_P> vec;
		// SIFT �Լ��� �̿��� feature Point�� �������ش�.
		SIFT(image, vec);

		int count = 0;
		// Ŭ������ 1�� feature vector�� ����
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

		// ���� ������ ����
		int i = 0;
		while(i <= count)
		{
			fprintf(fp, "-1 ");
			// ���� x,y��ǥ�� ����
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

	//���� �ݱ�
	fclose(fp);
	return 0;
}


