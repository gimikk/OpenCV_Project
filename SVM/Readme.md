
### SVM

- LBP로 이미지의 feature를 뽑고 이를 SVM을 통해 특징점인지 아닌지 학습시켜 특징점을 검출해내는 프로젝트입니다. 
- 기존 SIFT에 비해서 70%정도의 성능밖에 나오지 않지만 SVM의 전체적인 work flow에 대해 알 수 있는 프로젝트입니다.

# SVM

###### - Work Flow
###### 1.  DB 추출
SVM을 학습시키기 위한 DB를 만들기 위해 BSD500 이미지를 사용하였다. C++ 코드로 폴더 내 모든 이미지 파일을 읽어오고, 읽어온 이미지에서 SIFT를 이용해 Feature Point를 추출해 Vector에 좌표를 저장한다.  저장된 좌표의 LBP 벡터를 계산한다. LBP 벡터 추출에는 15 x 15 블록을 사용하였으며, 중간 값과 비교해 작으면 0, 크면 1을 부여하는 224차원의 벡터를 만들어낸다. SIFT Featrue Point의 벡터는 클래스 1(특징점)으로 분류시켜 train_Data.txt라는 파일에 저장된다. 저장 방식은“1 1:LBP[0] 2:LBP[1] 3:LBP[2] .... 224:LBP[223]”이다. 여기서 SVM 이진 분류기는 1클래스와 -1 클래스 두 개로 나뉘어 지는데, 1클래스는 Feature Point 클래스이고, -1은 Feature Point가 아닌 클래스이다. 이때 학습을 위해서는 -1 클래스 또한 만들어야 하는데, Feature Point가 아닌 좌표를 임의로 생성하며, 그 개수는 Feature Point의 개수와 같아야 한다.(같지 않으면 분류가 제대로 되지 않음을 확인했음)

###### 2. SVM Train(svm_learn.exe)
SVM은 SVMlight를 사용하였으며, exe 파일로 되어있어, Data_set만 잘 만들어 주면 알아서 학습을 시켜준다.
SVMlight가 저장된 폴더에 들어가 cmd 상에서 svmlight.exe train_Data.txt명령어를 쳐주면, 자동으로 학습이 시작된다. 이후 SVM_model이라는 파일이 학습되어 나온다.

![](https://github.com/gimikk/OpenCV_Project/blob/master/SVM/image/svm_light.PNG)

###### 3. Test(test_extractor.cpp, svm_classify.exe, test_rendering.cpp)
Test는 BSD test 데이터셋 중 한 장을 골랐으며, test 이미지에서 모든 픽셀의 LBP 벡터를 구한다. 구해진 LBP 벡터는 test.txt 파일로 저장된다.
저장된 LBP 벡터는 학습된 SVM을 통과해 해당 값이 Feature Point인지 아닌지 판별하는 정확도 값을 추출해 낸다. 
추출된 정확도 값을 읽어 들여, test 이미지 상에서 렌더링 해준다. 여기서 정확도 값이 1일시 클래스 1, Feature Point라는 소리임으로 test 이미지에 렌더링 해준다.

- work flow

![](https://github.com/gimikk/OpenCV_Project/blob/master/SVM/image/work_flow.PNG)

- origin

![](https://github.com/gimikk/OpenCV_Project/blob/master/SVM/image/1.%20%EC%9B%90%EB%B3%B8.PNG)

- SIFT

![](https://github.com/gimikk/OpenCV_Project/blob/master/SVM/image/1.%20SIFT.PNG)

- LBP_SVM

![](https://github.com/gimikk/OpenCV_Project/blob/master/SVM/image/1.%20LBP_SVM.PNG)

- origin

![](https://github.com/gimikk/OpenCV_Project/blob/master/SVM/image/2.%20%EC%9B%90%EB%B3%B8.PNG)

- SIFT

![](https://github.com/gimikk/OpenCV_Project/blob/master/SVM/image/2.%20SIFT.PNG)

- LBP_SVM

![](https://github.com/gimikk/OpenCV_Project/blob/master/SVM/image/2.%20LBP_SVM.PNG)



