# Object Detection & Segmentation

**서론**

Computer Vision은 visual data에 대한 연구가 이뤄지는 분야입니다. 이미지 처리, 로보틱스, 의료, 인지과학, 신경학 등 시각 데이터를 처리하는 다양한 분야에서 연구되고 있습니다. 이번에는 Computer vision에서 다루는 Object detection과 Segmentation에 대해 알아보고자 합니다.

Object Detection은 아래 그림과 같이 이미지에서 물체의 위치와 해당 물체가 무엇인지 탐지하는 작업입니다. 따라서 물체를 분류하는 Classification 문제와 물체 위치를 찾는 Localization 두 가지 문제를 해결한다고 할 수 있습니다. 이 두 문제를 순차적으로 해결하는 인공지능 모델을 2-stage detector라고 하며 동시에 해결하는 모델을 1-stage detector라고 합니다. 먼저 2-stage Detector인 R-CNN을 소개한 후 1-stage detector인 YOLO모델을 소개하고자 합니다.

<figure><img src="../.gitbook/assets/그림 1 Object Detection 예시.png" alt=""><figcaption><p>그림 1. Object Detection 예시</p></figcaption></figure>

Image Segmentation은 아래 그림과 같이 이미지를 영역으로 구분 지어 물체를 탐지하는 작업입니다. Segmentation은 색깔, 명도, 채도와 같은 이미지의 속성을 이용하는 Region-based segmentation, 물체 경계를 이용하는 Edge-based segmentation, 그리고 각 픽셀에 대해 labeling을 하는 Semantic segmentation 등으로 다양하게 풀이할 수 있습니다. 여기서는 Semantic segmentation의 방식을 활용한 FCN, U-Net 모델을 소개하고자 합니다.

<figure><img src="../.gitbook/assets/그림 2 Segmentation 예시.png" alt=""><figcaption><p>그림 2. Object Segmentation 예시</p></figcaption></figure>

## Object Detection

### **R-CNN**

R-CNN은 Region-based Convolutional Neural Network의 약자로 Object Detection을 수행하는 2-Stage detector 모델입니다. 모델의 동작은 크게 Object가 존재하는 영역을 찾는 Region Proposal, 찾은 영역의 특징들을 추출하는 Feature Extraction, 특징들로부터 물체를 분류하는 Classification과 물체의 위치를 찾는 Bounding box regression으로 이루어져 있습니다. 정리하면 아래 그림과 같이 모델이 동작한다고 할 수 있습니다.

<figure><img src="../.gitbook/assets/그림 3 R-CNN 모델.png" alt=""><figcaption><p>그림 3. R-CNN 모델 구조</p></figcaption></figure>

#### **Region Proposal**

Region Proposal은 물체가 있을법한 영역을 제시하는 과정으로 Selective Search 알고리즘에 의해 이뤄집니다. Selective Search 알고리즘은 이미지를 작은 영역들로 많이 나누어 각 영역에서의 색상, 질감, 명암에 따른 값을 설정합니다. 이후 Iteration을 통해 이 값들의 차이에 따라 유사도가 높은 영역들은 합쳐지고 최종적으로 약 2,000개의 영역만을 남겨놓습니다. 이렇게 남겨진 영역들은 우리가 실제로 관심을 가져야 하는 영역이기에 Region of Interest (RoI)라고 부릅니다.

<figure><img src="../.gitbook/assets/그림 4 Selective search.png" alt=""><figcaption><p>그림 4. Selective search with some iterations</p></figcaption></figure>

#### **Feature Extraction**

앞서 추출한 RoI들로부터 우리는 이 영역에 담긴 물체가 어떤 물체인지 알아야 합니다. 따라서 CNN을 이용하여 영역의 특징들을 추출하는 Feature Extraction이 이뤄집니다. 현재 각 영역의 크기가 다르기에 같은 크기의 output을 출력할 수 있도록 RoI들을 모두 동일한 크기로 변형합니다. 이후 Convolution network를 통과시켜 Feature vector를 얻게 됩니다.

#### **Classification**

추출된 Feature vector로부터 우리는 물체를 분류할 수 있습니다. 물체를 분류하는 방법들은 다양하게 존재하지만 R-CNN에서는 Support Vector Machine (SVM)을 사용하였습니다. 이로부터 RoI가 물체인지 아닌지, 그리고 물체라면 어떤 물체인지 분류할 수 있습니다. Classification의 과정을 정리하면 아래 그림과 같습니다.&#x20;

<figure><img src="../.gitbook/assets/그림 5 R-CNN Classification.png" alt=""><figcaption><p>그림 5. R-CNN 모델 구조(2)</p></figcaption></figure>

#### **Bounding Box Regression**

물체가 분류된다면 실제로 해당 물체가 이미지에서 어디에 위치하는지 Box로 표시해야 합니다. RoI로 추출된 물체의 영역은 정확하지 않기 때문에 regression을 활용하여 Bounding Box의 위치를 조정하는 작업이 진행됩니다. 모델로부터 예측된 Box의 위치와 실제 Box의 위치 차이를 줄이는 방향으로 Regression 모델의 학습이 이뤄지며, regression 모델을 바탕으로 Bounding Box의 위치가 조정됩니다.&#x20;

#### 결론

R-CNN은 Object Detection문제에 CNN을 최초로 적용한 모델로 이전의 Object Detection 방법들보다 높은 성능을 가졌습니다. 하지만 RoI마다 CNN을 연산해야하기에 동작이 느리고 RoI개수가 한정되어있으며 학습 또한 Feature Extraction과 SVM, Bounding Box Regression이 개별로 동작한다는 문제가 있습니다.



### **Fast R-CNN**

Fast R-CNN은 모델 네트워크가 입력에서 출력까지 한 번에 학습하고 동작할 수 있도록 설계한 1-stage detector 모델입니다. 하나의 네트워크로 동작하기에 기존 R-CNN의 느린 동작 문제를 해결하였습니다. Fast R-CNN은 R-CNN과 달리 CNN을 통과시킨 뒤 Feature map으로부터 Selective Search 알고리즘을 활용하여 RoI를 추출합니다. 이후 RoI Pooling을 통해 RoI를 고정된 크기로 변환한 후 신경망을 통과하여 Softmax를 통해 물체를 분류하고 Bounding Box Regression을 통해 물체의 Bounding Box 위치를 조정합니다.

<figure><img src="../.gitbook/assets/그림 6 Fast R-CNN 모델구조.png" alt=""><figcaption><p>그림 6. Fast R-CNN 모델 구조(1)</p></figcaption></figure>

#### **RoI Pooling**

Fast R-CNN에서 가장 핵심이 되는 아이디어입니다. R-CNN에서는 CNN의 출력이 Fully Connected Layer에 입력으로 들어가야 하기에 CNN의 입력을 모두 동일하게 맞춰야 했습니다. 따라서 이미지에서 추출한 RoI의 크기를 변형하는 작업이 존재했습니다. 하지만 Fully Connected layer에 입력 크기만 맞춰주면 되기에 CNN의 출력에서 크기를 변형할 수도 있습니다. RoI에 대한 작업이 CNN 이후로 이뤄진다는 것이 Fast R-CNN과 R-CNN의 차이라고 할 수 있습니다.

구체적으로 말하자면 Fast R-CNN은 R-CNN에서 각각의 RoI에 대해 CNN을 통과시키는 것이 아닌 전체 이미지를 통과시킨 후 출력된 Feature map에 RoI를 내적함으로써 RoI를 CNN에 통과시킨 효과를 얻었습니다. 이는 R-CNN에서 약 2,000개의 RoI를 각각 CNN연산을 하였던 것을 1번의 CNN연산으로 줄여주기에 연산 속도에서 큰 이득을 볼 수 있었습니다.

다음으로 Fully Connected Layer에 입력하기 위해 출력값을동일한 크기의 vector로 변형하는 작업을 진행합니다. 아래 그림과 같이 Feature map에 검은색 hxw의 크기만큼 RoI가 투영되었다면 이를 세로 H칸, 가로 W칸의 영역으로 분리하기 위해 (h/H)x(w/W)의 크기만큼 Grid를 생성합니다. 마지막으로 각 칸에 대해 최곳값만 추출한다면 HxW 크기의 고정된 크기 벡터가 생성됩니다.

<figure><img src="../.gitbook/assets/그림 7 RoI Pooling.png" alt=""><figcaption><p>그림 7. RoI Pooling</p></figcaption></figure>

#### **Multi-task loss**

Fast R-CNN은 End to end로 한 번에 전체 네트워크를 학습하기 위해 Multi-task loss를 아래 식과 같이 사용하였습니다. 이로써 Clasification과 Bounding Box Regression에 대해 하나의 Loss로 동시에 학습을 진행할 수 있었습니다. 이외에도 Fast R-CNN은 Truncated SVM을 사용하여 Fully Connected Layer의 파라미터 수를 줄이고 동작 속도를 30%이상 향상할 수 있었습니다.

$$L(p,u,t^u,v)=L_{cls}(p,u)+\lambda[u\geq1]L_{loc}(t^u,v)$$

* $$p=(p_0, ...,p_k) : (k+1)개의\;class\;score$$
* $$u: ground\;truth\;class\;score$$
* $$t^u=(t_x^u,t_y^u,t_w^u,t_h^u) : 예측한\;Bounding\;Box \;좌표를\;조정하는\;값$$
* $$v=(v_x,v_y,v_w,v_h) :실제\;Bounding\;Box의\;좌표값$$
* $$L_{cls}(p,u)=-log(p_u):classification\;loss$$
*   $$L_{loc}(t^u,v)=\sum_{i\in(x,y,w,h)}smooth_{L_1}(t^u_i-v_i):Regression\;loss(Smooth\;L1\;loss)$$



#### **결론**

정리하면 아래 그림과 같이 모델이 동작합니다. CNN으로 Feature map을 추출한 뒤 RoI pooling을 진행하고 Classification과 Bounding Box Regression을 한 번에 진행합니다. Fast R-CNN은 R-CNN보다 높은 정확도와 빠른 속도를 보이지만 여전히 Selective Search를 사용하여 RoI를 생성하기에 속도 저하가 발생합니다. 따라서 Selective Search 알고리즘을 CNN의 내부에서 GPU연산으로 진행하는 아이디어를 제시한 것이 Faster R-CNN입니다. 이는 Ybigta의 Data Science팀에서 더 자세하게 배우실 수 있습니다.

<figure><img src="../.gitbook/assets/그림 8 Fast R CNN 구조.png" alt=""><figcaption><p>그림 8. Fast R-CNN 모델 구조(2)</p></figcaption></figure>



### **YOLO**

YOLO는 You Only Look Once의 약자로 2016년에 제시된 1-stage Detector 모델입니다. 물체의 Classification과 Localization을 동시에 진행하여 빠른 Detection을 자랑합니다. YOLO는 v1, v2, v3 등으로 계속 발전하였고 최근 2023년에는 YOLO v8까지 등장하였습니다.

YOLO는 Classification과 Localization을 하나의 Regression문제로 정의합니다. 이로써 이미지 전체를 학습하여 곧바로 Detection이 진행되며 1초에 45프레임을 처리할 수 있을 정도로 빠른 성능을 보입니다. Fast R-CNN이 이미지 하나에 약 2초의 시간이 걸렸던 것을 생각하면 YOLO는 동영상에서 실시간으로 Detection이 가능할 정도의 성능을 보입니다.

#### **Unified Detection**

YOLO는 이미지 전체를 활용하여 물체를 검출하고 end to end로 학습하는 특징이 있습니다. 이를 위해 먼저 이미지 전체를 SxS 크기의 Grid로 나누는 작업을 진행합니다. 이렇게 Grid로 나눈 영역들을 Cell이라고 하며 각각의 Cell에 대해 Detection이 이뤄집니다. 정확히는 각 Cell은 B개의 Bounding box와 Confidence Score를 예측합니다. Confidence score는 Bounding box의 신뢰성을 평가하는 지표이며 물체가 있을 확률에 실제 Bounding box와 예측 Bounding Box의 비율을 곱하여 계산됩니다. 정리하면 Bounding box는 Cell에서 상대적인 Box의 중심점(x,y), 너비와 높이(w,h), confidence score로 이뤄집니다.

각 Cell은 Bounding box의 예측뿐만 아니라 물체의 Classification까지 함께 이뤄집니다. 물체가 현재 Cell에 존재한다는 조건하에 그 물체의 Class에 대한 확률 Conditional Class Probability를 예측합니다. 하나의 Cell에는 하나의 Class에 대한 확률값만 계산하기에 하나의 Cell에 여러 개의 물체가 있다면 이 중 가장 확률값이 높은 Class를 예측할 수 있습니다.

최종적으로 Cell에서 아래 그림과 같이 물체에 대한 Classification과 Bounding Box 예측을 함께 진행함으로써 한 번에 Object Detection이 이뤄질 수 있습니다.

<figure><img src="../.gitbook/assets/그림 9 YOLO 설명.png" alt=""><figcaption><p>그림 9. YOLO 모델 동작</p></figcaption></figure>

#### **Network Design**

YOLO는 이미지 분류에 사용되었던 GoogLeNet의 구조를 기반으로 하며 두 개의 FC layer가 뒤에 추가되었습니다.&#x20;

<figure><img src="../.gitbook/assets/그림 10 YOLO 구조.png" alt=""><figcaption><p>그림 10. YOLO 모델 구조</p></figcaption></figure>

훈련에 사용된 Loss function은 다음과 같습니다.&#x20;

$$L =\lambda_{coord}\sum\limits_{i=0}^{S^2}\sum\limits_{j=0}^{B}\mathbb{1}_{ij}^{obj}[(x_i-\hat{x_i})^2+(y_i-\hat{y_i})^2]\\ +\lambda_{coord}\sum\limits_{i=0}^{S^2}\sum\limits_{j=0}^{B}\mathbb{1}_{ij}^{obj}[(\sqrt{w_i}-\sqrt{\hat{w_i}})^2+(\sqrt{h_i}-\sqrt{\hat{h_i}})^2]\\   +\sum\limits_{i=0}^{S^2}\sum\limits_{j=0}^{B}\mathbb{1}_{ij}^{obj}(C_i-\hat{C_i})^2\\ +\lambda_{noobj}\sum\limits_{i=0}^{S^2}\sum\limits_{j=0}^{B}\mathbb{1}_{ij}^{noobj}(C_i-\hat{C_i})^2\\ +\sum\limits_{i=0}^{S^2}\mathbb{1}_{ij}^{obj}\sum\limits_{c \in classes}^{}(p_i(c)-\hat{p_i}(c))^2$$



기본적으로 Sum Squared Error를 Loss로 계산하였으며 각각 x와 y의 loss, w와 h의 loss, 물체가 존재하는 Cell의 confidence score의 loss, 물체가존재하지 않는 Cell의 confidence score의 loss, 물체가 존재하는 cell의 conditional class probability의 loss를 모두 더한 것입니다.

#### **결론**

YOLO는 Detection에 있어 굉장히 빠른 성능을 보이지만 정확도는 이전 Fast R-CNN에 비해 부족하였습니다. 또한 하나의 Cell에서는 하나의 물체만 탐지할 수 있기에 하나의 Cell에 여러 개의 물체가 존재한다면 제대로 동작할 수 없습니다. 이러한 문제들은 이후 YOLO v2, v3을 통해 점차 개선되었습니다.

#### ****

## **Object Segmentation**

### **FCN**

Semantic segmentation은 이미지의 픽셀을 하나하나 분류하는 방식입니다. 기본적으로 Sliding window방식을 생각할 수 있지만 정확도와 속도가 떨어지고 굉장히 비효율적입니다. 따라서 CNN을 통해 이미지의 특징들을 활용하며 한 번에 Segmentation이 이뤄질 수 있도록 한 모델구조 중 FCN을 소개하고자 합니다.&#x20;

FCN은 Fully Convolutional Networks의 약자로 아래와 같이 모든 Layer가 convolution layer로 이뤄져 있습니다. 크게 Encoder와 Decoder로 네트워크 구조가 나뉘며 Encoder에서는 Down-sampling을 통해 이미지로부터 고차원의 특징들을 추출합니다. Down-sampling한 이미지로부터 1x1 Convolution을 통해 Segmentation이 이뤄진 Heat map을 생성한 뒤 Up-sampling 과정으로 원본 이미지와 동일한 크기로 복원합니다. 여기서 중요한 점은 이전 Classification에서 사용된 Fully Connected Layer가 1x1 Convolution Layer로 대체되었다는 점입니다. Fully Connected layer와 1x1 convolution layer 모두 행렬의 내적연산이기에 수학적으로 계산이 동일하다는 점에서 대체할 수 있습니다. 따라서 1x1 Convolution을 통해 이미지의 크기에 상관없이 모델이 동작할 수 있습니다. Loss function은 모든 픽셀에서의 Cross Entropy를 구해 더한 값으로 계산됩니다.

<figure><img src="../.gitbook/assets/그림 11 FCN 구조.png" alt=""><figcaption><p>그림 11. FCN 모델 구조</p></figcaption></figure>

하지만 Down-sampling과 Up-sampling만으로는 정확한 Segmentation이 진행되지 않습니다. Down-sampling하여 생성된 Feature map은 이미지의 위치정보를 대략적으로만 가지고 있으며 이를 Up-sampling할 경우 기존 이미지보다 해상도가 낮아지게 됩니다. 그렇다고 Down-sampling 과정을 생략할 경우 이미지의 모든 픽셀에 대해 Classification을 해야 하므로 연산량이 극도로 증가합니다.

#### **Skip architecture**

이러한 문제를 해결하기 위해 Skip architecture라는 기법을 추가하였습니다. ResNet의 구조와 같이 원본에 가까운 이미지를 Up-sampling과정에 포함해 최대한 위치 정보의 손실을 막자는 것이 아이디어의 핵심입니다. 아래 그림과 같이 Up-sampling의 Feature map에 Down-sampling의 Feature map을 합연산을 하여 Up-sampling을진행하는 구조입니다. 이로써 Segmentation의 결과가 더욱 정교해지는 것을 확인할 수 있었습니다.

<figure><img src="../.gitbook/assets/그림 12 FCN skip connection.png" alt=""><figcaption><p>그림 12. FCN의 Skip connection 구조 및 결과 비교</p></figcaption></figure>

### **U-Net**

FCN은 Skip connection을 통해 Up-sampling의 Feature map과 이전 Feature map을 합연산합니다. 하지만 덧셈을 통해 우리는 쉽게 과거의 정보를 알 수 없습니다. 예를 들어 덧셈으로 7이 나왔다고 한다면 이는 1과 6의 합인지, 2와 5의 합인지 모델은 알 수 없습니다. 따라서 이전 Feature map의 정보들을 더욱 적극적으로 활용할 수 있도록 Skip connection에서 합연산이 아닌 그대로 Feature map 뒤에 이어 붙이는 Concatenation연산을 한 것이 U-Net입니다.

U-Net은 아래 그림과 같이 FCN의 구조를 바탕으로 Skip connection을 Feature map과 합연산이 아닌 Channel을 연결 지어 연장하는 방식으로 변경한 모델입니다. 모델의 구조가 U자형으로 생겼기에 U-Net이라는 이름을 가지며 Biomedical image에서 꽤 높은 성능을 보입니다. 평가지표를 실제 구분과 예측 구분의 비율로 하였을 때 PhC-U373 데이터 셋에서 92%라는 높은 성능을 보였습니다.

<figure><img src="../.gitbook/assets/그림 13 U-Net 모델구조.png" alt=""><figcaption><p>그림 13. U-Net 모델 구조</p></figcaption></figure>



### 그림출처

\[그림 1, 5] Ross Girshick, Jeff Donahue, Trevor Darrell, and Jitendra Malik, "Rich feature hierarchies for accurate object detection and semantic segmentation", arXiv:1311.2524, 2014.

\[그림 2] Hengshuang Zhao, Jianping Shi, Xiaojuan Qi, Xiaogang Wang, and Jiaya Jia, "Pyramid Scene Parsing Network", arXiv:1612.01105, 2017.

\[그림 3] Ross Girshick, Jeff Donahue, Trevor Darrell, and Jitendra Malik, "Rich feature hierarchies for accurate object detection and semantic segmentation", arXiv:1311.2524, 2014. Reproduced at stanford cs231n lecture 11.[ ](http://cs231n.stanford.edu/slides/2017/cs231n\_2017\_lecture11.pdf)[source](http://cs231n.stanford.edu/slides/2017/cs231n\_2017\_lecture11.pdf)

\[그림 4] J. Uijlings, K. van de Sande, T. Gevers, and A. Smeulders. Selective search for object recognition. IJCV, 2013.

\[그림 6] Ross Girshick, “Fast R-CNN”, ICCV 2015. Reproduced at stanford cs231n lecture 11. [source](http://cs231n.stanford.edu/slides/2017/cs231n\_2017\_lecture11.pdf)

\[그림 7] Duy Anh Tran, Pascal Fischer, Alen Smajic,  and Yujin So, "Real-time Object Detection for Autonomous Driving using Deep Learning", Goethe University Frankfurt, 2021.

\[그림 8] Ross Girshick, “Fast R-CNN”, ICCV 2015.

\[그림 9, 10] Joseph Redmon, Santosh Divvala, Ross Girshick, and Ali Farhadi, "You Only Look Once: Unified, Real-Time Object Detection", arXiv:1506.02640, 2016.

\[그림 11] Hadar Gorodissky, Daniel Harari, and Shimon Ullman, "Large Field and High Resolution: Detecting Needle in Haystack", arXiv:1804.03576, 2018.

\[그림 12] Jonathan Long, Evan Shelhamer, and Trevor Darrell, "Fully Convolutional Networks for Semantic Segmentation", arXiv:1411.4038, 2015.

\[그림 13] Olaf Ronneberger, Philipp Fischer, and homas Brox, "U-Net: Convolutional Networks for Biomedical Image Segmentation", arXiv:1505.04597, 2015.

