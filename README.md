# melanoma-detection
More than 9,500 people in the United States alone are diagnosed with skin cancer every day. 
According to the Skin Cancer Foundation, the number of melanoma deaths is expected to increase by 4.8 percent in 2021. 
In addition, at least one in five Americans will develop skin cancer by the age of 70. These statistics show that we need a more efficient way for skin cancer diagnosis. Machine learning algorthims and models can help in early skin cancer detection and can provide diagnoses to patients in remote locations. They can also help dermatologists verify their observations to make more accurate conclusions. Overall, machine learning and AI can be a very powerful tools in cancer prevention and aid. 

## Download Data
Data used in for this application can be found on [Kaggle](https://www.kaggle.com/c/siim-isic-melanoma-classification/overview).
You can also find the exact files & model used [here](https://drive.google.com/drive/folders/1kGOl3coyEj1pkcwGfC0FBbxXzTafuR3h?usp=sharing).

## Model
The model classifies skin lesion images as benign (without melanoma) or malignant (containing melanoma). 
Since this task involves images, we use [convolutional neural networks](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53). 
Convolutional neural networks (CNNs) examine images and find patterns to differentiate images of various classes. 
Pixels in images are translated into numerical values that act as the pixel intensity values. 
CNNs involve kernels which act as a sliding window, or matrix that moves over the image. 
Kernels perform dot products on the various areas of the image and gets a matrix output with these dot products. 

The specific type of CNN architecture I used for this classification task is ResNet50. [ResNet](https://en.wikipedia.org/wiki/Residual_neural_network) is a very 
popular architecture that uses skip connections to jump over 
certain layers in the network. It aids in preventing the infamous [Vanishing Gradient](https://en.wikipedia.org/wiki/Vanishing_gradient_problem) problem through these skip connections.

With the use of the [fastai](https://www.fast.ai/) deep learning library, I was able to easily implement a pre-trained version of ResNet50.
Most of the code I used for reference comes from [fastbook](https://github.com/fastai/fastbook), available on GitHub.

During training, the following were passed into the cnn_learner():
- metrics = [accuracy,
           error_rate, 
           RocAucBinary(), Recall(), Precision()]
- loss function : CrossEntropyLossFlat()
- learning rate: lr_max=slice(1e-6,1e-4) -> the optimum learning rate
- 4 epochs

Before unfreezing:

<img width="727" alt="Screen Shot 2021-07-16 at 6 01 39 PM" src="https://user-images.githubusercontent.com/75640165/126020763-180d9ff7-1db3-4275-a786-d3a916983ce8.png">


After unfreezing: 

<img width="718" alt="Screen Shot 2021-07-16 at 6 02 05 PM" src="https://user-images.githubusercontent.com/75640165/126020780-60def225-aca3-4b97-9808-488119d8ef0e.png">

(Unfreezing allows the weights of the model to get updated.)

The results from the initial model were not ideal, especially considering how crucial it is to predict results for this task correctly.

## Model Improvements

Our main goal is to improve recall while also maintaining a balance between precision and accuracy. We can do this by lowering the threshold. After testing multiple 
threshold values, I got the following results:


| Threshold Value| Recall|Precision| Accuracy|
|--|--|--|--|
| 0.1 | 1.0 | ~ 0.41|  ~ 0.68|
| 0.2 | 1.0 |  0.46|  ~ 0.74|
| 0.3 | 1.0 | ~ 0.51|  ~ 0.78|
| 0.4 | 1.0 | ~ 0.55|  ~ 0.81|
| 0.5 | ~ 0.96 | ~ 0.58|  ~ 0.83|
| 0.6 | ~ 0.87 | ~ 0.61|  ~ 0.84|
| 0.7 | ~ 0.74 | ~ 0.65|  ~ 0.85|
| 0.8 | ~ 0.74 | ~ 0.65|  ~ 0.85|
| 0.9 | ~ 0.70 | ~ 0.70|  ~ 0.86|

The best threshold values are between 0.4 - 0.5, as we want to maximize recall to prevent false negatives.

<img width="618" alt="Screen Shot 2021-07-16 at 6 14 19 PM" src="https://user-images.githubusercontent.com/75640165/126021050-59531a53-da2f-42a1-aa9c-29137edf8128.png">

## Use

An interactive application is available when opening the Google Colab notebook. It is located in the [Melanoma_Detection](https://github.com/anyaiyer/melanoma-detection/blob/main/Melanoma_Detection.ipynb) file.

You can view the exploratory data analysis and graphs in the [Skin_Cancer_EDA](https://github.com/anyaiyer/melanoma-detection/blob/main/Skin_Cancer_EDA.ipynb) file

You can download all the data through this [Google Drive shared folder](https://drive.google.com/drive/folders/1cuSLtigqqIx7_3OynNPADYU5xN7M1DNw?usp=sharing).

The folder contains:
- the loaded model (model_1.pth)
- train.csv (contains all labels and information about training data)
- test.csv
- train6 (all images used in training and validation)
- test_ (images that can be used on the interactive application; model has not seen this data)

Use of matplotlib, pandas, numpy, seaborn, ipywidgets, and fastai is needed. Update to Python 3.8 is recommended.



