# Football Club Logo Classification using Transfer Learning

## Transfer Learning:
Any image classification problem requires the model to learn features of objects in the dataset. If the objects have high level features such as complex shapes, colors, etc. then the model would require lot of data to learn these features. A pretrained model is a model that was previously trained on a large dataset, typically on a large-scale image-classification task. If this
original dataset is general enough, then the spatial hierarchy of features learned by the model can effectively act as a generic model of the visual world, and hence its features can prove useful for our problem, even though the classes in our problem are completely different. This technique is called Transfer Learning. 

## Dataset:
The dataset consists of a total of 300 images. 50 images per class.
The classes are:

1. Barcelona
2. Real Madrid
3. Manchester United
4. Borussia Dortmund
5. Inter Milan
6. Chelsea

Images souce: Google

## Data processing:
The model takes image of a fixed dimension. The image_processing.py file resizes each image to 224X224 dimensions. It converts the dataset to a numpy array and saves it in data.npy file. It also generates a labels.npy file consisting of target labels for each image in the dataset.

## Model:
The model makes use of the MobileNet_v2 architecture trained on the ImageNet dataset. The MobileNet_v2 model is smaller compared to other models. This helps in ease of training and the model can be used in android applications due to its relatively smaller size. The Dense layer of the model is replaced by our custom dense layer to classify 6 classes. The logo_classifier.ipynb generates a model.h5 file which cotains the trained model.

## Training:
Google Colab Notebook was used for training this model. The model was trained for 25 epochs with 20% validation split.

## Prediction:
The predictor.py file takes in images of club logos as input, resizes it to 224X224 to feed it to the trained model and outputs the target logo present in the image. The extra_images folder consists of some additional images for testing the model. 

## Results:
The accuracy of some of the available models for our problem are:

| Model |   Accuracy |
| ----- | -----------|
| VGG16 |           90% |
| NasNetMobile|        90% |
| MobileNet_v2 |       91.67%  |
| InceptionResNet_v2|  91.67%  |
| Inception_v3|        93.33%  |
| Xception|            93.33%  |