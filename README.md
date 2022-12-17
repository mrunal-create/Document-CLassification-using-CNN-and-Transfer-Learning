# Document-CLassification-using-CNN-and-Transfer-Learning

Document classification is a task that is performed in many domains on a regular basis,
especially in administration. This also constitutes classifying digital images of various
documents sent via emails. In this project we aim to use convolutional neural networks to help
automate document image classification. As part of our approach we used Convolutional
Models and compared its results with pre-trained CNN models used for Transfer Learning. For
the Convolutional Neural Network Models two variations were implemented - Time Distributed
Layers and Modified Custom Layers (DocNet), whereas for transfer learning VGG16, VGG19
and InceptionV3 were used.Usual document classification models were using Natural language
Processing and Text Classification but we planned to classify the documents using layout and
structure of the image. The evaluation metrics used for comparison and performance evaluation
of the models were accuracy,precision,recall,confusion matrix. Our DocNet CNN was able to
achieve 97% accuracy which was the highest among all the other models we implemented.


The Dataset used for this problem statement is the Ryerson Vision Lab Complex Document
Information Processing (RVL-CDIP) dataset. The original dataset consists of 16 classes which
consist letter,form,email,handwritten, advertisement, scientific report, scientific publication ,
specification ,file folder ,news article ,budget ,invoice ,presentation ,questionnaire ,resume.From this
dataset we decided to build our model on 4 subcategories namely letter, invoice, form, resume.
The dataset has been taken from Kaggle and it consists of Training,Validation and Testing
Folders.


The deep learning models used were Convolutional Neural Networks, pre-trained CNN’s like
RESNET50, VGG16, VGG19, InceptionV3.The deep learning models were trained using
Tensorflow. The project was set up and executed on Google Collab Pro with 14.5 GB of RAM
and 100 units of GPU. Image Processing was carried out using OpenCV and Matplotlib was
used to create graphs.
