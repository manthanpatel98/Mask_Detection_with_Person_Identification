# Mask_Detection_with_Person_Identification
A project for detecting person without mask and identifying that person with time and storing data in .csv file.

---

## Motivation:
* To create a combination of **Object detection model** and **Image classification model**, to detect person with no mask and identifying that person to store the data in a downloadable csv file with UI.

---


## WebApp:

<img src="https://github.com/manthanpatel98/Mask_Detection_with_Person_Identification/blob/main/Readme_Img/gif.gif">

**Downloaded CSV file:**

<img src="https://github.com/manthanpatel98/Mask_Detection_with_Person_Identification/blob/main/Readme_Img/Screenshot%20(473).png" width=500>



## Project:
* This Project is a Combination of **Faster-RCNN-Inception-v2-coco** and **Custom Image classification model** for identifying person.
* Dataset used for Mask Detection is on [Kaggle](https://www.kaggle.com/andrewmvd/face-mask-detection) and custom dataset is used for Person Identification.


<img src="https://github.com/manthanpatel98/Mask_Detection_with_Person_Identification/blob/main/Readme_Img/project-gif.gif" width=800>



### Entire Process:
1. Input Image will be passed to **faster-rcnn-inception-v2-coco** model for identifying people **with and without mask**.
2. Images of people without mask will be cropped and passed to **Custom Image Classification Model**.
3. People without mask will be identified with help of Image classification model and the data of identified people (in this case Manthan or Unknown) with the current time will be stored into .csv file.

Following is the flow of project:

<img src="https://github.com/manthanpatel98/Mask_Detection_with_Person_Identification/blob/main/Readme_Img/flow.jpg" width=650>

So, That makes it like below:

<img src="https://github.com/manthanpatel98/Mask_Detection_with_Person_Identification/blob/main/Readme_Img/Slide.JPG" width=700>

---

### Understanding Faster-RCNN:

* Faster RCNN is **attention based** and one of the classic algorithms in Object Detection. Faster RCNN solves the problems of RCNN and Fast RCNN by introducing **RPN** (Region Proposal Network) instead of using hard coded algorithm like **selective search**. 

**Structure:**

<img src="https://github.com/manthanpatel98/Mask_Detection_with_Person_Identification/blob/main/Readme_Img/FRCN_architecture.png" width=500>


1. Conv Layers: This Newtwork is a combination of Conv Layers, activation function and pooling layers which extracts feature maps from the given input image. This feature map will be later shared with RPN.

2. RPN: It is used to generate Region Proposals which uses softmax to classify the candidate box.and use bbox to perform regression correction on the candidate box to obtain the proposals. 

3. RoI pooling: Collect feature maps and proposals, extract the proposal feature map, and send it to the subsequent fully connected layer to determine the target category. 

4. Classification: Uses the proposal feature map to calculate the proposal category, and bbox regression again to obtain a more accurate positioning.

* The classical detection methods are very time-consuming. Faster rcnn can use RPN to automatically generate candidate frames, which greatly improves the speed of generating candidate frames. That is why Faster RCNN is much better than RCNN and Fast RCNN.


### Image Classification model:

* Integreted Custom Image Classification model is trained on **2** classes: **Manthan** and **Unknown**. (Model with More num of classes can be trained and used here.)

* Below is the model structure created with **[Netron]**(https://netron.app/).

<img src="https://github.com/manthanpatel98/Mask_Detection_with_Person_Identification/blob/main/Readme_Img/best_model.h5.png">

---

## Files:
* Here, folder [faster_rcnn_inception_v2_coco](https://github.com/manthanpatel98/Mask_Detection_with_Person_Identification/tree/main/faster_rcnn_inception_v2_coco) has trained inference graph of object detection model.
* Trained Image Classification model is [best_model.h5](https://github.com/manthanpatel98/Mask_Detection_with_Person_Identification/blob/main/best_model.h5).

* Main file for detection and classification is located in [mask_detector_and_person_classifier](https://github.com/manthanpatel98/Mask_Detection_with_Person_Identification/tree/main/mask_detector_and_person_classifier) folder.

* [mask_detector_and_person_classifier](https://github.com/manthanpatel98/Mask_Detection_with_Person_Identification/tree/main/mask_detector_and_person_classifier) has paths to frozen_inference_graph.pb and labelmap.pbtxt which should be changed if someone trains their own models.

## Improvements:
* As an improvement, we can use much faster models **SSD** and **YOLO** as Object Detection Model.
* Same way, Image Classification model can be replaced as per need.
* **An improved system can be useful for detecting people without mask which can be useful to stop the spread of virus by making more strict rules in premises.**




