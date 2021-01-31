# Mask_Detection_with_Person_Identification
A project for detecting person without mask and identifying that person with time and storing data in .csv file.

---

## Motivation:
* To create a combination of **Object detection model** and **Image classification model**, to detect person with no mask and identifying that person to store the data in a downloadable csv file with UI.

---

## Project:
This Project is a Combination of **Faster-RCNN-Inception-v2-coco** and **Custom Image classification model** for identifying person.

### WebApp:



### How it Works:


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



#### Understanding Faster-RCNN:

#### 



