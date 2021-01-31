import os
import sys
import traceback
import cv2
import numpy as np
import datetime
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np


class Mask_Person:
    def __init__(self, imagePath):
        # This is needed since the notebook is stored in the object_detection folder.
        
        self.PATH_TO_CKPT = r'E:\I-Neuron\My-Projects\Mask-Detection\faster_rcnn_inception_v2_coco\frozen_inference_graph.pb'
        # Path to label map file
        self.PATH_TO_LABELS = r'E:\I-Neuron\My-Projects\Mask-Detection\research\data\labelmap.pbtxt'
        #self.PATH_TO_LABELS = "data/labelmap.pbtxt"
        # Path to images
        self.PATH_TO_IMAGE = imagePath
        print(self.PATH_TO_IMAGE)
        # Number of classes the object detector can identify
        self.NUM_CLASSES = 2

        # Load the label map.
        # Label maps map indices to category names, so that when our convolution
        # network predicts `5`, we know that this corresponds to `king`.
        # Here we use internal utility functions, but anything that returns a
        # dictionary mapping integers to appropriate string labels would be fine
        self.label_map = label_map_util.load_labelmap(self.PATH_TO_LABELS)
        self.categories = label_map_util.convert_label_map_to_categories(self.label_map,
                                                                         max_num_classes=self.NUM_CLASSES,
                                                                         use_display_name=True)
        self.category_index = label_map_util.create_category_index(self.categories)

        # Declaring classes
        self.class_names_mapping = {1: "without_mask",2:"with_mask"}
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        # Define input and output tensors (i.e. data) for the object detection classifier

        # Input tensor is the image
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

        # Output tensors are the detection boxes, scores, and classes
        # Each box represents a part of the image where a particular object was detected
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represents level of confidence for each of the objects.
        # The score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')

        # Number of objects detected
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def getPrediction(self):
        # Load the Tensorflow model into memory.
        # Load image using OpenCV and
        # expand image dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        sess = tf.Session(graph=self.detection_graph)
        image = cv2.imread(self.PATH_TO_IMAGE)
        image_expanded = np.expand_dims(image, axis=0)
        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_expanded})

        result = scores.flatten()
        res = []
        for idx in range(0, len(result)):
            if result[idx] > .40:
                res.append(idx)

        top_classes = classes.flatten()
        
        res_list = [top_classes[i] for i in res]

        class_final_names = [self.class_names_mapping[x] for x in res_list]
        top_scores = [e for l2 in scores for e in l2 if e > 0.30]
        # final_output = list(zip(class_final_names, top_scores))

        # print(final_output)

        # new_classes = classes.flatten()
        new_scores = scores.flatten()

        new_boxes = boxes.reshape(300, 4)

        # get all boxes from an array
        max_boxes_to_draw = new_boxes.shape[0]
        # this is set as a default but feel free to adjust it to your needs
        min_score_thresh = .60
        # iterate over all objects found
        
        # Getting Coordinates
        listOfOutput = []
        for (name, score, i) in zip(class_final_names, top_scores, range(min(max_boxes_to_draw, new_boxes.shape[0]))):
            valDict = {}
            valDict["className"] = name
            
            valDict["confidence"] = str(score)
            if new_scores is None or new_scores[i] > min_score_thresh:
                print(valDict["className"])
                val = list(new_boxes[i])
                valDict["yMin"] = str(val[0])
                valDict["xMin"] = str(val[1])
                valDict["yMax"] = str(val[2])
                valDict["xMax"] = str(val[3])
                listOfOutput.append(valDict)
        # new_boxes = boxes.reshape(100,4)
        # print(new_boxes)
        # print(type(new_boxes))
        # print(new_boxes.shape)
        # print(boxes.shape)
        # Draw the results of the detection (aka 'visulaize the results')

        vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            self.category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.60)
        output_filename = 'detected'+datetime.datetime.now().time().strftime("%M%S")+'.jpg'
        cv2.imwrite('test_images/output/detected/'+output_filename, image)
        #cv2.imshow('image',output_filename)
        
        return listOfOutput, image
    
    
    def cropped_img(self,img,result):
        self.img = img
        self.result = result
        self.height = img.shape[0]
        self.width = img.shape[1]
        self.c = 0
        time = datetime.datetime.now().time()
        header = ("Person","Date","Time")
        filename = 'data.csv'
        data = []
        for i in range(len(self.result)):
            try:
                if((result[i]['className'])=='with_mask'):
                    print("Person with Mask Detected!!")
                    

                elif((result[i]['className'])=='without_mask'):
                    ymin = float(self.result[i]['yMin'])
                    xmin = float(self.result[i]['xMin'])
                    ymax = float(self.result[i]['yMax'])
                    xmax = float(self.result[i]['xMax'])
                    y1 = int(ymin*self.height/1.1)
                    y2 = int(ymax*self.height*1.1)
                    x1 = int(xmin*self.width/1.1)
                    x2 = int(xmax*self.width*1.1)
                    #print('Y-Min:',ymin,'\nX-Min:',xmin,'\nY-Max:',ymax,'\nX-Max:',xmax,'\nWidth:',width,'\nHeight:',height)

                    new_img = self.img[y1:y2,x1:x2]
                    #cv2.imwrite('output.png',new_img)
                    #plt.imshow(new_img)
                    #outfile = '%s/%s.jpg' % ('test_images', 'cropped' + str(datetime.now()))
                    #cv2.imwrite(outfile, new_img)
                    self.file = 'test_images/output/cropped/output(cropped)'+str(self.c)+datetime.datetime.now().time().strftime("%M%S")+'.jpg'
                    cv2.imwrite(self.file,new_img)
                    print("Image Saved")
                    
                    #outfile = '%s/%s.jpg' % (self.tgtdir, self.basename + str(datetime.now()))
                    op = self.model_predict()
                    
                    
                    data.append({'Person':op,'Date':datetime.datetime.now().date(),'Time':time})        
            except Exception:
                traceback.print_exc()
        if(os.path.isfile('data.csv') == True):
            self.writer(header,data,filename,'append')
        elif(os.path.isfile('data.csv')==False):
            self.writer(header,data,filename,'write')

        #return op            
                
        
    def model_predict(self):
        import glob
        
        self.image_path = self.file
        self.img_model_path = 'best_model.h5'

        saved_model = load_model(self.img_model_path,compile=False)
        img = image.load_img(self.image_path, target_size=(150, 150))
        #image = cv2.resize(img)
        # Preprocessing the image
        x = image.img_to_array(img,dtype='double')

        x=x/255
        x = np.expand_dims(x, axis=0)
        #x = preprocess_input(x)

        preds = saved_model.predict_classes(x)

        if preds==0:
            self.Output = "Manthan"
        elif preds==1:
            self.Output= "Unknown"
        print(self.Output)

        t_img = cv2.imread(self.image_path)
        t_img = cv2.putText(t_img, "O/P : " + str(self.Output), (20, 70),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)

        self.t_file = 'test_images/output/detected/final'+str(self.c)+datetime.datetime.now().time().strftime("%M%S")+'.jpg'
        cv2.imwrite(self.t_file,t_img)
        self.t_file = 'test_images/output/final_images/final'+str(self.c)+datetime.datetime.now().time().strftime("%M%S")+'.jpg'
        cv2.imwrite(self.t_file,t_img)

        #cv2.imwrite('test_images/output/output(final).jpg',t_img)
        self.c = self.c+1

        return self.Output

    def writer(self,header, data, filename, option):
        import csv
        if option == "write":
            with open (filename, "w", newline = "") as csvfile:
                    self.movies = csv.writer(csvfile)
                    self.movies.writerow(header)
                    for x in data:
                        movies.writerow(x)
        
        elif option == "append":        
            with open(filename, "a", newline='') as csvfile:
                self.writer = csv.DictWriter(csvfile, fieldnames=header)
                for x in data:
                    self.writer.writerow(x)

        else:
            print("Option is not known")
        
    '''def writer(self,header, data, filename):
                    import csv
                    with open (filename, "w", newline = "") as csvfile:
                        self.movies = csv.writer(csvfile)
                        self.movies.writerow(header)
                        for x in data:
                            self.movies.writerow(x)'''

    def list_only_jpg_files(self):
        if os.path.exists('./test_images/output/detected'):
            self.list_of_files_detected=os.listdir('./test_images/output/detected')
            print('------list of files------')
            print(self.list_of_files_detected)
            
        if os.path.exists('./test_images/output/final_images'):
            self.list_of_files_final=os.listdir('./test_images/output/final_images')
            print('------list of files------')
            print(self.list_of_files_final)

        if not (os.path.exists('./test_images/output/detected') and os.path.exists('./test_images/output/final_images')):
            pass
        return self.list_of_files_detected, self.list_of_files_final



    def delete_existing_image(self,list_of_files_detected,list_of_files_final):
        for self.image in list_of_files_detected:
            if(len(list_of_files_detected) == 0):
                print('No files in detected')
            else:
                try:
                    print("------Deleting File------")
                    os.remove("./test_images/output/detected/"+self.image)
                except Exception as e:
                    print('error in deleting:  ',e)
        
        for self.image in list_of_files_final:
            if(len(list_of_files_final) == 0):
                print('No files in final folder')
            else:
                try:
                    print("------Deleting File------")
                    os.remove("./test_images/output/final_images/"+self.image)
                except Exception as e:
                    print('error in deleting:  ',e)

