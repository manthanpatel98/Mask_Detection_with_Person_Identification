
import os
from flask_cors import CORS,cross_origin
from flask import Flask, render_template, request,jsonify
from flask import Flask, request, render_template, send_from_directory, send_file
from mask_detector_and_person_classifier.detector import Mask_Person

app = Flask(__name__)


APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def index():
    return render_template("upload.html")

@app.route("/upload", methods=["POST"])
def upload():
    target = os.path.join(APP_ROOT, 'images/')
    print(target)
    if not os.path.isdir(target):
            os.mkdir(target)
    else:
        print("Couldn't create upload directory: {}".format(target))
    print(request.files.getlist("file"))
    for upload in request.files.getlist("file"):
        print(upload)
        print("{} is the file name".format(upload.filename))
        filename = upload.filename
        destination = "/".join([target, filename])
        print ("Accept incoming file:", filename)
        print ("Save it to:", destination)              #destination
        upload.save(destination)
        
        mask_person = Mask_Person(destination)
        detection_img,detected_people=mask_person.list_only_jpg_files()

        mask_person.delete_existing_image(detection_img,detected_people)
        result, img = mask_person.getPrediction()
        mask_person.cropped_img(img, result)

        import glob
        mask_detect_imgs = os.listdir('./test_images/output/detected/') 
        person_imgs = os.listdir('./test_images/output/final_images/') 
        print(mask_detect_imgs)
        print(person_imgs)

    

        


    return render_template("display.html",person_imgs=person_imgs,mask_detect_imgs=mask_detect_imgs)


@app.route('/upload/<filename>')
def send_image_detected(filename):
    return send_from_directory('test_images/output/detected',filename)

@app.route('/upload/<filename>')
def send_image_person(filename):

    return send_from_directory('test_images/output/final_images',filename)

@app.route('/download')
def download_file():
    path = "data.csv"
    return send_file(path, as_attachment=True)

if __name__ == "__main__":
    app.run(port=4555, debug=True)