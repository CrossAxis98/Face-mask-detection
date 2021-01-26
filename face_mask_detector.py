from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import face_recognition
from PIL import Image
import smtplib, ssl
import numpy as np
import cv2
import os

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = load_model("face_mask.model")
video_capture = cv2.VideoCapture(0)

#Example data
gmail_user = 'face_mask_detector@gmail.com'
password = '1234'
recipient = 'client@gmail.com'

if not os.path.isdir("directory_with_people_not_wearing_mask"):
    os.mkdir("directory_with_people_not_wearing_mask")

while True:
    return_value, frame = video_capture.read()
    grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grayscale_frame, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60), flags=cv2.CASCADE_SCALE_IMAGE)
    predictions=[]
    faces_list=[]

    for (x, y, w, h) in faces:
        face = frame[y:y+h,x:x+w]
        face_frame = face
        face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
        face_frame = cv2.resize(face_frame, (100, 100))
        face_frame = img_to_array(face_frame)
        face_frame = np.expand_dims(face_frame, axis=0)
        face_frame = preprocess_input(face_frame)
        faces_list.append(face_frame)

        if len(faces_list)>0:
            for face_from_list in range(len(faces_list)):
                predictions = model.predict(faces_list[face_from_list])

        for prediction in predictions:
            (with_improperly_wear_mask, with_mask, without_mask) = prediction

        if (with_mask > without_mask) and (with_mask > with_improperly_wear_mask):
            label = "Mask"
            color = (0, 255, 0)
        elif (with_improperly_wear_mask > with_mask) and (with_improperly_wear_mask > without_mask):
            label = "Improperly wear"
            color = (0, 220, 220)
        else:
            label = "No Mask"
            color = (0, 0, 255)

            list_of_people_not_wearing_mask = os.listdir("directory_with_people_not_wearing_mask")
            amount_of_people_not_wearing_mask = len(list_of_people_not_wearing_mask)

            is_person_already_saved = False
            rgb_face= face[:, :, ::-1]

            try:
                 rgb_face_encoding = face_recognition.face_encodings(rgb_face)[0]
            except IndexError as index_error:
                print(index_error)
                break

            for file in list_of_people_not_wearing_mask:
                image_path = os.path.join("directory_with_people_not_wearing_mask", file)
                image_to_compare = face_recognition.load_image_file(image_path)
                image_to_compare_encoding = face_recognition.face_encodings(image_to_compare)[0]
                result = face_recognition.compare_faces([rgb_face_encoding], image_to_compare_encoding)

                if result[0]:
                    is_person_already_saved = True
                    break

            if not is_person_already_saved or amount_of_people_not_wearing_mask==0:
                rescaled = (255.0 / rgb_face.max() * (rgb_face - rgb_face.min())).astype(np.uint8)
                image_to_save = Image.fromarray(rescaled)
                file_name = "directory_with_people_not_wearing_mask/person{0}.png".format(amount_of_people_not_wearing_mask)
                image_to_save.save(file_name)

                with open(file_name, 'rb') as file:
                    multipart_message = MIMEMultipart()
                    multipart_message['Subject'] = 'Alert'
                    multipart_message.attach(MIMEText('This person is not wearing a face mask.'))
                    mimebase = MIMEBase('image', 'png', filename=file_name)
                    mimebase.set_payload(file.read())
                    encoders.encode_base64(mimebase)
                    multipart_message.attach(mimebase)

                try:
                        server = smtplib.SMTP('smtp.gmail.com', 587)
                        server.starttls(context=ssl.create_default_context())
                        server.login(gmail_user, password)
                        server.sendmail(gmail_user, recipient, multipart_message.as_string())
                except Exception as exception:
                        print(f'SMTP error occurred: {exception}')
                finally:
                        server.quit()


        label = "{}: {:.2f}%".format(label, max(with_mask, with_improperly_wear_mask, without_mask) * 100)
        cv2.rectangle(frame, (x, y), (x + w, y + h),color, 2)
        cv2.rectangle(frame,(x,y-30),(x+w,y),color,-1)
        cv2.putText(frame, label, (x, y- 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

    cv2.imshow('Monitoring', frame)

    key = cv2.waitKey(10)
    if key == 27:
        break

video_capture.release()
cv2.destroyAllWindows()
