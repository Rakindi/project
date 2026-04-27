from django.shortcuts import render
from .models import UserModels, LogStorage
import tensorflow as tf
import torch
from . import Prediction_Layer
import joblib
import cv2
import numpy as np

from io import BytesIO
import base64
from django.conf import settings

# cnn_model = tf.keras.models.load_model(r"manageApp\models\cnn.h5")
# mobilnet_model=tf.keras.models.load_model(r'manageApp\models\mobilenet.h5')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def classify_frature_type(base_64):
    newImage = preprocess_image_for_keras(base_64)
    features = feature_extractor.predict(newImage)
    features = features.reshape(features.shape[0] - 1)

    predict_index = rf_model.predict(features)[0]
    return class_names[predict_index]


def load_main_part(base_64):
    frature_prediction = predict_frature(base_64)
    predicted_frature = map_prediciton_to_lable(frature_prediction)

    if predicted_frature == "Frature":
        fracture_type = classify_frature_type(base_64)
        print(f"Detected Fracture Type: {fracture_type}")
    else:
        print("No Frature")


def index(request):
    user = LogStorage.objects.last()
    if user != None:
        store = getDetails()
        name = store["name"]
        caps = ""
        for i in range(len(name)):
            if i == 0:
                caps += name[i].upper()
            else:
                caps += name[i].lower()
        store["name"] = caps
        return render(request, "userpage.html", store)
    else:
        return render(request, "index.html")


def signup(request):
    return render(request, "signup.html")


def register(request):
    if request.method == "POST":
        name = request.POST.get("name", "")
        email = request.POST.get("email", "")
        mobile = request.POST.get("mobile", "")
        password = request.POST.get("password", "")
        if UserModels.objects.filter(email=email).exists():
            data = {
                "name": name,
                "email": email,
                "mobile": mobile,
                "password": password,
                "message": "Try with another Email",
            }
            return render(request, "signup.html", data)
        else:
            UserModels.objects.create(
                name=name, email=email, mobile=mobile, password=password
            ).save()
            return index(request)


def userlogin(request):

    if request.method == "POST":
        email = request.POST.get("email", "")
        password = request.POST.get("password", "")
        data = UserModels.objects.filter(email=email, password=password).last()
        if data is None:
            data = {"email": email, "password": password, "message": "Invalid User"}
            return render(request, "index.html", data)
        else:
            LogStorage.objects.create(userid=data.userid)
            return index(request)


def upload(request):

    user = LogStorage.objects.last()
    if user != None:
        store = getDetails()
        name = store["name"]
        caps = ""
        for i in range(len(name)):
            if i == 0:
                caps += name[i].upper()
            else:
                caps += name[i].lower()
        store["name"] = caps
        return render(request, "uploadpage.html", store)
    else:
        return render(request, "index.html")


def uploadscreen(request):
    if request.method == "POST":
        file = request.POST.get("options", "")
        base64Image = request.POST.get("base64", "")

        prediction = Prediction_Layer.main(base64Image)
       
        store = getDetails()
    print(f"{ prediction[0]['Cure']},{prediction[0]['About']}")
    return render(
        request,
        "uploadpage.html",
        {
            "image": base64Image,
            "select": file,
            "name": store["name"],
            "prediction": prediction[0]["prediction"],
            "Cure22": prediction[0]["Cure"],
            "About22": prediction[0]["About"]
        },
    )


def logout(request):
    LogStorage.objects.all().delete()
    return render(request, "index.html")


def profile(request):

    status = ""
    if request.method == "POST":
        try:
            mobile = request.POST.get("mobile", "")
            name = request.POST.get("name", "")
            password = request.POST.get("password", "")
            data = UserModels.objects.filter(
                userid=LogStorage.objects.last().userid
            ).last()
            data.mobile = mobile
            data.name = name
            data.password = password
            data.save()
            status = "Success"
        except:
            status = "Failed"

    store = getDetails()
    store["status"] = status
    return render(request, "profile.html", store)


def getDetails():
    details = UserModels.objects.filter(userid=LogStorage.objects.last().userid).last()
    data = {
        "name": details.name,
        "mobile": details.mobile,
        "password": details.password,
    }
    return data


# if "base64," in base64Image:
#     base64Image = base64Image.split("base64,")[1]

# missing = len(base64Image) % 4
# if missing != 0:
#     base64Image += "=" * (4 - missing)
# data = np.ndarray(shape=(1, 256, 256, 3), dtype=np.float32)
# byte = BytesIO(base64.b64decode(base64Image))
# image = Image.open(byte)
# if image.mode != "RGB":
#     image = image.convert("RGB")

# size = (256, 256)
# image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
# image_array = np.asarray(image)
# normalized = (image_array.astype(np.float32) / 127.5) - 1
# data[0] = normalized
# if file == "Mobile net":
#     prediction = mobilnet_model.predict(data)
# else:
#     prediction = cnn_model.predict(data)
# index = np.argmax(prediction)
# print("Index: ", index)
# names = class_names[index]
# confindenscore = prediction[0][index]

# print(names)
# uploaded['prediction']=names
# uploaded['confidence']=confindenscore
