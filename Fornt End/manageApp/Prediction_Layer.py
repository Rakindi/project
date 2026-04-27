import numpy as np
import cv2  # OpenCV for image loading and processing
import joblib
import tensorflow as tf
import torch
from torchvision import models, transforms
from torch import nn
from PIL import Image
import os
import base64
from io import BytesIO

# Set the device for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MobileNetModel(nn.Module):
    def __init__(self, num_classes):
        super(MobileNetModel, self).__init__()
        self.mobilenet = models.mobilenet_v2(pretrained=True)
        num_features = self.mobilenet.classifier[1].in_features
        self.mobilenet.classifier[1] = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.mobilenet(x)
    
fracture_model = MobileNetModel(num_classes=2)
fracture_model.load_state_dict(torch.load("manageApp/models/mobilenet.pt", map_location=device))
fracture_model = fracture_model.to(device)
fracture_model.eval()

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

feature_extractor = tf.keras.models.load_model('manageApp/models/mobilenet_feature_extractor.h5')
rf_model = joblib.load('manageApp/models/random_forest_classifier.pkl')

class_names = [
    'Avulsion fracture',
    'Comminuted fracture',
    'Fracture Dislocation',
    'Greenstick fracture',
    'Hairline Fracture',
    'Impacted fracture',
    'Longitudinal fracture',
    'Oblique fracture',
    'Pathological fracture',
    'Spiral Fracture'
]

about=[
    'This occurs when a tendon or ligament pulls off a piece of bone. Treatment depends on the severity. Non-displaced fractures can heal with rest, ice, and physical therapy, while displaced fractures may require surgical intervention. Early diagnosis and management are crucial to prevent long-term joint instability or dysfunction.',
    'A comminuted fracture involves the bone breaking into multiple pieces. Treatment often requires surgery, where the bone fragments are realigned and fixed with plates or screws. Rehabilitation is important for restoring strength and mobility, but recovery may take longer due to the complexity of healing multiple fractures.',
    'This is a combination of a fracture and joint dislocation. It is typically treated with a combination of reduction (realigning bones) and fixation. Prompt treatment is vital to prevent nerve or blood vessel damage and to restore joint function. Rehabilitation is essential to regain full range of motion and strength.',
    'A greenstick fracture is an incomplete break in a bone, commonly seen in children. The bone bends and partially fractures, similar to bending a green twig. Treatment usually involves casting to immobilize the bone while it heals. Healing time is shorter in children due to their fast bone regeneration.',
    'A hairline fracture is a small, thin crack in the bone that may not always be visible on X-rays. It often results from stress or overuse. Treatment includes rest, ice, and limited weight-bearing. Healing is usually fast if the bone is not subjected to further stress, and proper rest is crucial.',
    'This occurs when one bone fragment is driven into another, often from a fall or collision. Treatment involves realigning the bone and stabilizing it, sometimes with surgery. Early intervention is essential to restore proper bone function and prevent complications like joint stiffness or deformities.',
    'A longitudinal fracture runs parallel to the length of the bone. It often occurs due to direct trauma or twisting forces. Treatment typically involves immobilization through casting or splinting. In some cases, surgery may be required if the fracture is unstable. Rehabilitation focuses on restoring the bone’s strength.',
    'An oblique fracture occurs when the break is at an angle to the bone’s axis. This type of fracture can be caused by a direct blow or twisting motion. Treatment typically involves casting or surgical fixation, depending on the displacement. Proper alignment during healing is critical to prevent future complications.',
    'This type of fracture occurs due to a pre-existing bone condition like osteoporosis or cancer, making the bone weaker. Treatment focuses on addressing the underlying condition while healing the fracture, which may involve surgery or bracing. Bone-strengthening therapies and medications are often necessary to prevent future fractures.',
    'A spiral fracture is caused by a twisting or rotational force. The bone breaks in a spiral pattern, often resulting from falls or sports injuries. Treatment typically requires realigning the bone, either surgically or with a cast. Physical therapy is crucial after healing to restore movement and strength.'
]
cure=[
    'An avulsion fracture occurs when a tendon or ligament pulls off a piece of bone. The severity of treatment depends on the extent of the fracture and whether the tendon or ligament is damaged. If the fracture is minor and non-displaced (the bone is still aligned), the first treatment is usually conservative: rest, ice, compression, and elevation (R.I.C.E.), followed by immobilization with a cast or splint. Non-weight-bearing activities should be avoided to prevent further stress on the healing bone. If the fracture is displaced or the ligament or tendon is severely torn, surgical intervention may be necessary. A surgeon will reattach the tendon or ligament to the bone using screws, pins, or anchors. Post-surgery, the patient will need a period of immobilization, followed by physical therapy to regain full function. Physical therapy focuses on strengthening the muscles around the joint and improving flexibility, gradually increasing activity levels as healing progresses. For mild cases, recovery may take around 6–8 weeks, but for more severe injuries, it can take several months. Early intervention is essential to prevent long-term complications, such as joint instability, weakness, or loss of motion. Regular follow-up with an orthopedic specialist is recommended.',
    'A comminuted fracture occurs when the bone shatters into three or more pieces. This is a complex fracture that requires careful treatment and management. The first step in treating a comminuted fracture is to reduce the fracture, which may involve realigning the bone fragments either manually or surgically. Non-surgical treatment is rare unless the fracture is stable and the bone fragments are not displaced. Surgical intervention is often required, and this may involve the use of metal plates, screws, or rods to hold the bone fragments together while they heal. If necessary, bone grafting might be performed to promote healing, especially if there is significant bone loss. Post-surgery, immobilization is necessary to allow the bone fragments to heal properly. A cast, splint, or external fixator might be used to stabilize the area. The patient may be restricted from putting weight on the affected limb for several weeks to months. Rehabilitation is a crucial part of the recovery process. Physical therapy helps to restore strength, flexibility, and range of motion once the bone is sufficiently healed. The healing time for comminuted fractures can vary, but it usually takes several months, with more time required for complete recovery.',
    'A fracture dislocation is a complex injury in which a bone is broken and the joint is dislocated. Immediate medical attention is required for such injuries to avoid complications like nerve, artery, or ligament damage. Treatment typically begins with closed reduction, where the dislocated bones are realigned and the fracture is reduced (realigned). In some cases, if closed reduction is unsuccessful or the fracture is unstable, surgery may be necessary. Once the bones are aligned, they are immobilized using a cast, brace, or splint. In some cases, surgical fixation (with plates, screws, or pins) is required to stabilize the fracture. Joint dislocations often need more time to heal due to soft tissue and ligament damage. If ligaments are torn or severely stretched, surgical repair might be needed. In the case of a fracture dislocation, rehabilitation is especially important to restore joint stability and function. After the initial healing phase, physical therapy will focus on regaining range of motion, strengthening the muscles around the joint, and gradually returning to weight-bearing activities. The recovery time for a fracture dislocation varies but can range from 8 to 12 weeks, depending on the severity of the injury. Regular follow-up visits to an orthopedic specialist are essential to ensure proper healing.',
    'A greenstick fracture is an incomplete fracture in which the bone bends and cracks on one side without breaking completely through, resembling the way a green twig might bend. This type of fracture typically occurs in children due to the flexibility of their bones. Treatment for a greenstick fracture generally involves conservative management, as the bone usually heals quickly and without the need for surgery. Initially, the affected area is immobilized with a splint or cast to prevent further movement and allow the bone to heal. Depending on the location and severity of the fracture, the cast may need to be worn for 3 to 6 weeks. During this period, the bone gradually realigns itself and strengthens. In cases where the fracture is more displaced or the bone is significantly bent, manual reduction (a procedure where the bones are gently repositioned) might be required. Once repositioned, the bone is immobilized to ensure proper alignment during the healing process. Physical therapy is often not needed for greenstick fractures, as they typically heal faster and do not cause long-term complications. However, gentle movements and stretches might be advised after the cast is removed to restore mobility. Recovery is generally quick, with most children returning to normal activities within 6 weeks.',
    'A hairline fracture is a small crack in the bone, often caused by repetitive stress or overuse, and can be difficult to detect on X-rays. These fractures are common in athletes or individuals who engage in high-impact or repetitive activities. Treatment for a hairline fracture generally involves conservative measures, with the primary goal being rest and protection of the injured area to promote healing. The first step is typically to stop any activity that could worsen the injury. The area should be immobilized using a brace, cast, or splint, depending on the location of the fracture. Crutches or a walker may be used to avoid putting weight on the affected limb. In some cases, a hairline fracture may require an MRI or more advanced imaging to ensure that the bone is healing correctly and that there are no hidden complications. The fracture should heal in 4 to 8 weeks, but this varies depending on the severity and location. Physical therapy may be recommended after the bone has healed sufficiently to help restore strength, flexibility, and range of motion to the injured area. Preventive measures, such as gradually increasing activity intensity and incorporating cross-training, can help avoid re-injury once healing is complete.',
    'An impacted fracture occurs when one bone fragment is driven into another, often due to a fall or high-impact trauma. These fractures can be unstable and may cause the bones to become misaligned. The first step in treatment is to reduce the fracture by gently realigning the bone fragments, either through manual manipulation (closed reduction) or surgery (open reduction). In cases where the bone fragments are well aligned, a cast or splint may be sufficient to stabilize the fracture during the healing process. If the fracture is displaced or if the bone is in a precarious position, surgery may be necessary to insert screws, plates, or rods to stabilize the bone fragments and promote healing. Pain management is a key component of treatment, with medications prescribed to control pain and reduce inflammation. The area will likely need to be immobilized for several weeks, with weight-bearing avoided for some time to allow the bone to heal. Physical therapy is often recommended once the bone has sufficiently healed. Therapy will focus on strengthening the surrounding muscles, improving range of motion, and restoring mobility. Recovery time typically ranges from 8 to 12 weeks, but full recovery may take longer depending on the severity of the fracture.',
    'A longitudinal fracture runs parallel to the length of the bone, often caused by a direct blow or twisting force. These fractures are typically treated conservatively if they are stable and the bone fragments are not displaced. The first step is usually to apply a cast, splint, or brace to immobilize the bone and allow it to heal in the correct position. In some cases, manual reduction (where a doctor carefully repositions the bone fragments) is required to ensure proper alignment. If the fracture is more complex or unstable, surgical intervention may be needed to insert pins, screws, or plates to stabilize the bone. During the healing process, weight-bearing activities may be restricted for several weeks, depending on the location and severity of the fracture. Pain management with nonsteroidal anti-inflammatory drugs (NSAIDs) may be necessary to reduce swelling and discomfort. Rehabilitation begins once the fracture has sufficiently healed and the cast or splint is removed. Physical therapy will focus on restoring mobility, flexibility, and strength. The rehabilitation process is crucial to help the individual regain full function and reduce the risk of long-term complications like joint stiffness or weakness.',
    'An oblique fracture occurs when the bone breaks at an angle to the bone’s axis, typically resulting from a direct blow or twisting force. The treatment depends on the severity of the fracture and whether the bone is displaced. If the fracture is stable and the bone fragments are aligned, the treatment involves immobilization with a cast or splint. In more severe cases, when the bone is displaced, surgical intervention may be necessary to realign the bone fragments and hold them together with plates, screws, or rods. This surgery ensures that the bone heals correctly and prevents further complications. For most oblique fractures, healing typically takes 6 to 8 weeks, with a gradual return to normal activities. During the healing process, weight-bearing should be avoided until the bone is stable. Regular follow-up visits to monitor healing progress are important. Once the bone has healed sufficiently, rehabilitation through physical therapy helps restore strength and mobility. Therapy may include exercises to improve joint flexibility, muscle strength, and functional movement patterns. The goal is to return the individual to their normal activities while preventing future injuries.',
    'A pathological fracture occurs when a bone breaks due to an underlying condition, such as osteoporosis, cancer, or an infection, which weakens the bone. Treatment for pathological fractures requires addressing both the fracture itself and the underlying cause of bone weakness. In many cases, surgery may be necessary to stabilize the bone, especially if the fracture is displaced or the bone is severely weakened. Surgical procedures may include fixation with metal plates, rods, or screws, and in some cases, bone grafts might be used to promote healing. Radiation or chemotherapy may be used if the fracture is caused by cancer or a tumor. For osteoporosis-related fractures, medications such as bisphosphonates or calcitonin may be prescribed to strengthen the bones. Pain management is important for managing discomfort associated with the fracture and underlying condition. Post-surgical rehabilitation or physical therapy focuses on restoring mobility, strength, and functional use of the affected area. Additionally, the patient will need to receive treatment for the underlying condition, whether it involves medication, lifestyle changes, or specialized therapy. Long-term management may be required to prevent further fractures. Depending on the severity of the condition, recovery can take several months.',
    'A spiral fracture is caused by a twisting or rotational force, often seen in sports injuries or accidents where the limb is suddenly twisted or rotated. Treatment for spiral fractures generally involves aligning the bone fragments either manually (closed reduction) or surgically (open reduction). If the fracture is displaced, surgery is typically required to insert metal rods, screws, or plates to stabilize the bone. Once the fracture is stabilized, a cast or splint will be applied to immobilize the bone while it heals. The patient is usually advised to avoid weight-bearing activities, especially during the early stages of healing, to prevent further damage. Pain management is essential in the acute phase, with over-the-counter or prescribed painkillers helping to control discomfort and swelling. The healing time for spiral fractures typically ranges from 6 to 12 weeks, depending on the location and severity. After the bone has sufficiently healed, physical therapy will be necessary to restore strength, flexibility, and function to the affected area. Rehabilitation may focus on range-of-motion exercises, strengthening exercises, and activities that improve overall mobility and stability. Full recovery from a spiral fracture may take several months, depending on the injury.' 
]

def map_prediction_to_label(prediction):
    label_mapping = {0: "No Fracture", 1: "Fracture"}
    return label_mapping.get(prediction, "Unknown")

def predict_fracture(image):
    image = image_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = fracture_model(image)
        _, predicted = torch.max(output, 1)

    return predicted.item()


def preprocess_image_for_keras(image):
    image = cv2.resize(image, (256, 256))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def classify_fracture_type(image):
    new_image = preprocess_image_for_keras(image)
    features = feature_extractor.predict(new_image)
    features = features.reshape(features.shape[0], -1)
    prediction_index = rf_model.predict(features)[0]


    return [
        {
            'prediction':class_names[prediction_index],
            'Cure':cure[prediction_index],
            'About':about[prediction_index]
        }
    ]


def decode_base64_to_image(base64_string):
    try:
        if "base64," in base64_string:
            base64_string = base64_string.split("base64,")[1]

        missing = len(base64_string) % 4
        if missing != 0:
            base64_string += "=" * (4 - missing)


        img_data = base64.b64decode(base64_string)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.COLOR_BGR2RGB)
        
        if img is None:
            raise ValueError("Decoded image is empty or invalid.")
        
        return img
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None


def main(base64_image):
    image = decode_base64_to_image(base64_image)
    
    if image is None:
        return "Failed to decode the image."
    
    try:
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) 
        fracture_prediction = predict_fracture(pil_image)
        predicted_label = map_prediction_to_label(fracture_prediction)

        if predicted_label == "Fracture":
            return  classify_fracture_type(image)
            
        else:
            return "No Fracture Detected."
    except Exception as e:
        print(f"Error during prediction: {e}")


