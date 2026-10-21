"""
Bangladeshi AI Eye Doctor - Gradio Web Application
A retinal fundus image classifier for 10 eye disease classes.
This is a master's thesis project prototype and is NOT intended for clinical use.
"""
import os
import torch
import torch.nn as nn
import timm
from torchvision import transforms
from PIL import Image
import gradio as gr

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DISEASE_CLASSES = [
    'Central Serous Chorioretinopathy',
    'Diabetic Retinopathy',
    'Disc Edema',
    'Glaucoma',
    'Healthy',
    'Macular Scar',
    'Myopia',
    'Pterygium',
    'Retinal Detachment',
    'Retinitis Pigmentosa'
]

DISEASE_INFO = {
    'Central Serous Chorioretinopathy': (
        "A condition where fluid accumulates under the retina, causing distorted or blurred central vision. "
        "It most commonly affects men aged 20-50 and is often linked to stress or corticosteroid use. "
        "Recommendation: Consult an ophthalmologist. Most cases resolve on their own within a few months, "
        "but persistent cases may require laser treatment or photodynamic therapy."
    ),
    'Diabetic Retinopathy': (
        "A diabetes complication that damages blood vessels in the retina, potentially leading to vision loss. "
        "It is the leading cause of blindness in working-age adults worldwide. "
        "Recommendation: Urgent referral to an ophthalmologist. Strict blood sugar, blood pressure, and "
        "cholesterol control is essential. Treatment may include laser therapy or anti-VEGF injections."
    ),
    'Disc Edema': (
        "Swelling of the optic disc, which can indicate increased intracranial pressure, inflammation, or ischemia. "
        "It may present with blurred vision, headaches, or transient visual obscurations. "
        "Recommendation: Seek immediate medical attention. A thorough neurological and ophthalmological "
        "evaluation is needed to identify the underlying cause."
    ),
    'Glaucoma': (
        "A group of eye conditions that damage the optic nerve, often due to abnormally high intraocular pressure. "
        "It is one of the leading causes of irreversible blindness, especially in people over 60. "
        "Recommendation: Consult an ophthalmologist promptly. Early detection and treatment with eye drops, "
        "laser treatment, or surgery can slow or prevent vision loss."
    ),
    'Healthy': (
        "The retinal fundus appears normal with no signs of disease. The optic disc, macula, and blood vessels "
        "show healthy characteristics. "
        "Recommendation: Continue routine eye examinations every 1-2 years, or as advised by your eye care provider. "
        "Maintain a healthy lifestyle to support long-term eye health."
    ),
    'Macular Scar': (
        "Scarring of the macula, the central part of the retina responsible for sharp vision. "
        "It can result from infections, inflammation, injury, or age-related degeneration. "
        "Recommendation: Consult an ophthalmologist. While existing scars are usually permanent, treatment "
        "can address underlying causes and prevent further damage. Low-vision aids may help."
    ),
    'Myopia': (
        "Pathological myopia involves severe nearsightedness with degenerative changes to the retina and eye structure. "
        "It increases the risk of retinal detachment, macular degeneration, and glaucoma. "
        "Recommendation: Regular monitoring by an ophthalmologist is essential. Watch for sudden changes in "
        "vision, flashes of light, or floaters, which may signal complications."
    ),
    'Pterygium': (
        "A benign growth of tissue on the conjunctiva that can extend onto the cornea, often associated with "
        "UV exposure, wind, and dust. It is common in tropical and subtropical regions like Bangladesh. "
        "Recommendation: Consult an ophthalmologist if the growth affects vision or causes discomfort. "
        "Surgical removal may be necessary. Wear UV-protective sunglasses for prevention."
    ),
    'Retinal Detachment': (
        "A serious condition where the retina separates from its supporting tissue, causing sudden flashes of light, "
        "floaters, or a shadow over the visual field. Without prompt treatment, it can lead to permanent vision loss. "
        "Recommendation: This is an EMERGENCY. Seek immediate ophthalmological care. Surgical intervention "
        "(pneumatic retinopexy, scleral buckle, or vitrectomy) is typically required."
    ),
    'Retinitis Pigmentosa': (
        "A group of inherited disorders causing progressive degeneration of the retina's photoreceptor cells. "
        "Symptoms typically begin with night blindness and tunnel vision in adolescence or early adulthood. "
        "Recommendation: Consult a retinal specialist. While there is no cure, vitamin A supplementation, "
        "gene therapy trials, and assistive devices can help manage the condition."
    )
}


class EfficientNetB0Classifier(nn.Module):
    def __init__(self, num_classes=10, dropout=0.3):
        super().__init__()
        self.model = timm.create_model(
            'efficientnet_b0', pretrained=False,
            num_classes=num_classes, drop_rate=dropout
        )
        self.num_classes = num_classes

    def forward(self, x):
        return self.model(x)


def load_model():
    model = EfficientNetB0Classifier(num_classes=len(DISEASE_CLASSES), dropout=0.3)
    checkpoint_path = os.path.join(BASE_DIR, 'final_model_for_deployment.pth')
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        if not any(k.startswith('model.') for k in state_dict.keys()):
            state_dict = {'model.' + k: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    model.to(DEVICE)
    model.eval()
    return model


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("Loading model...")
model = load_model()
print(f"Model loaded on {DEVICE}.")


def predict(image):
    if image is None:
        return {}, "Please upload a retinal fundus image."

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    image = image.convert('RGB')

    tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)[0]

    confidences = {DISEASE_CLASSES[i]: float(probs[i]) for i in range(len(DISEASE_CLASSES))}

    top_class = DISEASE_CLASSES[probs.argmax().item()]
    top_conf = probs.max().item()

    info = f"Top Prediction: {top_class} ({top_conf:.1%} confidence)\n\n"
    info += DISEASE_INFO.get(top_class, "No information available.")

    return confidences, info


example_dir = os.path.join(BASE_DIR, 'eye_images')
examples = []
if os.path.isdir(example_dir):
    for fname in sorted(os.listdir(example_dir)):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            examples.append(os.path.join(example_dir, fname))

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Retinal Fundus Image"),
    outputs=[
        gr.Label(num_top_classes=5, label="Disease Predictions"),
        gr.Textbox(label="Medical Information", lines=6)
    ],
    title="Bangladeshi AI Eye Doctor",
    description=(
        "Upload a retinal fundus photograph to get an AI-powered preliminary screening for 10 eye conditions. "
        "This tool uses an EfficientNet-B0 model trained on Bangladeshi patient data.\n\n"
        "**Disclaimer:** This is a master's thesis research prototype and is NOT intended for clinical diagnosis. "
        "Always consult a qualified ophthalmologist for medical advice."
    ),
    examples=examples if examples else None,
    flagging_mode="never",
    analytics_enabled=False,
)

if __name__ == '__main__':
    demo.launch(share=False, server_name="0.0.0.0")
