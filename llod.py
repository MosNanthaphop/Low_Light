import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Func
import torchvision.transforms as transforms
from PIL import Image
import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO

# กำหนดอุปกรณ์ (ใช้ GPU ถ้ามี)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# โครงสร้างของโมเดล Zero-DCE (Zero-Reference Deep Curve Estimation)
class ZeroDCE(nn.Module):
    def __init__(self):
        super(ZeroDCE, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        number_f = 32
        self.e_conv1 = nn.Conv2d(3,number_f,3,1,1,bias=True)
        self.e_conv2 = nn.Conv2d(number_f,number_f,3,1,1,bias=True)
        self.e_conv3 = nn.Conv2d(number_f,number_f,3,1,1,bias=True)
        self.e_conv4 = nn.Conv2d(number_f,number_f,3,1,1,bias=True)
        self.e_conv5 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True)
        self.e_conv6 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True)
        self.e_conv7 = nn.Conv2d(number_f*2,24,3,1,1,bias=True)
    
    def forward(self, x):
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3,x4],1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2,x5],1)))
        x_r = Func.tanh(self.e_conv7(torch.cat([x1,x6],1))) * 2.5
        r1,r2,r3,r4,r5,r6,r7,r8 = torch.split(x_r, 3, dim=1)
        x = x + r1*(torch.pow(x,2)-x)
        x = x + r2*(torch.pow(x,2)-x)
        x = x + r3*(torch.pow(x,2)-x)
        enhance_image_1 = x + r4*(torch.pow(x,2)-x)
        
        #use encahnce_image for darker
        x = enhance_image_1 + r5*(torch.pow(enhance_image_1,2)-enhance_image_1)
        x = x + r6*(torch.pow(x,2)-x)
        x = x + r7*(torch.pow(x,2)-x)
        enhance_image = x + r8*(torch.pow(x,2)-x)
        
        return enhance_image_1, enhance_image

# โหลดโมเดลจากไฟล์
model_path = "models/Epoch99.pth"
model = ZeroDCE().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

yolo_model = YOLO("models/yolo12l.pt")

# ฟังก์ชันสำหรับตรวจจับวัตถุด้วย YOLO
def detect_objects(image, model):
    results = model(image)
    return results

# ฟังก์ชันปรับปรุงภาพแสงน้อย
def enhance_image(image_path, model, transform, device):

    # เปิดรูปด้วย PIL และแปลงเป็น RGB
    original_image = Image.open(image_path).convert("RGB")
    original_size = original_image.size

    # แปลงรูปเป็น Tensor และส่งไปที่ GPU
    image = transform(original_image).unsqueeze(0).to(device)
    with torch.no_grad(): # ปิดการเรียนรู้ เพื่อลดการใช้หน่วยความจำ
        enhanced_image = model(image)[0].squeeze(0).cpu()

    # แปลงกลับจาก Tensor เป็นรูปภาพปกติ
    enhanced_image = transforms.ToPILImage()(enhanced_image)
    enhanced_image = enhanced_image.resize(original_size, Image.LANCZOS)
    return original_image, enhanced_image


# การแปลงภาพ
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

# ซ่อนหน้าต่างหลักของ Tkinter
root = tk.Tk()
root.withdraw()

# เปิดหน้าต่างให้ผู้ใช้เลือกไฟล์ภาพ
file_path = filedialog.askopenfilename(
    title="Choose file", 
    filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.webp")])

# ถ้าผู้ใช้เลือกไฟล์ ระบบจะนำไปประมวลผล
if file_path:
    original_image, enhanced_image = enhance_image(file_path, model, transform, device)

    #แปลงภาพจาก PIL เป็น NumPy array เพื่อใช้กับ OpenCV
    original_cv = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
    enhanced_cv = cv2.cvtColor(np.array(enhanced_image), cv2.COLOR_RGB2BGR)

    #ปรับปรุงภาพหลังจากใช้ Zero-DCE (Contrast, Brightness)
    brightened_image = cv2.convertScaleAbs(enhanced_cv, alpha=1.4, beta=1)

    #Unsharp Masking ถ้ามากไปเกิด Ringing
    sharpened_image = cv2.addWeighted(brightened_image, 1.3, cv2.GaussianBlur(enhanced_cv, (0,0), 1), -0.3, 0)
    
    # ตรวจจับวัตถุบนภาพที่ผ่านการปรับปรุงแล้ว
    results = detect_objects(sharpened_image, yolo_model)
    annotated_cv = results[0].plot()

    # ปรับขนาดภาพเพื่อแสดงผล
    max_size = 700
    h, w = original_cv.shape[:2]
    scale = min(max_size / max(h, w), 1.0)
    new_size = (int(w * scale), int(h * scale))
    original_resized = cv2.resize(original_cv, new_size, interpolation=cv2.INTER_LINEAR)
    detected_resized = cv2.resize(annotated_cv, new_size, interpolation=cv2.INTER_LINEAR)

    #แสดงภาพ
    cv2.imshow("Original Image", original_resized)
    cv2.imshow("Enchanted and Detected", detected_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()