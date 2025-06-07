import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
# مسار مجلد الصور
data_dir = 'gemstone_colors_dataset'

# إعدادات التحويل (resizing + normalization)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# تحميل البيانات
dataset = datasets.ImageFolder(root=data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# معرفة عدد الفئات (الألوان)
num_classes = len(dataset.classes)

# استخدام نموذج جاهز (ResNet18) وتعديله
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# تعريف الخسارة والمُحسّن
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# هل يوجد GPU؟
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# تدريب سريع لعدة epochs
for epoch in range(5):
    model.train()
    running_loss = 0.0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    print(f"📚 Epoch {epoch+1}/5 - Loss: {running_loss:.4f}")

# حفظ النموذج
torch.save(model.state_dict(), "gem_color_classifier.pt")
print("✅ Model saved as gem_color_classifier.pt")
