import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from MyResBet18 import ModifiedResNet18
from Triplet import TripletAttention
import sys
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_model(model_path, num_classes=5, n_div=2):
    model = ModifiedResNet18(n_div, num_classes).to(device)
    custom_module = TripletAttention()
    model.resnet.conv1 = nn.Sequential(
        model.resnet.conv1,
        custom_module
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def predict_image(model, image_path, class_names=None):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    image = image.to(device)
    
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
        confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted.item()].item() * 100
    
    if class_names:
        predicted_class = class_names[predicted.item()]
    else:
        predicted_class = predicted.item()
    
    return predicted_class, confidence

def get_class_names(root_dir='split_dataset/train'):
    if os.path.exists(root_dir):
        return sorted(os.listdir(root_dir))
    return None

def batch_predict(model, image_dir, class_names=None):
    results = []
    for img_name in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_name)
        if os.path.isfile(img_path) and img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            pred_class, confidence = predict_image(model, img_path, class_names)
            results.append({
                'image': img_name,
                'predicted_class': pred_class,
                'confidence': f"{confidence:.2f}%"
            })
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Garlic Leaf Disease Identification Inference')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the pre-trained model')
    parser.add_argument('--image_path', type=str, help='Path to a single image for prediction')
    parser.add_argument('--image_dir', type=str, help='Directory containing images for batch prediction')
    parser.add_argument('--num_classes', type=int, default=5, help='Number of disease classes')
    parser.add_argument('--n_div', type=int, default=2, help='n_div parameter for the model')
    
    args = parser.parse_args()
    
    if not args.image_path and not args.image_dir:
        print("Error: Either --image_path or --image_dir must be provided")
        sys.exit(1)
    
    class_names = get_class_names()
    if class_names:
        print(f"Class names: {class_names}")
    else:
        print("Warning: Could not load class names from dataset directory")
    
    print(f"Loading model from {args.model_path}...")
    model = load_model(args.model_path, args.num_classes, args.n_div)
    print("Model loaded successfully")
    
    if args.image_path:
        if os.path.exists(args.image_path):
            pred_class, confidence = predict_image(model, args.image_path, class_names)
            print(f"\nPrediction for {os.path.basename(args.image_path)}:")
            print(f"Class: {pred_class}")
            print(f"Confidence: {confidence:.2f}%")
        else:
            print(f"Error: Image file {args.image_path} not found")
    
    if args.image_dir:
        if os.path.isdir(args.image_dir):
            print(f"\nPerforming batch prediction on images in {args.image_dir}...")
            results = batch_predict(model, args.image_dir, class_names)
            
            print("\nBatch prediction results:")
            for result in results:
                print(f"{result['image']}: {result['predicted_class']} ({result['confidence']})")
        else:
            print(f"Error: Directory {args.image_dir} not found")