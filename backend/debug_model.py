import torch
import torchvision.models as models
import numpy as np
import os

def debug_model():
    """Debug the model to see what's happening"""
    print("🔧 Debugging Liver Disease Model...")
    
    model_path = 'final.pt'
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("📁 Current directory files:")
        for file in os.listdir('.'):
            print(f"  - {file}")
        return
    
    try:
        # Try to load as YOLO first
        print("🔄 Trying to load as YOLO model...")
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
        print("✅ Loaded as YOLO model!")
        print(f"📊 Model classes: {model.names if hasattr(model, 'names') else 'Not found'}")
        
        # Test prediction
        print("\n🧪 Testing YOLO model with random image...")
        test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        results = model(test_image)
        print(f"📋 YOLO results: {results}")
        if hasattr(results, 'pred'):
            print(f"🔍 Predictions: {results.pred[0] if len(results.pred) > 0 else 'No detections'}")
        
    except Exception as e:
        print(f"❌ YOLO loading failed: {e}")
        
        # Try as standard PyTorch model
        print("🔄 Trying to load as standard PyTorch model...")
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            print("📋 Checkpoint keys:", list(checkpoint.keys()))
            
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                print("📊 Class names:", checkpoint.get('class_names', 'Not found'))
                print("🖼️ Image size:", checkpoint.get('img_size', 'Not found'))
            else:
                state_dict = checkpoint
                
            print("📋 State dict keys (first 10):", list(state_dict.keys())[:10])
            
            # Test with a simple model
            model = models.resnet50(pretrained=False)
            model.fc = torch.nn.Linear(model.fc.in_features, 4)
            
            try:
                model.load_state_dict(state_dict)
                print("✅ Standard model loaded successfully")
                
                # Test inference
                print("\n🧪 Testing with random inputs:")
                for i in range(2):
                    random_input = torch.randn(1, 3, 640, 640)
                    with torch.no_grad():
                        output = model(random_input)
                        probabilities = torch.sigmoid(output)  # Using sigmoid for multi-label
                        print(f"Test {i+1} - Output shape: {output.shape}")
                        print(f"Test {i+1} - Probabilities: {probabilities}")
                        print(f"Test {i+1} - Max: {torch.max(probabilities).item():.4f}")
                        print("-" * 50)
                        
            except Exception as e2:
                print(f"❌ Error loading standard model: {e2}")
                
        except Exception as e3:
            print(f"❌ All loading methods failed: {e3}")

if __name__ == '__main__':
    debug_model()