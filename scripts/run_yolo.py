import argparse
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description='Run YOLO training')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to run on (e.g. cuda:0 or cpu)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    return parser.parse_args()

def train_yolo(epochs, device):
    # Load a model
    model = YOLO("yolo11m.pt")

    # Train the model on custom dataset
    train_results = model.train(
        data="./config/yolo/dataset.yml",  # path to custom dataset YAML
        epochs=epochs,  # number of training epochs
        imgsz=640,  # training image size
        device=device,  # use specified device
    )

    # Evaluate model performance on the validation set
    metrics = model.val()

    # Example inference on test image
    results = model("data/wood-defects-parsed/images/test/test_image.jpg")  
    results[0].show()

    # Export the model to ONNX format
    path = model.export(format="onnx")  # return path to exported model


if __name__ == '__main__':
    args = parse_args()
    train_yolo(args.epochs, args.device)

