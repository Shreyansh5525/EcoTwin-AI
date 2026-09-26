import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=1
)

model.load_state_dict(
    torch.load(
        "best_model.pth",
        map_location=DEVICE
    )
)

model.to(DEVICE)
model.eval()


def predict_tree_cover(image):

    # Ensure image is RGB
    image = image.convert("RGB")

    image_np = np.array(image)

    h, w = image_np.shape[:2]

    img = cv2.resize(
        image_np,
        (256, 256)
    )

    img = img.astype(np.float32) / 255.0

    tensor = torch.tensor(
        img.transpose(2, 0, 1),
        dtype=torch.float32
    ).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        pred = model(tensor)

    pred = torch.sigmoid(pred)

    pred = pred[0, 0].cpu().numpy()

    pred = (pred > 0.5).astype(np.uint8)

    pred = cv2.resize(
        pred,
        (w, h),
        interpolation=cv2.INTER_NEAREST
    )

    tree_pixels = np.sum(pred == 1)

    total_pixels = pred.size

    tree_cover = (
        tree_pixels / total_pixels
    ) * 100

    overlay = image_np.copy()

    overlay[pred == 1] = [0, 255, 0]

    result = cv2.addWeighted(
        image_np,
        0.7,
        overlay,
        0.3,
        0
    )

    return result, tree_cover, pred