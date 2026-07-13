"""
create_mock_data.py — generate a tiny mock dataset under data/frames/mock_dataset
with dummy frame images to run training and evaluation smoke tests.
"""
from pathlib import Path
from PIL import Image, ImageDraw

def create_dummy_frame(path: Path, color: str, text: str):
    # Create a simple 224x224 image with solid color and text
    img = Image.new("RGB", (224, 224), color=color)
    draw = ImageDraw.Draw(img)
    draw.text((10, 100), text, fill="white")
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, "JPEG")

def main():
    root = Path("data/frames/mock_dataset")
    print(f"Generating mock dataset under: {root.resolve()}")

    # Define layout
    # split -> label -> list of video IDs
    layout = {
        "train": {
            "real": ["vid_r1", "vid_r2"],
            "fake": ["vid_f1", "vid_f2"],
        },
        "val": {
            "real": ["vid_r3"],
            "fake": ["vid_f3"],
        },
        "test": {
            "real": ["vid_r4"],
            "fake": ["vid_f4"],
        }
    }

    # Generate 8 frames per video
    total_frames = 0
    for split, labels in layout.items():
        for label, videos in labels.items():
            color = "green" if label == "real" else "red"
            for vid in videos:
                for frame_idx in range(8):
                    frame_name = f"frame_{frame_idx:04d}.jpg"
                    frame_path = root / split / label / vid / frame_name
                    create_dummy_frame(
                        frame_path, 
                        color, 
                        f"{split} | {label} | {vid} | f{frame_idx}"
                    )
                    total_frames += 1

    print(f"Successfully created {total_frames} dummy frames in mock dataset!")

if __name__ == "__main__":
    main()
