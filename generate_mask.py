from dataclasses import dataclass
from typing import List, Optional, Union, Tuple

import numpy as np
import cv2
import torch
from PIL import Image
from transformers import AutoModelForMaskGeneration, AutoProcessor
import pathlib

@dataclass
class BoundingBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def xyxy(self) -> List[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]

def mask_to_polygon(mask: np.ndarray) -> List[List[int]]:
    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area
    largest_contour = max(contours, key=cv2.contourArea)

    # Extract the vertices of the contour
    polygon = largest_contour.reshape(-1, 2).tolist()

    return polygon

def polygon_to_mask(polygon: List[Tuple[int, int]], image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Convert a polygon to a segmentation mask.

    Args:
    - polygon (list): List of (x, y) coordinates representing the vertices of the polygon.
    - image_shape (tuple): Shape of the image (height, width) for the mask.

    Returns:
    - np.ndarray: Segmentation mask with the polygon filled.
    """
    # Create an empty mask
    mask = np.zeros(image_shape, dtype=np.uint8)

    # Convert polygon to an array of points
    pts = np.array(polygon, dtype=np.int32)

    # Fill the polygon with white color (255)
    cv2.fillPoly(mask, [pts], color=(255,))

    return mask

def refine_masks(masks: torch.BoolTensor, polygon_refinement: bool = False) -> List[np.ndarray]:
    masks = masks.cpu().float()
    masks = masks.permute(0, 2, 3, 1)
    masks = masks.mean(axis=-1)
    masks = (masks > 0).int()
    masks = masks.numpy().astype(np.uint8)
    masks = list(masks)

    if polygon_refinement:
        for idx, mask in enumerate(masks):
            shape = mask.shape
            polygon = mask_to_polygon(mask)
            mask = polygon_to_mask(polygon, shape)
            masks[idx] = mask

    return masks

def click_box_points(
    image: Image.Image,
    window_name="Select Box Points (Click 4 corners, press 's' to segment, 'r' to reset, 'q' to quit)"
) -> Optional[BoundingBox]:
    """
    Open a window to let the user select 4 points to define a bounding box.
    
    Args:
        image: PIL Image
        window_name: Window title
    
    Returns:
        BoundingBox object or None if canceled
    """
    # Convert PIL image to numpy for OpenCV
    np_image = np.array(image)
    if np_image.shape[2] == 3:
        np_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
    
    # Make a copy for drawing
    display_image = np_image.copy()
    
    # Store box points
    box_points = []
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal display_image
        
        if event == cv2.EVENT_LBUTTONDOWN:  # Box corner selection
            if len(box_points) < 4:
                box_points.append([x, y])
                
                # Draw the point
                cv2.circle(display_image, (x, y), 5, (255, 0, 255), -1)
                
                # Draw lines between box points
                if len(box_points) > 1:
                    for i in range(1, len(box_points)):
                        cv2.line(display_image, 
                                (box_points[i-1][0], box_points[i-1][1]),
                                (box_points[i][0], box_points[i][1]),
                                (255, 0, 255), 2)
                
                cv2.imshow(window_name, display_image)
                print(f"Added box point {len(box_points)} at ({x}, {y})")
                
                # If we have 4 points, close the box
                if len(box_points) == 4:
                    # Connect the last point to the first
                    cv2.line(display_image, 
                            (box_points[3][0], box_points[3][1]),
                            (box_points[0][0], box_points[0][1]),
                            (255, 0, 255), 2)
                    cv2.imshow(window_name, display_image)
                    print("Box completed with 4 points, press 's' to segment with this box")
    
    # Create a window and set up callback
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    cv2.imshow(window_name, display_image)
    
    # Wait for user to select points and press 's' to segment
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):  # Quit
            cv2.destroyAllWindows()
            return None
        elif key == ord('r'):  # Reset points
            box_points = []
            display_image = np_image.copy()
            cv2.imshow(window_name, display_image)
            print("Reset box points")
        elif key == ord('s'):  # Segment - create bounding box
            # Only proceed if we have 4 box points
            if len(box_points) == 4:
                # Find min/max x and y from the 4 points
                x_coords = [p[0] for p in box_points]
                y_coords = [p[1] for p in box_points]
                xmin = min(x_coords)
                ymin = min(y_coords)
                xmax = max(x_coords)
                ymax = max(y_coords)
                
                cv2.destroyAllWindows()
                return BoundingBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
            else:
                print(f"Need 4 points to create a box (currently have {len(box_points)})")
    
    return None


def segment_with_box(
    image: Image.Image,
    box: BoundingBox,
    polygon_refinement: bool = False,
    segmenter_id: Optional[str] = None
) -> np.ndarray:
    """
    Use Segment Anything (SAM) to generate a mask given an image + a bounding box.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    segmenter_id = segmenter_id if segmenter_id is not None else "facebook/sam-vit-base"
    segmentator = AutoModelForMaskGeneration.from_pretrained(segmenter_id).to(device)
    processor = AutoProcessor.from_pretrained(segmenter_id)
    
    # Prepare inputs with only the box
    input_boxes = [[box.xyxy]]
    default_is_cuda = torch.tensor(0.).is_cuda
    if default_is_cuda:
        torch.set_default_tensor_type(torch.FloatTensor)
    try:
        inputs = processor(
            images=image,
            input_boxes=input_boxes,
            return_tensors="pt"
        )
    finally:
        if default_is_cuda:
            torch.set_default_tensor_type(torch.cuda.FloatTensor)

    model_inputs = {}
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor) and key in ("pixel_values", "input_boxes"):
            model_inputs[key] = value.to(device)
        else:
            model_inputs[key] = value

    # Generate mask
    outputs = segmentator(**model_inputs)
    
    # Process outputs
    masks = processor.post_process_masks(
        masks=outputs.pred_masks,
        original_sizes=inputs.original_sizes,
        reshaped_input_sizes=inputs.reshaped_input_sizes
    )[0]
    
    # Get the best mask
    masks = refine_masks(masks, polygon_refinement)
    return masks[0] if masks else None


def box_based_segmentation(
    image: Union[Image.Image, np.ndarray],
    polygon_refinement: bool = False,
    segmenter_id: Optional[str] = None
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Open an interactive UI to select a bounding box on the image and use SAM to generate a mask.
    
    Args:
        image: Input image (PIL Image or numpy array)
        polygon_refinement: Whether to refine the mask using polygon processing
        segmenter_id: The SAM model ID to use
    
    Returns:
        Tuple of (image as numpy array, binary mask as numpy array)
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Open UI to let user select a bounding box
    box = click_box_points(image)
    
    # If no box selected, return None for mask
    if box is None:
        return np.array(image), None
    
    # Generate mask with the box
    mask = segment_with_box(image, box, polygon_refinement, segmenter_id)
    
    return np.array(image), mask


# Function to generate binary mask using interactive box-based SAM
def generate_binary_mask_box(image, polygon_refinement=True):
    if isinstance(image, np.ndarray):
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        pil_image = image
        
    segmenter_id = "facebook/sam-vit-base"
    
    _, mask = box_based_segmentation(pil_image, polygon_refinement, segmenter_id)
    return mask


def save_box_mask_to_file(workspace, polygon_refinement=True):
    """Save a mask generated with interactive box selection to a file"""
    pathlib.Path(f'{workspace}/masks').mkdir(parents=True, exist_ok=True)
    
    image = cv2.imread(f'{workspace}/rgb/frame_0000.png')
    binary_mask = generate_binary_mask_box(image, polygon_refinement)
    
    if binary_mask is not None:
        binary_mask = (binary_mask > 0).astype(np.uint8) * 255
        cv2.imwrite(f'{workspace}/masks/frame_0000.png', binary_mask)
        print(f"Mask saved to {workspace}/masks/frame_0000.png")
    else:
        print("No mask generated")
    
    return binary_mask