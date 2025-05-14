#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RTMW-X Whole-Body Pose Estimation Script

This script processes a directory of images using the RTMW-X model for whole-body
pose estimation and saves the keypoint predictions to a JSON file.
"""

import os
import glob
import json
import logging
import traceback
import numpy as np
import cv2
from mmpose.apis import MMPoseInferencer

from limb_polygon import generate_target_areas

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rtmw_debug.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
INPUT_DIR = r"mmpose_sample\input_images"
OUTPUT_DIR = r"mmpose_sample\output"
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "rtmw-x_keypoint_detections.json")
VIS_DIR = os.path.join(OUTPUT_DIR, "visualizations")

# Create output directories if they don't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

logger.info(f"Input directory: {INPUT_DIR}")
logger.info(f"Output directory: {OUTPUT_DIR}")
logger.info(f"Visualization directory: {VIS_DIR}")

# Verify directory permissions
try:
    # Check write permissions
    test_file = os.path.join(VIS_DIR, "test_permissions.txt")
    with open(test_file, 'w') as f:
        f.write("Testing write permissions")
    os.remove(test_file)
    logger.info("Write permissions verified")
except Exception as e:
    logger.error(f"Permission error: {str(e)}")
    raise

# Initialize MMPose inferencer with RTMW-X model
try:
    inferencer = MMPoseInferencer(
        pose2d='rtmw-x_8xb320-270e_cocktail14-384x288',  # Model config name
        #pose2d_weights=r'checkpoints\rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth',
        device='cuda:0'  # Use 'cpu' if no GPU available
    )
    logger.info(f"Successfully initialized MMPose inferencer with RTMW-X model")
except Exception as e:
    logger.error(f"Failed to initialize inferencer: {str(e)}")
    logger.error(traceback.format_exc())
    raise

# Get all image files from the input directory
supported_formats = ['.jpg', '.jpeg', '.png', '.bmp']
image_files = []
for ext in supported_formats:
    image_files.extend(glob.glob(os.path.join(INPUT_DIR, f"*{ext}")))
    image_files.extend(glob.glob(os.path.join(INPUT_DIR, f"*{ext.upper()}")))

# Sort image files for consistent processing
image_files.sort()
logger.info(f"Found {len(image_files)} images in {INPUT_DIR}")

# Prepare output data structure
output_data = []

# Process each image
for img_idx, image_path in enumerate(image_files):
    img_name = os.path.basename(image_path)
    logger.info(f"Processing image {img_idx+1}/{len(image_files)}: {img_name}")
    
    try:
        # Run inference for keypoint detection
        result_generator = inferencer(image_path, return_datasamples=True)
        result = next(result_generator)
        # Exhaust the generator to ensure processing completes
        for _ in result_generator:
            pass
        
        predictions = result['predictions'][0]  # Get the data sample
        
        # Prepare data for this image
        image_data = {
            "image_filename": img_name,
            "persons": []
        }
        
        # Process each person detected in the image
        if hasattr(predictions, 'pred_instances'):
            # Extract keypoints and scores
            if isinstance(predictions.pred_instances.keypoints, np.ndarray):
                keypoints = predictions.pred_instances.keypoints
            else:
                keypoints = predictions.pred_instances.keypoints.cpu().numpy()

            if isinstance(predictions.pred_instances.keypoint_scores, np.ndarray):
                keypoint_scores = predictions.pred_instances.keypoint_scores
            else:
                keypoint_scores = predictions.pred_instances.keypoint_scores.cpu().numpy()
            
            # Extract bounding boxes if available
            bboxes = None
            if hasattr(predictions.pred_instances, 'bboxes'):
                if isinstance(predictions.pred_instances.bboxes, np.ndarray):
                    bboxes = predictions.pred_instances.bboxes
                else:
                    bboxes = predictions.pred_instances.bboxes.cpu().numpy()
            
            # Process each person
            for person_idx in range(len(keypoints)):
                person_data = {
                    "person_id": person_idx + 1,
                    "keypoints": []
                }
                
                # Format keypoints as [x, y, score]
                for kpt_idx in range(len(keypoints[person_idx])):
                    x, y = keypoints[person_idx, kpt_idx]
                    score = float(keypoint_scores[person_idx, kpt_idx])
                    person_data["keypoints"].append([float(x), float(y), score])
                
                # Add bounding box if available
                if bboxes is not None:
                    x1, y1, x2, y2 = bboxes[person_idx]
                    person_data["bbox"] = [
                        float(x1), 
                        float(y1), 
                        float(x2 - x1),  # width
                        float(y2 - y1)   # height
                    ]
                
                person_data["target_areas"] = generate_target_areas(person_data["keypoints"], confidence_threshold=0.3)
                # Add person to image data
                image_data["persons"].append(person_data)
        
        # Add image data to output
        output_data.append(image_data)
        
        # Generate and explicitly save visualization for this image
        logger.info(f"Generating visualization for {img_name}")
        vis_file_path = os.path.join(VIS_DIR, img_name)
        
        try:
            # Request visualization with return_vis=True to get the visualization data
            vis_generator = inferencer(
                image_path, 
                vis_out_dir=VIS_DIR,
                return_vis=True,
                show=False,
                draw_bbox=True,
                kpt_thr=0.5
            )
            
            # Process the visualization generator
            vis_results = []
            for vis_result in vis_generator:
                if 'visualization' in vis_result and vis_result['visualization']:
                    vis_results.extend(vis_result['visualization'])
            
            # Check if the visualization was saved automatically
            if os.path.exists(vis_file_path):
                logger.info(f"Visualization automatically saved to {vis_file_path}")
            else:
                # Save the visualization manually if we have results
                if vis_results:
                    # Convert from RGB to BGR for OpenCV
                    vis_img = cv2.cvtColor(vis_results[0], cv2.COLOR_RGB2BGR)
                    cv2.imwrite(vis_file_path, vis_img)
                    logger.info(f"Visualization manually saved to {vis_file_path}")
                    
                    if os.path.exists(vis_file_path):
                        logger.info(f"Confirmed visualization file exists at {vis_file_path}")
                    else:
                        logger.error(f"Failed to save visualization to {vis_file_path}")
                else:
                    logger.warning(f"No visualization data returned for {img_name}")
                    
        except Exception as vis_err:
            logger.error(f"Error generating visualization for {img_name}: {str(vis_err)}")
            logger.error(traceback.format_exc())
    
    except Exception as e:
        logger.error(f"Error processing image {img_name}: {str(e)}")
        logger.error(traceback.format_exc())

# Save output to JSON file
try:
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(output_data, f, indent=2)
    logger.info(f"Results successfully saved to {OUTPUT_JSON}")
except Exception as e:
    logger.error(f"Error saving results to JSON: {str(e)}")
    logger.error(traceback.format_exc())

# Final check on visualizations
vis_files = [f for f in os.listdir(VIS_DIR) if os.path.isfile(os.path.join(VIS_DIR, f))]
logger.info(f"Total visualization files saved: {len(vis_files)}")
if len(vis_files) != len(image_files):
    logger.warning(f"Number of visualization files ({len(vis_files)}) does not match number of input images ({len(image_files)})")
    # List some of the visualization files as a sample
    if vis_files:
        logger.info(f"Sample visualization files: {vis_files[:min(5, len(vis_files))]}")

logger.info(f"Processing complete!")
logger.info(f"Results saved to {OUTPUT_JSON}")
logger.info(f"Visualizations saved to {VIS_DIR}")