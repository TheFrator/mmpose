import numpy as np
from scipy.spatial import ConvexHull

def create_convex_hull_polygon(keypoints, indices, confidence_threshold=0.4):
    """
    Create a polygon using convex hull from the specified keypoints.
    
    Args:
        keypoints: List of [x, y, confidence] keypoints
        indices: List of indices to use from keypoints
        confidence_threshold: Minimum confidence to include a keypoint
        
    Returns:
        List of (x,y) vertices defining the polygon,
        Average confidence score of the used keypoints
    """
    # Filter keypoints by confidence
    valid_points = []
    valid_confidences = []
    for idx in indices:
        if idx < len(keypoints) and keypoints[idx][2] >= confidence_threshold:
            valid_points.append((keypoints[idx][0], keypoints[idx][1]))
            valid_confidences.append(keypoints[idx][2])
    
    # Need at least 3 points for a polygon
    if len(valid_points) < 3:
        return None, 0.0
    
    # Calculate convex hull
    points = np.array(valid_points)
    hull = ConvexHull(points)
    
    # Extract vertices in order
    vertices = [points[idx].tolist() for idx in hull.vertices]
    
    # Calculate average confidence
    avg_confidence = np.mean(valid_confidences) if valid_confidences else 0.0
    
    return vertices, float(avg_confidence)

def create_limb_polygon(keypoints, joint1_idx, joint2_idx, width_ratio=0.2, confidence_threshold=0.4):
    """
    Create a capsule-like polygon around a limb defined by two joints.
    
    Args:
        keypoints: List of [x, y, confidence] keypoints
        joint1_idx: Index of first joint
        joint2_idx: Index of second joint
        width_ratio: Width of limb relative to torso width
        confidence_threshold: Minimum confidence to include a keypoint
        
    Returns:
        List of (x,y) vertices defining the polygon,
        Average confidence score of the two joints
    """
    # Check confidence
    if (joint1_idx >= len(keypoints) or keypoints[joint1_idx][2] < confidence_threshold or 
        joint2_idx >= len(keypoints) or keypoints[joint2_idx][2] < confidence_threshold):
        return None, 0.0
    
    # Get joint coordinates
    p1 = np.array([keypoints[joint1_idx][0], keypoints[joint1_idx][1]])
    p2 = np.array([keypoints[joint2_idx][0], keypoints[joint2_idx][1]])
    
    # Calculate limb direction vector
    direction = p2 - p1
    length = np.linalg.norm(direction)
    
    # If length is too small, return None
    if length < 1:
        return None, 0.0
    
    # Normalize direction vector
    direction = direction / length
    
    # Calculate perpendicular vector
    perp = np.array([-direction[1], direction[0]])
    
    # Estimate width based on shoulder width (if available)
    width = length * width_ratio
    if 5 < len(keypoints) and 6 < len(keypoints) and keypoints[5][2] >= confidence_threshold and keypoints[6][2] >= confidence_threshold:
        shoulder_width = np.linalg.norm([keypoints[5][0] - keypoints[6][0], keypoints[5][1] - keypoints[6][1]])
        width = shoulder_width * width_ratio
    
    # Calculate four corners of the capsule
    half_width = width / 2
    p1_left = p1 + perp * half_width
    p1_right = p1 - perp * half_width
    p2_left = p2 + perp * half_width
    p2_right = p2 - perp * half_width
    
    # Return vertices in clockwise order and average confidence
    vertices = [
        p1_left.tolist(),
        p2_left.tolist(),
        p2_right.tolist(),
        p1_right.tolist()
    ]
    
    avg_confidence = (keypoints[joint1_idx][2] + keypoints[joint2_idx][2]) / 2
    
    return vertices, float(avg_confidence)

def create_torso_polygon(keypoints, top_left_idx, top_right_idx, bottom_right_idx, bottom_left_idx, 
                      confidence_threshold=0.4):
    """
    Create a quadrilateral polygon from four keypoints.
    
    Args:
        keypoints: List of [x, y, confidence] keypoints
        top_left_idx, etc.: Indices for the four corners
        confidence_threshold: Minimum confidence to include a keypoint
        
    Returns:
        List of (x,y) vertices defining the polygon,
        Average confidence score of the four corners
    """
    required_indices = [top_left_idx, top_right_idx, bottom_right_idx, bottom_left_idx]
    
    # Check if all required keypoints are available with sufficient confidence
    valid_confidences = []
    for idx in required_indices:
        if idx >= len(keypoints) or keypoints[idx][2] < confidence_threshold:
            return None, 0.0
        valid_confidences.append(keypoints[idx][2])
    
    # Create polygon vertices
    vertices = [
        [keypoints[top_left_idx][0], keypoints[top_left_idx][1]],
        [keypoints[top_right_idx][0], keypoints[top_right_idx][1]],
        [keypoints[bottom_right_idx][0], keypoints[bottom_right_idx][1]],
        [keypoints[bottom_left_idx][0], keypoints[bottom_left_idx][1]]
    ]
    
    # Calculate average confidence
    avg_confidence = np.mean(valid_confidences) if valid_confidences else 0.0
    
    return vertices, float(avg_confidence)

def create_neck_polygon(keypoints, confidence_threshold=0.4):
    """
    Create a neck polygon using shoulders and derived points below ears.
    
    Args:
        keypoints: List of [x, y, confidence] keypoints
        confidence_threshold: Minimum confidence to include a keypoint
        
    Returns:
        List of (x,y) vertices defining the polygon,
        Average confidence score of the used keypoints
    """
    # Check if required keypoints are available
    required_indices = [3, 4, 5, 6]  # Left ear, Right ear, Left shoulder, Right shoulder
    valid_confidences = []
    for idx in required_indices:
        if idx >= len(keypoints) or keypoints[idx][2] < confidence_threshold:
            return None, 0.0
        valid_confidences.append(keypoints[idx][2])
    
    # Shoulder points
    left_shoulder = (keypoints[5][0], keypoints[5][1])
    right_shoulder = (keypoints[6][0], keypoints[6][1])
    
    # Ear points
    left_ear = (keypoints[3][0], keypoints[3][1])
    right_ear = (keypoints[4][0], keypoints[4][1])
    
    # Create points below ears (for top of neck)
    # Adjust this ratio to control where the neck starts
    ratio = 0.3
    left_neck_top = [
        left_ear[0] + ratio * (left_shoulder[0] - left_ear[0]),
        left_ear[1] + ratio * (left_shoulder[1] - left_ear[1])
    ]
    right_neck_top = [
        right_ear[0] + ratio * (right_shoulder[0] - right_ear[0]),
        right_ear[1] + ratio * (right_shoulder[1] - right_ear[1])
    ]
    
    # Return vertices in clockwise order
    vertices = [left_neck_top, right_neck_top, 
                [right_shoulder[0], right_shoulder[1]], 
                [left_shoulder[0], left_shoulder[1]]]
    
    # Calculate average confidence
    avg_confidence = np.mean(valid_confidences) if valid_confidences else 0.0
    
    return vertices, float(avg_confidence)

def generate_target_areas(keypoints, confidence_threshold=0.4):
    """
    Generate all body region polygons from the keypoints.
    
    Args:
        keypoints: List of [x, y, confidence] keypoints in COCO WholeBody format
        confidence_threshold: Minimum confidence to include a keypoint
        
    Returns:
        List of target area dictionaries
    """
    target_areas = []
    
    # Head (using convex hull with face keypoints)
    head_indices = [0, 1, 2, 3, 4] + list(range(23, 91))  # Nose, Eyes, Ears + Face keypoints
    head_polygon, head_confidence = create_convex_hull_polygon(keypoints, head_indices, confidence_threshold)
    if head_polygon:
        target_areas.append({
            "region_name": "Head",
            "polygon": head_polygon,
            "confidence_score": head_confidence
        })
    
    # Neck
    neck_polygon, neck_confidence = create_neck_polygon(keypoints, confidence_threshold)
    if neck_polygon:
        target_areas.append({
            "region_name": "Neck",
            "polygon": neck_polygon,
            "confidence_score": neck_confidence
        })
    
    # Torso Upper (shoulders to hips)
    torso_upper_polygon, torso_upper_confidence = create_torso_polygon(
        keypoints, 5, 6, 12, 11, confidence_threshold
    )
    if torso_upper_polygon:
        target_areas.append({
            "region_name": "Torso_Upper",
            "polygon": torso_upper_polygon,
            "confidence_score": torso_upper_confidence
        })
    
    # For Torso_Lower and Pelvis, we need to derive additional points
    if torso_upper_polygon:
        # Create a midpoint of the hips
        hip_midpoint_x = (keypoints[11][0] + keypoints[12][0]) / 2
        hip_midpoint_y = (keypoints[11][1] + keypoints[12][1]) / 2
        
        # Create points for torso lower (hip line and points lower)
        lower_y_offset = 0.2 * abs(keypoints[5][1] - keypoints[11][1])  # 20% of torso height
        
        torso_lower_points = [
            [keypoints[11][0], keypoints[11][1]],  # Left hip
            [keypoints[12][0], keypoints[12][1]],  # Right hip
            [keypoints[12][0], keypoints[12][1] + lower_y_offset],  # Lower right
            [keypoints[11][0], keypoints[11][1] + lower_y_offset]   # Lower left
        ]
        torso_lower_confidence = (keypoints[11][2] + keypoints[12][2]) / 2
        
        target_areas.append({
            "region_name": "Torso_Lower",
            "polygon": torso_lower_points,
            "confidence_score": float(torso_lower_confidence)
        })
        
        # Create points for pelvis (below torso_lower)
        pelvis_height = 0.15 * abs(keypoints[5][1] - keypoints[11][1])  # 15% of torso height
        pelvis_points = [
            [keypoints[11][0], keypoints[11][1] + lower_y_offset],  # Upper left
            [keypoints[12][0], keypoints[12][1] + lower_y_offset],  # Upper right
            [keypoints[12][0], keypoints[12][1] + lower_y_offset + pelvis_height],  # Lower right
            [keypoints[11][0], keypoints[11][1] + lower_y_offset + pelvis_height]   # Lower left
        ]
        
        target_areas.append({
            "region_name": "Pelvis",
            "polygon": pelvis_points,
            "confidence_score": float(torso_lower_confidence)  # Same as torso_lower
        })
    
    # Arms
    left_arm_upper_polygon, left_arm_upper_confidence = create_limb_polygon(
        keypoints, 5, 7, width_ratio=0.2, confidence_threshold=confidence_threshold
    )
    if left_arm_upper_polygon:
        target_areas.append({
            "region_name": "Left_Arm_Upper",
            "polygon": left_arm_upper_polygon,
            "confidence_score": left_arm_upper_confidence
        })
    
    right_arm_upper_polygon, right_arm_upper_confidence = create_limb_polygon(
        keypoints, 6, 8, width_ratio=0.2, confidence_threshold=confidence_threshold
    )
    if right_arm_upper_polygon:
        target_areas.append({
            "region_name": "Right_Arm_Upper",
            "polygon": right_arm_upper_polygon,
            "confidence_score": right_arm_upper_confidence
        })
    
    left_arm_lower_polygon, left_arm_lower_confidence = create_limb_polygon(
        keypoints, 7, 9, width_ratio=0.18, confidence_threshold=confidence_threshold
    )
    if left_arm_lower_polygon:
        target_areas.append({
            "region_name": "Left_Arm_Lower",
            "polygon": left_arm_lower_polygon,
            "confidence_score": left_arm_lower_confidence
        })
    
    right_arm_lower_polygon, right_arm_lower_confidence = create_limb_polygon(
        keypoints, 8, 10, width_ratio=0.18, confidence_threshold=confidence_threshold
    )
    if right_arm_lower_polygon:
        target_areas.append({
            "region_name": "Right_Arm_Lower",
            "polygon": right_arm_lower_polygon,
            "confidence_score": right_arm_lower_confidence
        })
    
    # Hands (using convex hull)
    left_hand_indices = list(range(91, 112))  # 21 left hand keypoints
    left_hand_polygon, left_hand_confidence = create_convex_hull_polygon(
        keypoints, left_hand_indices, confidence_threshold
    )
    if left_hand_polygon:
        target_areas.append({
            "region_name": "Left_Hand",
            "polygon": left_hand_polygon,
            "confidence_score": left_hand_confidence
        })
    
    right_hand_indices = list(range(112, 133))  # 21 right hand keypoints
    right_hand_polygon, right_hand_confidence = create_convex_hull_polygon(
        keypoints, right_hand_indices, confidence_threshold
    )
    if right_hand_polygon:
        target_areas.append({
            "region_name": "Right_Hand",
            "polygon": right_hand_polygon,
            "confidence_score": right_hand_confidence
        })
    
    # Legs
    left_leg_upper_polygon, left_leg_upper_confidence = create_limb_polygon(
        keypoints, 11, 13, width_ratio=0.3, confidence_threshold=confidence_threshold
    )
    if left_leg_upper_polygon:
        target_areas.append({
            "region_name": "Left_Leg_Upper",
            "polygon": left_leg_upper_polygon,
            "confidence_score": left_leg_upper_confidence
        })
    
    right_leg_upper_polygon, right_leg_upper_confidence = create_limb_polygon(
        keypoints, 12, 14, width_ratio=0.3, confidence_threshold=confidence_threshold
    )
    if right_leg_upper_polygon:
        target_areas.append({
            "region_name": "Right_Leg_Upper",
            "polygon": right_leg_upper_polygon,
            "confidence_score": right_leg_upper_confidence
        })
    
    left_leg_lower_polygon, left_leg_lower_confidence = create_limb_polygon(
        keypoints, 13, 15, width_ratio=0.25, confidence_threshold=confidence_threshold
    )
    if left_leg_lower_polygon:
        target_areas.append({
            "region_name": "Left_Leg_Lower",
            "polygon": left_leg_lower_polygon,
            "confidence_score": left_leg_lower_confidence
        })
    
    right_leg_lower_polygon, right_leg_lower_confidence = create_limb_polygon(
        keypoints, 14, 16, width_ratio=0.25, confidence_threshold=confidence_threshold
    )
    if right_leg_lower_polygon:
        target_areas.append({
            "region_name": "Right_Leg_Lower",
            "polygon": right_leg_lower_polygon,
            "confidence_score": right_leg_lower_confidence
        })
    
    # Feet (using available ankle and foot keypoints)
    left_foot_indices = [15, 17, 19, 21]  # Left ankle + foot keypoints
    left_foot_polygon, left_foot_confidence = create_convex_hull_polygon(
        keypoints, left_foot_indices, confidence_threshold
    )
    if left_foot_polygon:
        target_areas.append({
            "region_name": "Left_Foot",
            "polygon": left_foot_polygon,
            "confidence_score": left_foot_confidence
        })
    
    right_foot_indices = [16, 18, 20, 22]  # Right ankle + foot keypoints
    right_foot_polygon, right_foot_confidence = create_convex_hull_polygon(
        keypoints, right_foot_indices, confidence_threshold
    )
    if right_foot_polygon:
        target_areas.append({
            "region_name": "Right_Foot",
            "polygon": right_foot_polygon,
            "confidence_score": right_foot_confidence
        })
    
    return target_areas