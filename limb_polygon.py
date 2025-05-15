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

def create_improved_head_polygon(keypoints, person_bbox=None, confidence_threshold=0.4):
    """
    Create an improved head polygon using facial landmarks, elliptical fitting,
    and person bounding box information.
    
    Args:
        keypoints: List of [x, y, confidence] keypoints in COCO-WholeBody format
        person_bbox: Optional [x1, y1, width, height] representing the person's bounding box
        confidence_threshold: Minimum confidence to include a keypoint
        
    Returns:
        List of (x,y) vertices defining the polygon,
        Average confidence score of the used keypoints
    """
    # Check if we have enough face keypoints
    face_indices = [0, 1, 2, 3, 4] + list(range(23, 91))  # Nose, eyes, ears + face
    jawline_indices = list(range(23, 32))  # Approximate jawline points
    
    # Filter valid keypoints
    valid_face_points = []
    valid_face_confidences = []
    for idx in face_indices:
        if idx < len(keypoints) and keypoints[idx][2] >= confidence_threshold:
            valid_face_points.append((keypoints[idx][0], keypoints[idx][1]))
            valid_face_confidences.append(keypoints[idx][2])
    
    # Check if we have enough points
    if len(valid_face_points) < 3:
        return None, 0.0
    
    # Determine head orientation (frontal vs profile)
    is_profile = False
    left_ear_conf = 0.0
    right_ear_conf = 0.0
    if 3 < len(keypoints) and 4 < len(keypoints):  # Check if ears are detected
        left_ear_conf = keypoints[3][2]
        right_ear_conf = keypoints[4][2]
        # If one ear is significantly more visible than the other, it's likely a profile view
        if abs(left_ear_conf - right_ear_conf) > 0.3:
            is_profile = True
    
    # Extract jawline points if available
    jawline_points = []
    for idx in jawline_indices:
        if idx < len(keypoints) and keypoints[idx][2] >= confidence_threshold:
            jawline_points.append((keypoints[idx][0], keypoints[idx][1]))
    
    # If we have enough jawline points, use them to define the lower part of the head
    final_points = []
    if len(jawline_points) >= 3:
        final_points.extend(jawline_points)
    
    # Add eyes, nose, and ears with high confidence
    for idx in [0, 1, 2, 3, 4]:  # Nose, eyes, ears
        if idx < len(keypoints) and keypoints[idx][2] >= confidence_threshold + 0.1:  # Higher threshold
            final_points.append((keypoints[idx][0], keypoints[idx][1]))
    
    # Check if we have enough points for a polygon
    if len(final_points) < 3:
        # Fall back to using all face points
        final_points = valid_face_points
    
    # Convert to numpy array for calculations
    points_array = np.array(final_points)
    
    # Cranium points (top of head)
    top_center = None
    top_left = None
    top_right = None
    
    # Estimate top of head if possible
    if 1 < len(keypoints) and 2 < len(keypoints):  # Check if eyes are detected
        left_eye = np.array([keypoints[1][0], keypoints[1][1]])
        right_eye = np.array([keypoints[2][0], keypoints[2][1]])
        eye_center = (left_eye + right_eye) / 2
        
        # Estimate face height
        if 0 < len(keypoints) and keypoints[0][2] >= confidence_threshold:  # Nose
            nose = np.array([keypoints[0][0], keypoints[0][1]])
            face_height = 0
            
            # If we have jawline points, use the lowest point
            if jawline_points:
                jaw_y = max(p[1] for p in jawline_points)
                face_height = jaw_y - min(left_eye[1], right_eye[1])
            else:
                # Rough estimate based on eyes to nose
                face_height = abs(nose[1] - min(left_eye[1], right_eye[1])) * 2
            
            # Estimate top of head
            if face_height > 0:
                # Head height is approximately 1.3 times face height
                head_height = face_height * 1.3
                # Create points for top of head
                top_center = np.array([eye_center[0], eye_center[1] - head_height * 0.7])
                
                # Add more points to create a realistic cranium shape
                if is_profile:
                    # For profile view, add points to create a curved top
                    side_offset = head_height * 0.25
                    if left_ear_conf > right_ear_conf:  # Left profile
                        top_left = np.array([top_center[0] - side_offset, top_center[1] + side_offset * 0.5])
                        top_right = np.array([top_center[0] + side_offset * 0.8, top_center[1] + side_offset * 0.3])
                    else:  # Right profile
                        top_left = np.array([top_center[0] - side_offset * 0.8, top_center[1] + side_offset * 0.3])
                        top_right = np.array([top_center[0] + side_offset, top_center[1] + side_offset * 0.5])
                else:
                    # For frontal view, create a curved top
                    side_offset = head_height * 0.3
                    top_left = np.array([top_center[0] - side_offset, top_center[1] + side_offset * 0.5])
                    top_right = np.array([top_center[0] + side_offset, top_center[1] + side_offset * 0.5])
                
                # NEW: Constraining top of head to the bounding box (if provided)
                if person_bbox is not None:
                    # Add a small padding (2 pixels) to avoid clipping exactly at the bbox edge
                    bbox_top = person_bbox[1] + 2  
                    
                    # Adjust top_center if it extends beyond the bbox
                    if top_center[1] < bbox_top:
                        top_center[1] = bbox_top
                    
                    # Adjust top_left if it extends beyond the bbox
                    if top_left is not None and top_left[1] < bbox_top:
                        top_left[1] = bbox_top
                    
                    # Adjust top_right if it extends beyond the bbox
                    if top_right is not None and top_right[1] < bbox_top:
                        top_right[1] = bbox_top
                
                # Add cranium points to final points
                if top_center is not None:
                    final_points.append((top_center[0], top_center[1]))
                if top_left is not None:
                    final_points.append((top_left[0], top_left[1]))
                if top_right is not None:
                    final_points.append((top_right[0], top_right[1]))
    
    # NEW: Adapt head polygon shape based on proximity to bounding box sides
    if person_bbox is not None:
        # Determine which side the head is facing
        facing_left = False
        facing_right = False
        
        # Use ears to determine head orientation if available
        if 3 < len(keypoints) and 4 < len(keypoints):
            left_ear = np.array([keypoints[3][0], keypoints[3][1]]) if keypoints[3][2] >= confidence_threshold else None
            right_ear = np.array([keypoints[4][0], keypoints[4][1]]) if keypoints[4][2] >= confidence_threshold else None
            
            # Check left ear proximity to left bbox edge
            if left_ear is not None:
                left_dist = left_ear[0] - person_bbox[0]
                facing_left = left_dist < 0.3 * person_bbox[2]  # If left ear is within 30% of bbox width from left edge
            
            # Check right ear proximity to right bbox edge
            if right_ear is not None:
                right_dist = (person_bbox[0] + person_bbox[2]) - right_ear[0]
                facing_right = right_dist < 0.3 * person_bbox[2]  # If right ear is within 30% of bbox width from right edge
            
            # If both ears are detected, use confidence to decide
            if left_ear is not None and right_ear is not None:
                if left_ear_conf > right_ear_conf + 0.3:
                    facing_left = True
                    facing_right = False
                elif right_ear_conf > left_ear_conf + 0.3:
                    facing_right = True
                    facing_left = False
        
        # If ear detection inconclusive, use nose and eye positions (if available)
        if not (facing_left or facing_right) and 0 < len(keypoints) and keypoints[0][2] >= confidence_threshold:
            nose = np.array([keypoints[0][0], keypoints[0][1]])
            
            # Determine if nose is significantly off-center
            bbox_center_x = person_bbox[0] + person_bbox[2] / 2
            nose_offset = (nose[0] - bbox_center_x) / person_bbox[2]  # Normalized offset
            
            if nose_offset < -0.15:  # Nose is more than 15% left of center
                facing_left = True
            elif nose_offset > 0.15:  # Nose is more than 15% right of center
                facing_right = True
        
        # Add points to extend the head polygon towards the dominant side
        if facing_left or facing_right:
            # Find extremes in current points
            if final_points:
                points_array = np.array(final_points)
                min_x = points_array[:, 0].min()
                max_x = points_array[:, 0].max()
                min_y = points_array[:, 1].min()
                max_y = points_array[:, 1].max()
                
                # Find a representative y-coordinate for the side of the head
                # (halfway between the top and bottom of the current points)
                mid_y = (min_y + max_y) / 2
                
                # Add padding to stay slightly inside the bbox (3 pixels)
                bbox_padding = 3
                
                if facing_left:
                    # Find the leftmost point's x-coordinate
                    # Don't extend beyond the bbox left edge
                    left_edge = max(person_bbox[0] + bbox_padding, min_x - person_bbox[2] * 0.05)
                    
                    # Add points to extend the left side of the head
                    # Add points at multiple heights to create a smoother side profile
                    side_heights = [min_y + (max_y - min_y) * h for h in [0.25, 0.5, 0.75]]
                    for side_y in side_heights:
                        final_points.append((left_edge, side_y))
                
                if facing_right:
                    # Find the rightmost point's x-coordinate
                    # Don't extend beyond the bbox right edge
                    right_edge = min(person_bbox[0] + person_bbox[2] - bbox_padding, max_x + person_bbox[2] * 0.05)
                    
                    # Add points to extend the right side of the head
                    # Add points at multiple heights to create a smoother side profile
                    side_heights = [min_y + (max_y - min_y) * h for h in [0.25, 0.5, 0.75]]
                    for side_y in side_heights:
                        final_points.append((right_edge, side_y))
    
    # Recalculate points_array after all modifications
    points_array = np.array(final_points)
    
    # Calculate convex hull with final points
    hull = ConvexHull(points_array)
    
    # Extract vertices
    vertices = [points_array[idx].tolist() for idx in hull.vertices]
    
    # Apply expansion factor for hair and accessories
    centroid = np.mean(points_array, axis=0)
    expanded_vertices = []
    for vertex in vertices:
        # Vector from centroid to vertex
        direction = np.array(vertex) - centroid
        # Scale by expansion factor
        expansion_factor = 1.07  # 7% expansion
        expanded_vertex = centroid + direction * expansion_factor
        expanded_vertices.append(expanded_vertex.tolist())
    
    # Calculate average confidence score
    avg_confidence = np.mean(valid_face_confidences) if valid_face_confidences else 0.0
    
    return expanded_vertices, float(avg_confidence)


def create_improved_neck_polygon(keypoints, head_polygon=None, confidence_threshold=0.4):
    """
    Create an improved neck polygon using anatomically guided proportions.
    
    Args:
        keypoints: List of [x, y, confidence] keypoints
        head_polygon: Optional head polygon to ensure no overlap
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
    
    # Get key points
    left_ear = np.array([keypoints[3][0], keypoints[3][1]])
    right_ear = np.array([keypoints[4][0], keypoints[4][1]])
    left_shoulder = np.array([keypoints[5][0], keypoints[5][1]])
    right_shoulder = np.array([keypoints[6][0], keypoints[6][1]])
    
    # Calculate shoulder width and other anatomical proportions
    shoulder_width = np.linalg.norm(right_shoulder - left_shoulder)
    shoulder_midpoint = (left_shoulder + right_shoulder) / 2
    
    # Determine head orientation
    is_profile = False
    left_ear_conf = keypoints[3][2]
    right_ear_conf = keypoints[4][2]
    if abs(left_ear_conf - right_ear_conf) > 0.3:
        is_profile = True
    
    # Calculate rotation angle for head
    rotation_angle = 0
    if left_ear_conf > confidence_threshold and right_ear_conf > confidence_threshold:
        rotation_angle = np.arctan2(right_ear[1] - left_ear[1], right_ear[0] - left_ear[0])
    
    # Neck width should be around 60-70% of shoulder width at midpoint
    neck_width_ratio = 0.65
    neck_width = shoulder_width * neck_width_ratio
    
    # Estimate neck length (approximately 1/3 of the distance from shoulders to top of head)
    # Since we don't know the exact top of head, we'll use ears and add some offset
    ears_midpoint = (left_ear + right_ear) / 2
    est_neck_length = np.linalg.norm(shoulder_midpoint - ears_midpoint) * 0.6
    
    # Create neck top points (just below the jawline/ears)
    # We'll calculate these relative to the ears but lower
    neck_top_offset_y = est_neck_length * 0.2  # 20% down from ears
    
    # Adjust based on head orientation
    if is_profile:
        # For profile view, adjust the neck width
        if left_ear_conf > right_ear_conf:  # Left profile
            dominant_ear = left_ear
            dominant_shoulder = left_shoulder
        else:  # Right profile
            dominant_ear = right_ear
            dominant_shoulder = right_shoulder
        
        # Create neck top points with profile orientation
        neck_vector = dominant_shoulder - dominant_ear
        neck_length = np.linalg.norm(neck_vector) * 0.4
        neck_direction = neck_vector / np.linalg.norm(neck_vector)
        
        # Create top of neck points
        neck_top_center = dominant_ear + neck_direction * neck_length * 0.3
        neck_width_profile = neck_width * 0.8  # Narrower for profile
        
        # Create perpendicular vector for width
        perp_vector = np.array([-neck_direction[1], neck_direction[0]])
        
        # Neck top points
        left_neck_top = neck_top_center - perp_vector * neck_width_profile * 0.5
        right_neck_top = neck_top_center + perp_vector * neck_width_profile * 0.5
    else:
        # For frontal view, create a more standard trapezoid
        # Apply rotation if head is tilted
        if rotation_angle != 0:
            # Create rotation matrix
            cos_rot = np.cos(rotation_angle)
            sin_rot = np.sin(rotation_angle)
            rot_matrix = np.array([[cos_rot, -sin_rot], [sin_rot, cos_rot]])
            
            # Create neck top width vector with rotation
            neck_top_width_vector = rot_matrix.dot(np.array([neck_width * 0.8 / 2, 0]))
            
            # Create neck top center
            neck_top_center = np.array([
                (left_ear[0] + right_ear[0]) / 2,
                (left_ear[1] + right_ear[1]) / 2 + neck_top_offset_y
            ])
            
            # Create neck top points
            left_neck_top = neck_top_center - neck_top_width_vector
            right_neck_top = neck_top_center + neck_top_width_vector
        else:
            # Without rotation, use a simpler approach
            left_neck_top = np.array([
                left_ear[0] + (left_shoulder[0] - left_ear[0]) * 0.3,
                left_ear[1] + (left_shoulder[1] - left_ear[1]) * 0.3
            ])
            right_neck_top = np.array([
                right_ear[0] + (right_shoulder[0] - right_ear[0]) * 0.3,
                right_ear[1] + (right_shoulder[1] - right_ear[1]) * 0.3
            ])
    
    # Check if the head polygon is provided to ensure no overlap
    if head_polygon and len(head_polygon) > 2:
        # Find the lowest point of the head polygon
        head_bottom_y = max(p[1] for p in head_polygon)
        
        # Ensure neck top points are below the head bottom
        if left_neck_top[1] < head_bottom_y:
            left_neck_top[1] = head_bottom_y + 1
        if right_neck_top[1] < head_bottom_y:
            right_neck_top[1] = head_bottom_y + 1
    
    # Return vertices in clockwise order
    vertices = [
        left_neck_top.tolist(),
        right_neck_top.tolist(),
        right_shoulder.tolist(),
        left_shoulder.tolist()
    ]
    
    # Calculate average confidence
    avg_confidence = np.mean(valid_confidences) if valid_confidences else 0.0
    
    return vertices, float(avg_confidence)

def create_improved_lower_torso_polygon(keypoints, confidence_threshold=0.4):
    """
    Create an improved lower torso polygon using body-specific proportions.
    
    Args:
        keypoints: List of [x, y, confidence] keypoints
        confidence_threshold: Minimum confidence to include a keypoint
        
    Returns:
        List of (x,y) vertices defining the polygon,
        Average confidence score of the used keypoints
    """
    # Check if required keypoints are available
    required_indices = [5, 6, 11, 12]  # Shoulders and hips
    valid_confidences = []
    for idx in required_indices:
        if idx >= len(keypoints) or keypoints[idx][2] < confidence_threshold:
            return None, 0.0
        valid_confidences.append(keypoints[idx][2])
    
    # Get key points
    left_shoulder = np.array([keypoints[5][0], keypoints[5][1]])
    right_shoulder = np.array([keypoints[6][0], keypoints[6][1]])
    left_hip = np.array([keypoints[11][0], keypoints[11][1]])
    right_hip = np.array([keypoints[12][0], keypoints[12][1]])
    
    # Calculate body proportions
    shoulder_width = np.linalg.norm(right_shoulder - left_shoulder)
    hip_width = np.linalg.norm(right_hip - left_hip)
    torso_height = np.linalg.norm((left_shoulder + right_shoulder) / 2 - (left_hip + right_hip) / 2)
    
    # Calculate midpoints
    shoulder_midpoint = (left_shoulder + right_shoulder) / 2
    hip_midpoint = (left_hip + right_hip) / 2
    
    # Infer lower rib points (approximately 60% down from shoulders to hips)
    lower_rib_factor = 0.6
    lower_rib_left = left_shoulder + (left_hip - left_shoulder) * lower_rib_factor
    lower_rib_right = right_shoulder + (right_hip - right_shoulder) * lower_rib_factor
    
    # Calculate waist points (approximately 75% down from shoulders to hips)
    waist_factor = 0.75
    waist_left = left_shoulder + (left_hip - left_shoulder) * waist_factor
    waist_right = right_shoulder + (right_hip - right_shoulder) * waist_factor
    
    # Calculate waist width ratio (typically narrower than shoulders and hips)
    # Use confidence-weighted approach
    shoulder_conf = (keypoints[5][2] + keypoints[6][2]) / 2
    hip_conf = (keypoints[11][2] + keypoints[12][2]) / 2
    
    if shoulder_conf > 0 and hip_conf > 0:
        waist_width_factor = (shoulder_width * shoulder_conf + hip_width * hip_conf) / (shoulder_conf + hip_conf)
        waist_width_factor *= 0.85  # Waist is typically narrower
    else:
        waist_width_factor = (shoulder_width + hip_width) / 2 * 0.85
    
    # Adjust waist points based on calculated width
    waist_mid = (waist_left + waist_right) / 2
    waist_dir = waist_right - waist_left
    waist_dir_norm = waist_dir / np.linalg.norm(waist_dir)
    waist_new_half_width = waist_width_factor / 2
    
    waist_left_adjusted = waist_mid - waist_dir_norm * waist_new_half_width
    waist_right_adjusted = waist_mid + waist_dir_norm * waist_new_half_width
    
    # Define torso lower polygon (from lower ribs to top of iliac crests/hips)
    vertices = [
        lower_rib_left.tolist(),
        lower_rib_right.tolist(),
        waist_right_adjusted.tolist(),
        waist_left_adjusted.tolist()
    ]
    
    # Calculate average confidence
    avg_confidence = np.mean(valid_confidences)
    
    return vertices, float(avg_confidence)

def create_improved_pelvis_polygon(keypoints, confidence_threshold=0.4):
    """
    Create an improved pelvis polygon using anatomical landmark inference.
    
    Args:
        keypoints: List of [x, y, confidence] keypoints
        confidence_threshold: Minimum confidence to include a keypoint
        
    Returns:
        List of (x,y) vertices defining the polygon,
        Average confidence score of the used keypoints
    """
    # Check if required keypoints are available
    required_indices = [11, 12, 13, 14]  # Hips and knees
    valid_confidences = []
    for idx in required_indices:
        if idx >= len(keypoints) or keypoints[idx][2] < confidence_threshold:
            return None, 0.0
        valid_confidences.append(keypoints[idx][2])
    
    # Get key points
    left_hip = np.array([keypoints[11][0], keypoints[11][1]])
    right_hip = np.array([keypoints[12][0], keypoints[12][1]])
    left_knee = np.array([keypoints[13][0], keypoints[13][1]])
    right_knee = np.array([keypoints[14][0], keypoints[14][1]])
    
    # Calculate hip width
    hip_width = np.linalg.norm(right_hip - left_hip)
    hip_midpoint = (left_hip + right_hip) / 2
    
    # Infer ASIS (anterior superior iliac spine) points
    # Typically slightly above and outward from hip keypoints
    asis_offset_y = hip_width * 0.05  # Upward offset
    asis_offset_x = hip_width * 0.08  # Outward offset
    
    left_asis = np.array([
        left_hip[0] - asis_offset_x,
        left_hip[1] - asis_offset_y
    ])
    
    right_asis = np.array([
        right_hip[0] + asis_offset_x,
        right_hip[1] - asis_offset_y
    ])
    
    # Infer pubic symphysis position
    # Typically centered between hip keypoints and below
    pubic_offset_y = hip_width * 0.12  # Downward offset
    pubic_symphysis = np.array([
        hip_midpoint[0],
        hip_midpoint[1] + pubic_offset_y
    ])
    
    # Calculate iliofemoral paths (from ASIS toward proximal thigh)
    # This connects the pelvis to the upper thigh
    thigh_factor = 0.25  # How far down the thigh to extend
    
    left_thigh_vector = left_knee - left_hip
    left_iliofemoral = left_hip + left_thigh_vector * thigh_factor
    
    right_thigh_vector = right_knee - right_hip
    right_iliofemoral = right_hip + right_thigh_vector * thigh_factor
    
    # Construct polygon vertices
    # Format: ASIS left, ASIS right, iliofemoral right, pubic symphysis, iliofemoral left
    vertices = [
        left_asis.tolist(),
        right_asis.tolist(),
        right_iliofemoral.tolist(),
        pubic_symphysis.tolist(),
        left_iliofemoral.tolist()
    ]
    
    # Calculate average confidence
    avg_confidence = np.mean(valid_confidences)
    
    return vertices, float(avg_confidence)

def create_improved_foot_polygon(keypoints, foot_indices, is_left=True, confidence_threshold=0.4):
    """
    Create an improved foot polygon using ankle, big toe, small toe, and heel keypoints.
    
    Args:
        keypoints: List of [x, y, confidence] keypoints
        foot_indices: List of indices for this foot's keypoints
        is_left: Boolean indicating if this is the left foot
        confidence_threshold: Minimum confidence to include a keypoint
        
    Returns:
        List of (x,y) vertices defining the polygon,
        Average confidence score of the used keypoints
    """
    # Filter valid points
    valid_points = []
    valid_confidences = []
    for idx in foot_indices:
        if idx < len(keypoints) and keypoints[idx][2] >= confidence_threshold:
            valid_points.append((keypoints[idx][0], keypoints[idx][1]))
            valid_confidences.append(keypoints[idx][2])
    
    # Need at least 3 points for a polygon
    if len(valid_points) < 3:
        return None, 0.0
    
    # Extract specific keypoints if available
    ankle = None
    big_toe = None
    small_toe = None
    heel = None
    
    # For left foot: ankle=15, big_toe=17, small_toe=18, heel=19
    # For right foot: ankle=16, big_toe=20, small_toe=21, heel=22
    if is_left:
        if 15 < len(keypoints) and keypoints[15][2] >= confidence_threshold:
            ankle = np.array([keypoints[15][0], keypoints[15][1]])
        if 17 < len(keypoints) and keypoints[17][2] >= confidence_threshold:
            big_toe = np.array([keypoints[17][0], keypoints[17][1]])
        if 18 < len(keypoints) and keypoints[18][2] >= confidence_threshold:
            small_toe = np.array([keypoints[18][0], keypoints[18][1]])
        if 19 < len(keypoints) and keypoints[19][2] >= confidence_threshold:
            heel = np.array([keypoints[19][0], keypoints[19][1]])
    else:
        if 16 < len(keypoints) and keypoints[16][2] >= confidence_threshold:
            ankle = np.array([keypoints[16][0], keypoints[16][1]])
        if 20 < len(keypoints) and keypoints[20][2] >= confidence_threshold:
            big_toe = np.array([keypoints[20][0], keypoints[20][1]])
        if 21 < len(keypoints) and keypoints[21][2] >= confidence_threshold:
            small_toe = np.array([keypoints[21][0], keypoints[21][1]])
        if 22 < len(keypoints) and keypoints[22][2] >= confidence_threshold:
            heel = np.array([keypoints[22][0], keypoints[22][1]])
    
    # If we have all the specific points, create a more anatomically correct foot
    final_points = []
    
    if ankle is not None and heel is not None:
        # Add ankle and heel
        final_points.append(ankle.tolist())
        final_points.append(heel.tolist())
        
        # Add points to create the arch of the foot
        ankle_to_heel = heel - ankle
        ankle_heel_midpoint = ankle + ankle_to_heel * 0.5
        # Create a point for the arch (inward from the midpoint)
        arch_offset = np.array([ankle_to_heel[1], -ankle_to_heel[0]])
        arch_offset = arch_offset / np.linalg.norm(arch_offset) * np.linalg.norm(ankle_to_heel) * 0.2
        arch_point = ankle_heel_midpoint + arch_offset * (-1 if is_left else 1)
        final_points.append(arch_point.tolist())
    
    # Add toe points
    if big_toe is not None:
        final_points.append(big_toe.tolist())
    if small_toe is not None:
        final_points.append(small_toe.tolist())
    
    # If we don't have enough specific points, fall back to convex hull
    if len(final_points) < 3:
        points_array = np.array(valid_points)
        hull = ConvexHull(points_array)
        vertices = [points_array[idx].tolist() for idx in hull.vertices]
    else:
        # Use the improved foot shape
        # Sort points to form a proper polygon
        if ankle is not None and (big_toe is not None or small_toe is not None):
            # Sort based on angle from ankle to each point
            center = ankle
            def angle_from_center(point):
                return np.arctan2(point[1] - center[1], point[0] - center[0])
            
            sorted_points = sorted(final_points, key=angle_from_center)
            vertices = sorted_points
        else:
            # If we don't have ankle or toes, just use the points as is
            vertices = final_points
    
    # Add a small expansion factor
    if vertices:
        centroid = np.mean(np.array(vertices), axis=0)
        expanded_vertices = []
        for vertex in vertices:
            # Vector from centroid to vertex
            direction = np.array(vertex) - centroid
            # Scale by expansion factor
            expansion_factor = 1.05  # 5% expansion
            expanded_vertex = centroid + direction * expansion_factor
            expanded_vertices.append(expanded_vertex.tolist())
        vertices = expanded_vertices
    
    # Calculate average confidence
    avg_confidence = np.mean(valid_confidences) if valid_confidences else 0.0
    
    return vertices, float(avg_confidence)

def generate_target_areas(keypoints, person_bbox = None, confidence_threshold=0.4):
    """
    Generate all body region polygons from the keypoints with improved anatomical accuracy.
    
    Args:
        keypoints: List of [x, y, confidence] keypoints in COCO WholeBody format
        confidence_threshold: Minimum confidence to include a keypoint
        
    Returns:
        List of target area dictionaries
    """
    target_areas = []
    
    # Improved Head
    head_polygon, head_confidence = create_improved_head_polygon(keypoints, person_bbox, confidence_threshold)
    if head_polygon:
        target_areas.append({
            "region_name": "Head",
            "polygon": head_polygon,
            "confidence_score": head_confidence
        })
    
    # Improved Neck - we pass the head polygon to avoid overlap
    neck_polygon, neck_confidence = create_improved_neck_polygon(keypoints, head_polygon, confidence_threshold)
    if neck_polygon:
        target_areas.append({
            "region_name": "Neck",
            "polygon": neck_polygon,
            "confidence_score": neck_confidence
        })
    
    # Torso Upper (shoulders to hips) - using existing function for now
    torso_upper_polygon, torso_upper_confidence = create_torso_polygon(
        keypoints, 5, 6, 12, 11, confidence_threshold
    )
    if torso_upper_polygon:
        target_areas.append({
            "region_name": "Torso_Upper",
            "polygon": torso_upper_polygon,
            "confidence_score": torso_upper_confidence
        })
    
    # Improved Lower Torso
    lower_torso_polygon, lower_torso_confidence = create_improved_lower_torso_polygon(keypoints, confidence_threshold)
    if lower_torso_polygon:
        target_areas.append({
            "region_name": "Torso_Lower",
            "polygon": lower_torso_polygon,
            "confidence_score": lower_torso_confidence
        })
    
    # Improved Pelvis
    pelvis_polygon, pelvis_confidence = create_improved_pelvis_polygon(keypoints, confidence_threshold)
    if pelvis_polygon:
        target_areas.append({
            "region_name": "Pelvis",
            "polygon": pelvis_polygon,
            "confidence_score": pelvis_confidence
        })
    
    # Arms - using existing functions for now
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
    
    # Hands - using existing functions for now
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
    
    # Legs - using existing functions for now
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
    
    # Improved Feet
    left_foot_indices = [15, 17, 18, 19]  # Left ankle, big toe, small toe, heel
    left_foot_polygon, left_foot_confidence = create_improved_foot_polygon(
        keypoints, left_foot_indices, is_left=True, confidence_threshold=confidence_threshold
    )
    if left_foot_polygon:
        target_areas.append({
            "region_name": "Left_Foot",
            "polygon": left_foot_polygon,
            "confidence_score": left_foot_confidence
        })
    
    right_foot_indices = [16, 20, 21, 22]  # Right ankle, big toe, small toe, heel
    right_foot_polygon, right_foot_confidence = create_improved_foot_polygon(
        keypoints, right_foot_indices, is_left=False, confidence_threshold=confidence_threshold
    )
    if right_foot_polygon:
        target_areas.append({
            "region_name": "Right_Foot",
            "polygon": right_foot_polygon,
            "confidence_score": right_foot_confidence
        })
    
    return target_areas