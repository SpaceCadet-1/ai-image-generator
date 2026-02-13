"""Lightweight face detection and embedding extraction using ONNX Runtime.

Replaces the `insightface` Python package (which requires MSVC to build on Windows)
with direct ONNX model inference. Uses the same antelopev2 models:
  - det_10g.onnx (SCRFD face detector) → bounding boxes + 5 landmarks
  - w600k_r50.onnx (ArcFace recognizer) → 512-dim face embedding

Only two public functions:
  - load(model_dir, providers) → initializes the ONNX sessions
  - get_face_embedding(image) → returns the 512-dim normed embedding
"""

import cv2
import numpy as np
import onnxruntime

# Module-level ONNX sessions
_det_session = None
_rec_session = None

# ArcFace alignment reference points for 112x112 crop
_REFERENCE_LANDMARKS = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

# SCRFD detection settings
_DET_INPUT_SIZE = (640, 640)
_NMS_THRESHOLD = 0.4
_SCORE_THRESHOLD = 0.5
_FMC = 3  # Feature map count
_FEAT_STRIDE = [8, 16, 32]
_NUM_ANCHORS = 2


def load(model_dir: str, providers: list[str] | None = None):
    """Load ONNX models for face detection and recognition.

    Args:
        model_dir: Path to directory containing antelopev2 ONNX models.
        providers: ONNX Runtime execution providers.
    """
    global _det_session, _rec_session

    if providers is None:
        providers = ["CPUExecutionProvider"]

    from pathlib import Path

    model_path = Path(model_dir) / "models" / "antelopev2"

    det_path = model_path / "scrfd_10g_bnkps.onnx"
    rec_path = model_path / "glintr100.onnx"

    if not det_path.exists():
        raise FileNotFoundError(
            f"Face detection model not found at {det_path}. "
            "Run: python scripts/download_models.py"
        )
    if not rec_path.exists():
        raise FileNotFoundError(
            f"Face recognition model not found at {rec_path}. "
            "Run: python scripts/download_models.py"
        )

    opts = onnxruntime.SessionOptions()
    opts.log_severity_level = 3  # Suppress warnings

    _det_session = onnxruntime.InferenceSession(str(det_path), opts, providers=providers)
    _rec_session = onnxruntime.InferenceSession(str(rec_path), opts, providers=providers)


def _distance2bbox(points, distance):
    """Convert distance predictions to bounding boxes."""
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def _distance2kps(points, distance):
    """Convert distance predictions to keypoints."""
    num_kps = distance.shape[1] // 2
    kps = np.zeros((distance.shape[0], num_kps, 2), dtype=np.float32)
    for i in range(num_kps):
        kps[:, i, 0] = points[:, 0] + distance[:, 2 * i]
        kps[:, i, 1] = points[:, 1] + distance[:, 2 * i + 1]
    return kps


def _nms(boxes, scores, threshold):
    """Non-maximum suppression."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= threshold)[0]
        order = order[inds + 1]

    return keep


def _detect_faces(img_bgr):
    """Detect faces using SCRFD model.

    Returns list of (bbox, score, landmarks) tuples.
    """
    h, w = img_bgr.shape[:2]
    input_h, input_w = _DET_INPUT_SIZE

    # Resize with aspect ratio preservation
    scale = min(input_h / h, input_w / w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(img_bgr, (new_w, new_h))

    # Pad to input size
    padded = np.zeros((input_h, input_w, 3), dtype=np.uint8)
    padded[:new_h, :new_w, :] = resized

    # Prepare input blob
    blob = cv2.dnn.blobFromImage(
        padded, 1.0 / 128.0, _DET_INPUT_SIZE, (127.5, 127.5, 127.5), swapRB=True
    )

    # Run detection
    outputs = _det_session.run(None, {_det_session.get_inputs()[0].name: blob})

    all_boxes = []
    all_scores = []
    all_kps = []

    for idx, stride in enumerate(_FEAT_STRIDE):
        feat_h = input_h // stride
        feat_w = input_w // stride

        # Generate anchor centers
        anchor_x, anchor_y = np.meshgrid(np.arange(feat_w), np.arange(feat_h))
        anchor_centers = np.stack([anchor_x, anchor_y], axis=-1).reshape(-1, 2)
        anchor_centers = (anchor_centers * stride).astype(np.float32)

        if _NUM_ANCHORS > 1:
            anchor_centers = np.stack([anchor_centers] * _NUM_ANCHORS, axis=1).reshape(-1, 2)

        scores = outputs[idx]
        bbox_preds = outputs[idx + _FMC]
        kps_preds = outputs[idx + _FMC * 2]

        scores = scores.flatten()
        bbox_preds = bbox_preds.reshape(-1, 4) * stride
        kps_preds = kps_preds.reshape(-1, 10) * stride

        # Filter by score
        mask = scores >= _SCORE_THRESHOLD
        if not mask.any():
            continue

        scores = scores[mask]
        bbox_preds = bbox_preds[mask]
        kps_preds = kps_preds[mask]
        centers = anchor_centers[mask]

        boxes = _distance2bbox(centers, bbox_preds)
        kps = _distance2kps(centers, kps_preds)

        all_boxes.append(boxes)
        all_scores.append(scores)
        all_kps.append(kps)

    if not all_boxes:
        return []

    all_boxes = np.concatenate(all_boxes, axis=0)
    all_scores = np.concatenate(all_scores, axis=0)
    all_kps = np.concatenate(all_kps, axis=0)

    # NMS
    keep = _nms(all_boxes, all_scores, _NMS_THRESHOLD)
    results = []
    for i in keep:
        bbox = all_boxes[i] / scale  # Scale back to original image size
        score = all_scores[i]
        kps = all_kps[i] / scale
        results.append((bbox, float(score), kps))

    return results


def _align_face(img_bgr, landmarks):
    """Align face using 5-point landmarks to 112x112 for ArcFace."""
    src_pts = landmarks.astype(np.float32)
    dst_pts = _REFERENCE_LANDMARKS.astype(np.float32)

    # Estimate affine transform
    M = cv2.estimateAffinePartial2D(src_pts, dst_pts)[0]
    if M is None:
        M = cv2.getAffineTransform(src_pts[:3], dst_pts[:3])

    aligned = cv2.warpAffine(img_bgr, M, (112, 112), borderValue=0)
    return aligned


def _get_embedding(aligned_face_bgr):
    """Extract 512-dim face embedding from aligned 112x112 face."""
    # Normalize to [-1, 1] and convert to NCHW float32
    img = aligned_face_bgr.astype(np.float32)
    img = (img / 127.5) - 1.0
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)  # Add batch dim

    embedding = _rec_session.run(None, {_rec_session.get_inputs()[0].name: img})[0]
    embedding = embedding.flatten()

    # L2 normalize
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm

    return embedding


def get_face_embedding(img_bgr: np.ndarray) -> np.ndarray:
    """Detect the most prominent face and return its 512-dim normed embedding.

    Args:
        img_bgr: Input image in BGR format (numpy array).

    Returns:
        512-dim normalized face embedding.

    Raises:
        ValueError: If no face is detected.
    """
    if _det_session is None or _rec_session is None:
        raise RuntimeError("Face models not loaded. Call face_analysis.load() first.")

    faces = _detect_faces(img_bgr)
    if not faces:
        raise ValueError(
            "No face detected in the reference image. "
            "Please upload a clear photo with a visible face."
        )

    # Pick the largest face (most prominent)
    largest = max(faces, key=lambda f: (f[0][2] - f[0][0]) * (f[0][3] - f[0][1]))
    _, _, landmarks = largest

    aligned = _align_face(img_bgr, landmarks)
    embedding = _get_embedding(aligned)

    return embedding
