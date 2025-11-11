"""
Improved prediction system with proper leaf detection
- Detects ALL leaves in video frames
- Lower confidence threshold (30% instead of 50%)
- Proper multi-leaf tracking per frame
- No frame skipping
"""
import os
import logging
import traceback
import numpy as np
import cv2
from PIL import Image
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from .models import TreeAnalysis, LeafImage, Plant

logger = logging.getLogger(__name__)

CONF_THRESHOLD = 0.30  # 30% instead of 50%

ALLOWED_CLASSES = {"healthy", "dried leaf", "leaf rust", "powdery mildew"}
CLASS_NAMES = ['dried leaf', 'healthy', 'leaf rust', 'powdery mildew']

# Model cache
_MODEL_CACHE = {}

def load_yolo_model():
    """Load YOLO model with caching"""
    global _MODEL_CACHE

    if 'yolo_model' in _MODEL_CACHE:
        logger.info("✅ Using cached YOLO model")
        return _MODEL_CACHE['yolo_model']

    model_path = os.path.join(settings.MEDIA_ROOT, 'best.pt')
    if not os.path.exists(model_path):
        logger.error(f"❌ YOLO model file not found at: {model_path}")
        return None

    try:
        from ultralytics import YOLO
        logger.info("🔄 Loading YOLO model...")
        model = YOLO(model_path)
        _MODEL_CACHE['yolo_model'] = model
        logger.info("✅ YOLO model loaded successfully")
        return model

    except Exception as e:
        logger.error(f"❌ Error loading YOLO model: {e}")
        traceback.print_exc()
        return None


@csrf_exempt
@require_POST
def predict_all_leaves(request):
    """
    New endpoint that detects ALL leaves in a frame
    - Uses lower confidence threshold
    - Detects ALL leaves without tracking
    - Stores each detected leaf
    """
    try:
        model = load_yolo_model()
        if model is None:
            return JsonResponse({"success": False, "error": "YOLO model not loaded"})

        frame_file = request.FILES.get('frame')
        plant_id = request.POST.get("plant_id")
        analysis_id = request.POST.get("tree_analysis_id")

        if not frame_file:
            return JsonResponse({"success": False, "error": "No frame received"})

        file_bytes = np.frombuffer(frame_file.read(), np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if frame is None:
            return JsonResponse({"success": False, "error": "Failed to decode frame"})

        logger.info(f"✅ Processing frame - shape: {frame.shape}")

        results = model.predict(frame, conf=CONF_THRESHOLD, verbose=False)[0]

        all_detections = []

        if results.boxes is not None:
            logger.info(f"📊 Found {len(results.boxes)} potential leaves")

            for idx, box in enumerate(results.boxes):
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                # Get class name
                class_name = model.names[cls].lower().replace('-', ' ')

                if class_name not in ALLOWED_CLASSES:
                    continue

                # Confidence scores for all classes
                confidences = {}
                for i, class_name_check in enumerate(CLASS_NAMES):
                    if i < len(model.names):
                        confidences[class_name_check.title()] = float(results.conf[idx][i]) if hasattr(results.conf[idx], '__getitem__') else conf

                detection = {
                    "index": idx,
                    "box": [x1, y1, x2, y2],
                    "confidence": conf,
                    "class": class_name.title(),
                    "confidences": confidences
                }

                all_detections.append(detection)
                logger.info(f"  ✅ Detected leaf #{idx + 1}: {class_name.title()} ({conf*100:.1f}%)")

        logger.info(f"🎯 Total valid leaves detected: {len(all_detections)}")

        if analysis_id:
            try:
                tree_analysis = TreeAnalysis.objects.get(id=analysis_id)

                for det in all_detections:
                    # Save each detected leaf
                    leaf_image = LeafImage.objects.create(
                        tree_analysis=tree_analysis,
                        prediction=det["class"],
                        healthy_confidence=det["confidences"].get("Healthy", 0),
                        dried_leaf_confidence=det["confidences"].get("Dried Leaf", 0),
                        leaf_rust_confidence=det["confidences"].get("Leaf Rust", 0),
                        powdery_mildew_confidence=det["confidences"].get("Powdery Mildew", 0),
                    )
                    logger.info(f"  💾 Saved leaf #{det['index'] + 1} to database")

                # Recalculate overall health
                tree_analysis.calculate_health()
                tree_analysis.save()
                logger.info(f"✅ Analysis updated - Total leaves: {tree_analysis.total_leaves}")

            except TreeAnalysis.DoesNotExist:
                logger.warning(f"TreeAnalysis ID {analysis_id} not found")
            except Exception as e:
                logger.error(f"Error saving to database: {e}")
                traceback.print_exc()

        return JsonResponse({
            "success": True,
            "detections_count": len(all_detections),
            "detections": all_detections,
            "message": f"✅ Detected {len(all_detections)} leaves successfully"
        })

    except Exception as e:
        logger.error(f"❌ Error in predict_all_leaves: {e}")
        traceback.print_exc()
        return JsonResponse({"success": False, "error": str(e)})


@csrf_exempt
@require_POST
def predict_from_image(request):
    """
    Endpoint for single image prediction
    - Detects all leaves in the image
    - Returns detailed confidence scores
    """
    try:
        model = load_yolo_model()
        if model is None:
            return JsonResponse({"success": False, "error": "YOLO model not loaded"})

        # Get image from request
        image_data = request.POST.get('image')
        tree_analysis_id = request.POST.get('tree_analysis_id')

        if not image_data:
            return JsonResponse({"success": False, "error": "No image data received"})

        import base64
        try:
            image_data_clean = image_data.split(',')[1] if ',' in image_data else image_data
            image_bytes = base64.b64decode(image_data_clean)
            nparr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as e:
            logger.error(f"Error decoding image: {e}")
            return JsonResponse({"success": False, "error": "Failed to decode image"})

        if frame is None:
            return JsonResponse({"success": False, "error": "Failed to process image"})

        results = model.predict(frame, conf=CONF_THRESHOLD, verbose=False)[0]

        all_detections = []
        class_counts = {}

        if results.boxes is not None:
            logger.info(f"📊 Detected {len(results.boxes)} leaves in image")

            for idx, box in enumerate(results.boxes):
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = model.names[cls].lower().replace('-', ' ')

                if class_name not in ALLOWED_CLASSES:
                    continue

                # Count detections by class
                class_name_title = class_name.title()
                class_counts[class_name_title] = class_counts.get(class_name_title, 0) + 1

                all_detections.append({
                    "class": class_name_title,
                    "confidence": round(conf * 100, 2),
                    "box": box.xyxy[0].tolist()
                })

        if tree_analysis_id and all_detections:
            try:
                tree_analysis = TreeAnalysis.objects.get(id=tree_analysis_id)
                for det in all_detections:
                    LeafImage.objects.create(
                        tree_analysis=tree_analysis,
                        prediction=det["class"],
                        healthy_confidence=100 if det["class"] == "Healthy" else 0,
                        dried_leaf_confidence=100 if det["class"] == "Dried Leaf" else 0,
                        leaf_rust_confidence=100 if det["class"] == "Leaf Rust" else 0,
                        powdery_mildew_confidence=100 if det["class"] == "Powdery Mildew" else 0,
                    )

                tree_analysis.calculate_health()
                tree_analysis.save()
                logger.info(f"✅ Saved {len(all_detections)} leaves to analysis {tree_analysis_id}")
            except Exception as e:
                logger.error(f"Error saving to database: {e}")

        return JsonResponse({
            "success": True,
            "total_detections": len(all_detections),
            "detection_summary": class_counts,
            "detections": all_detections,
            "message": f"✅ Successfully detected {len(all_detections)} leaves"
        })

    except Exception as e:
        logger.error(f"❌ Error: {e}")
        traceback.print_exc()
        return JsonResponse({"success": False, "error": str(e)})
