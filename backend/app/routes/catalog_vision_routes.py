"""
Catalog Vision API Routes
Handles /stylesense/catalog endpoints for image tagging and similarity search.
"""
from flask import Blueprint, request, jsonify
from typing import Optional
import traceback

from ..StyleSense.catalog_vision_service import get_catalog_vision_service

catalog_vision_bp = Blueprint('catalog_vision', __name__)


@catalog_vision_bp.route('/stylesense/catalog/ingest', methods=['POST'])
def ingest_item():
    """
    Tag an item from its image.
    
    Request body:
    {
        "image_uri": "data:image/jpeg;base64,..." or "s3://...",
        "sku": "SKU-123",
        "hints": {"category": "Tops"}  # optional
    }
    
    Response:
    {
        "image_uri": "...",
        "sku": "SKU-123",
        "vector_dim": 768,
        "tags": {
            "category": {"label": "Tops", "conf": 0.98},
            "subcat": {"label": "Crew Neck Tee", "conf": 0.94},
            "occasion": [{"label": "Casual", "conf": 0.91}],
            "attributes": [...]
        },
        "quality_flags": ["NO_OCCLUSION"],
        "request_id": "req_abc123",
        "embedding_uri": "s3://...",
        "latency_ms": 473
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "Request body is required"}), 400
        
        image_uri = data.get('image_uri')
        if not image_uri:
            return jsonify({"error": "image_uri is required"}), 400
        
        sku = data.get('sku')
        if not sku:
            return jsonify({"error": "sku is required"}), 400
        
        hints = data.get('hints', {})
        hint_category = hints.get('category') if hints else None
        
        service = get_catalog_vision_service()
        result = service.tag_item(image_uri, sku, hint_category)
        
        return jsonify(result), 200
        
    except Exception as e:
        error_msg = str(e)
        traceback.print_exc()
        return jsonify({
            "error": "Internal server error",
            "message": error_msg
        }), 500


@catalog_vision_bp.route('/stylesense/catalog/similar', methods=['POST'])
def find_similar():
    """
    Find similar items to a given SKU.
    
    Request body:
    {
        "sku": "SKU-123",
        "k": 6,  # optional, default 6
        "filters": {"category": "Tops"}  # optional
    }
    
    Response:
    {
        "items": [
            {"sku": "SKU-456", "sim": 0.9234},
            ...
        ],
        "latency_ms": 120
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "Request body is required"}), 400
        
        sku = data.get('sku')
        if not sku:
            return jsonify({"error": "sku is required"}), 400
        
        k = data.get('k', 6)
        if not isinstance(k, int) or k < 1 or k > 50:
            k = 6
        
        filters = data.get('filters', {})
        
        service = get_catalog_vision_service()
        result = service.find_similar_items(sku, k, filters)
        
        return jsonify(result), 200
        
    except ValueError as e:
        return jsonify({
            "error": "Invalid request",
            "message": str(e)
        }), 400
    except Exception as e:
        error_msg = str(e)
        traceback.print_exc()
        return jsonify({
            "error": "Internal server error",
            "message": error_msg
        }), 500

