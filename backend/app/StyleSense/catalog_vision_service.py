"""
Catalog Vision Service
Handles image tagging, embedding generation, and similarity search for the StyleSense CatalogVision module.
"""
import base64
import time
import uuid
import hashlib
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import pandas as pd

from .data_loader import load_stylesense_data


class CatalogVisionService:
    """Service for catalog vision operations: tagging and similarity search."""
    
    # Category mappings
    CATEGORIES = [
        "Tops", "Bottoms", "Dresses", "Outerwear", "Shoes", 
        
    ]
    
    SUBCATEGORIES = {
        "Tops": ["Crew Neck Tee", "V-Neck Tee", "Tank Top", "Blouse", "Sweater", "Hoodie"],
        
    }
    
    OCCASIONS = ["Casual", "Formal", "Party", "Streetwear", "Business", "Athletic"]
    
    ATTRIBUTES = [
        "Color: Navy", "Color: Black", "Color: White", "Color: Red", "Color: Blue",
        
    ]
    
    QUALITY_FLAGS = ["NO_OCCLUSION", "GOOD_LIGHTING", "CLEAR_BACKGROUND", "SINGLE_ITEM"]
    
    def __init__(self):
        """Initialize the service with data and embeddings."""
        self.vector_dim = 768
        self.embeddings_cache: Dict[str, np.ndarray] = {}
        self.item_metadata: Dict[str, Dict] = {}
        self._load_catalog_data()
    
    def _load_catalog_data(self):
        """Load catalog data and generate mock embeddings."""
        try:
            df = load_stylesense_data("catalog_vision")
            df = df.dropna(subset=['sku', 'image_uri'])
            
            # Generate mock embeddings for each item
            np.random.seed(42)  # For reproducibility
            for idx, row in df.iterrows():
                sku = str(row['sku']).strip()
                # Generate a deterministic embedding based on SKU hash
                sku_hash = int(hashlib.md5(sku.encode()).hexdigest()[:8], 16)
                np.random.seed(sku_hash)
                embedding = np.random.randn(self.vector_dim)
                embedding = embedding / np.linalg.norm(embedding)  # Normalize
                
                self.embeddings_cache[sku] = embedding
                self.item_metadata[sku] = {
                    'category': row.get('category', 'Tops'),
                    'color': row.get('color', 'Navy'),
                    'pattern': row.get('pattern', 'Solid'),
                    'image_uri': row.get('image_uri', '')
                }
        except Exception as e:
            print(f"Warning: Could not load catalog data: {e}")
            # Initialize with empty data
    
    def _generate_embedding_from_image(self, image_base64: str) -> np.ndarray:
        """
        Generate embedding from base64 image.
        In production, this would use a vision model (e.g., CLIP, ResNet).
        For now, we generate a deterministic mock embedding.
        """
        # Extract image hash for deterministic embedding
        image_data = base64.b64decode(image_base64.split(',')[-1] if ',' in image_base64 else image_base64)
        image_hash = hashlib.md5(image_data[:1000]).hexdigest()  # Use first 1KB for hash
        hash_int = int(image_hash[:8], 16)
        
        np.random.seed(hash_int)
        embedding = np.random.randn(self.vector_dim)
        embedding = embedding / np.linalg.norm(embedding)
        return embedding
    
    def _predict_category(self, embedding: np.ndarray, hint_category: Optional[str] = None) -> Tuple[str, float]:
        """Predict category from embedding."""
        if hint_category and hint_category in self.CATEGORIES:
            return hint_category, 0.95
        
        # Mock prediction based on embedding
        category_idx = int(abs(embedding[0]) * len(self.CATEGORIES)) % len(self.CATEGORIES)
        category = self.CATEGORIES[category_idx]
        confidence = 0.85 + abs(embedding[1]) * 0.1
        return category, min(confidence, 0.99)
    
    def _predict_subcategory(self, category: str, embedding: np.ndarray) -> Tuple[str, float]:
        """Predict subcategory from category and embedding."""
        if category not in self.SUBCATEGORIES:
            return "Standard", 0.80
        
        subcats = self.SUBCATEGORIES[category]
        subcat_idx = int(abs(embedding[2]) * len(subcats)) % len(subcats)
        subcat = subcats[subcat_idx]
        confidence = 0.88 + abs(embedding[3]) * 0.08
        return subcat, min(confidence, 0.97)
    
    def _predict_occasion(self, embedding: np.ndarray) -> List[Tuple[str, float]]:
        """Predict occasions from embedding."""
        occasions = []
        # Primary occasion
        occ_idx = int(abs(embedding[4]) * len(self.OCCASIONS)) % len(self.OCCASIONS)
        primary_occ = self.OCCASIONS[occ_idx]
        primary_conf = 0.88 + abs(embedding[5]) * 0.09
        occasions.append((primary_occ, min(primary_conf, 0.96)))
        
        # Secondary occasion (if confidence is high enough)
        if abs(embedding[6]) > 0.5:
            occ_idx2 = (occ_idx + 1) % len(self.OCCASIONS)
            secondary_occ = self.OCCASIONS[occ_idx2]
            secondary_conf = 0.70 + abs(embedding[7]) * 0.15
            occasions.append((secondary_occ, min(secondary_conf, 0.85)))
        
        return occasions
    
    def _predict_attributes(self, embedding: np.ndarray, category: str) -> List[Tuple[str, float]]:
        """Predict attributes from embedding."""
        attributes = []
        num_attrs = 3 + int(abs(embedding[8]) * 3)  # 3-6 attributes
        
        for i in range(num_attrs):
            attr_idx = int(abs(embedding[9 + i]) * len(self.ATTRIBUTES)) % len(self.ATTRIBUTES)
            attr = self.ATTRIBUTES[attr_idx]
            conf = 0.82 + abs(embedding[10 + i]) * 0.13
            attributes.append((attr, min(conf, 0.95)))
        
        return attributes
    
    def _generate_quality_flags(self, embedding: np.ndarray) -> List[str]:
        """Generate quality flags based on embedding characteristics."""
        flags = []
        # Mock logic: flags are present if certain embedding values are in range
        if abs(embedding[20]) < 0.3:
            flags.append("NO_OCCLUSION")
        if abs(embedding[21]) > 0.5:
            flags.append("GOOD_LIGHTING")
        if abs(embedding[22]) < 0.4:
            flags.append("CLEAR_BACKGROUND")
        if abs(embedding[23]) < 0.5:
            flags.append("SINGLE_ITEM")
        
        return flags if flags else ["NO_OCCLUSION"]  # At least one flag
    
    def tag_item(
        self, 
        image_uri: str, 
        sku: str, 
        hint_category: Optional[str] = None
    ) -> Dict:
        """
        Tag an item from its image.
        
        Args:
            image_uri: Base64 encoded image or image URI
            sku: SKU identifier
            hint_category: Optional category hint
            
        Returns:
            Dictionary with tagging results
        """
        start_time = time.time()
        request_id = f"req_{uuid.uuid4().hex[:8]}"
        
        # Generate embedding
        if image_uri.startswith('data:image') or len(image_uri) > 100:
            # Base64 image
            embedding = self._generate_embedding_from_image(image_uri)
        else:
            # URI - generate deterministic embedding
            embedding = self._generate_embedding_from_image(image_uri)
        
        # Predict tags
        category, cat_conf = self._predict_category(embedding, hint_category)
        subcategory, subcat_conf = self._predict_subcategory(category, embedding)
        occasions = self._predict_occasion(embedding)
        attributes = self._predict_attributes(embedding, category)
        quality_flags = self._generate_quality_flags(embedding)
        
        # Store embedding for similarity search
        self.embeddings_cache[sku] = embedding
        self.item_metadata[sku] = {
            'category': category,
            'subcategory': subcategory,
            'image_uri': image_uri
        }
        
        # Generate embedding URI
        embedding_uri = f"s3://styleverse/embeddings/{sku.replace('/', '_')}.npy"
        
        latency_ms = int((time.time() - start_time) * 1000)
        
        # Format response
        response = {
            "image_uri": image_uri,
            "sku": sku,
            "vector_dim": self.vector_dim,
            "tags": {
                "category": {
                    "label": category,
                    "conf": round(cat_conf, 2)
                },
                "subcat": {
                    "label": subcategory,
                    "conf": round(subcat_conf, 2)
                },
                "occasion": [
                    {"label": occ, "conf": round(conf, 2)} 
                    for occ, conf in occasions
                ],
                "attributes": [
                    {"label": attr, "conf": round(conf, 2)} 
                    for attr, conf in attributes
                ]
            },
            "quality_flags": quality_flags,
            "request_id": request_id,
            "embedding_uri": embedding_uri,
            "latency_ms": latency_ms
        }
        
        return response
    
    def find_similar_items(
        self, 
        sku: str, 
        k: int = 6, 
        filters: Optional[Dict] = None
    ) -> Dict:
        """
        Find similar items to a given SKU.
        
        Args:
            sku: SKU identifier
            k: Number of similar items to return
            filters: Optional filters (e.g., {"category": "Tops"})
            
        Returns:
            Dictionary with similar items
        """
        start_time = time.time()
        
        if sku not in self.embeddings_cache:
            raise ValueError(f"SKU {sku} not found. Please tag the item first.")
        
        query_embedding = self.embeddings_cache[sku]
        query_metadata = self.item_metadata.get(sku, {})
        
        # Calculate similarities
        similarities = []
        for other_sku, other_embedding in self.embeddings_cache.items():
            if other_sku == sku:
                continue
            
            # Apply filters
            if filters:
                if 'category' in filters:
                    other_metadata = self.item_metadata.get(other_sku, {})
                    if other_metadata.get('category') != filters['category']:
                        continue
            
            # Calculate cosine similarity
            similarity = float(np.dot(query_embedding, other_embedding))
            similarities.append({
                'sku': other_sku,
                'sim': similarity,
                'metadata': self.item_metadata.get(other_sku, {})
            })
        
        # Sort by similarity and take top k
        similarities.sort(key=lambda x: x['sim'], reverse=True)
        top_k = similarities[:k]
        
        latency_ms = int((time.time() - start_time) * 1000)
        
        response = {
            "items": [
                {
                    "sku": item['sku'],
                    "sim": round(item['sim'], 4)
                }
                for item in top_k
            ],
            "latency_ms": latency_ms
        }
        
        return response


# Global service instance
_catalog_vision_service = None

def get_catalog_vision_service() -> CatalogVisionService:
    """Get or create the global catalog vision service instance."""
    global _catalog_vision_service
    if _catalog_vision_service is None:
        _catalog_vision_service = CatalogVisionService()
    return _catalog_vision_service

