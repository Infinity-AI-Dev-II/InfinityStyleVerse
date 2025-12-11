# Catalog Vision Backend API

This module provides the backend API for the StyleSense CatalogVision Playground frontend.

## Endpoints

### POST `/stylesense/catalog/ingest`

Tags an item from its image.

**Request Body:**
```json
{
  "image_uri": "data:image/jpeg;base64,..." or "s3://bucket/image.jpg",
  "sku": "SKU-123",
  "hints": {
    "category": "Tops"  // optional
  }
}
```

**Response:**
```json
{
  "image_uri": "...",
  "sku": "SKU-123",
  "vector_dim": 768,
  "tags": {
    "category": {"label": "Tops", "conf": 0.98},
    "subcat": {"label": "Crew Neck Tee", "conf": 0.94},
    "occasion": [{"label": "Casual", "conf": 0.91}],
    "attributes": [
      {"label": "Color: Navy", "conf": 0.95},
      {"label": "Material: Cotton", "conf": 0.88}
    ]
  },
  "quality_flags": ["NO_OCCLUSION", "GOOD_LIGHTING"],
  "request_id": "req_abc123",
  "embedding_uri": "s3://styleverse/embeddings/SKU-123.npy",
  "latency_ms": 473
}
```

### POST `/stylesense/catalog/similar`

Finds similar items to a given SKU.

**Request Body:**
```json
{
  "sku": "SKU-123",
  "k": 6,  // optional, default 6, max 50
  "filters": {
    "category": "Tops"  // optional
  }
}
```

**Response:**
```json
{
  "items": [
    {"sku": "SKU-456", "sim": 0.9234},
    {"sku": "SKU-789", "sim": 0.8912}
  ],
  "latency_ms": 120
}
```

## Architecture

- **Service Layer**: `catalog_vision_service.py` - Core business logic for tagging and similarity search
- **Routes**: `catalog_vision_routes.py` - Flask blueprint with API endpoints
- **Data Loader**: `data_loader.py` - Loads catalog seed data

## Implementation Notes

- Currently uses mock embeddings generated deterministically from image data
- Embeddings are stored in memory (can be replaced with vector DB)
- Similarity search uses cosine similarity on normalized embeddings
- The service loads catalog data from `data/stylesense_seed_data.csv` on initialization

## Future Enhancements

- Replace mock embeddings with actual vision model (e.g., CLIP, ResNet)
- Add persistent vector database (e.g., Pinecone, Weaviate, Qdrant)
- Add caching layer for frequently accessed items
- Add batch processing support
- Add authentication/authorization if needed

