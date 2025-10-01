from flask import Blueprint, jsonify, request
from flask_cors import CORS
import pickle
import csv
import os
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import random
from flask_jwt_extended import jwt_required
import joblib
import pandas as pd
from ..models import Product, ProductImage
from ..utils.esg_utils import generate_esg_columns, compute_esg_score

esg_model_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', 'models', 'esg_model.pkl')
)
esg_model = joblib.load(esg_model_path)

recommendation_bp = Blueprint('recommendation', __name__)

# Loading the saved model data
MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', 'models', 'new_recommender_model.pkl')
)

with open(MODEL_PATH, 'rb') as f:
    model_data = pickle.load(f)
    df = model_data['df']
    combined_similarities = model_data['similarities']

df['product_id'] = df['product_id'].astype(str).str.strip()

df['combined_text'] = df['title'] + " " + df['descriptions']

# Recompute TF-IDF only which is needed to process new user text input
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['combined_text'])

# Log file setup
# Set up a relative path for the log file
log_file = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'logs', 'recommendation_log.csv')
)

# Ensure the parent directory exists
os.makedirs(os.path.dirname(log_file), exist_ok=True)

if not os.path.exists(log_file):
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'product_id', 'title', 'action', 'input'])

# Log function
def log_event(product_id, title, action, user_input):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, product_id, title, action, user_input])


# Add ESG columns and compute ESG scores
df = generate_esg_columns(df)
df = compute_esg_score(df, esg_model)

# Preview results
print(df.head())

# Recommend products
@recommendation_bp.route('/api/recommend', methods=['POST'])
@jwt_required()
def recommend():
    try:
        data = request.get_json()
        user_input = data.get('query', '').strip().lower()

        if not user_input:
            return jsonify({"error": "No input provided"}), 400

        # Convert user input to TF-IDF vector
        user_vector = tfidf.transform([user_input])
        text_scores = cosine_similarity(user_vector, tfidf_matrix).flatten()

        category_scores = np.max(combined_similarities, axis=1)
        color_scores = np.max(combined_similarities, axis=1)

        # Combine weighted scores
        weight_text = 0.6
        weight_category = 0.2
        weight_color = 0.2

        combined_scores = (
            weight_text * text_scores +
            weight_category * category_scores +
            weight_color * color_scores
        )

        # Get top 3
        scores_with_indices = sorted(enumerate(combined_scores), key=lambda x: x[1], reverse=True)
        top_3 = scores_with_indices[:3]

        recommendations = []
        for i, score in top_3:
            product = {
                'product_id': df.loc[i, 'product_id'],
                'title': df.loc[i, 'title'],
                'description': df.loc[i, 'descriptions'],
                'category': df.loc[i, 'category'],
                'color': df.loc[i, 'color'],
                'image_url': df.loc[i, 'image_url'],
                'score': float(score)
            }
            log_event(product['product_id'], product['title'], 'shown', user_input)
            recommendations.append(product)

        # fallback process
        if max(score for _, score in top_3) < 0.1:
            recommendations = []
            fallback_indices = random.sample(range(len(df)), 3)
            for i in fallback_indices:
                product = {
                    'product_id': df.loc[i, 'product_id'],
                    'title': df.loc[i, 'title'],
                    'description': df.loc[i, 'descriptions'],
                    'category': df.loc[i, 'category'],
                    'color': df.loc[i, 'color'],
                    'image_url': df.loc[i, 'image_url'],
                    'score': 0.0
                }
                log_event(product['product_id'], product['title'], 'shown', user_input)
                recommendations.append(product)

        return jsonify({"results": recommendations})

    except Exception as e:
        print("Error in /api/recommend:", e)
        return jsonify({"error": str(e)}), 500

# Log event
@recommendation_bp.route('/api/log-click', methods=['POST'])
@jwt_required()
def log_click():
    try:
        data = request.get_json()
        product_id = data.get('product_id')
        title = data.get('title')
        user_input = data.get('input', '')
        if not product_id or not title:
            return jsonify({"error": "Missing product_id or title"}), 400

        log_event(product_id, title, 'clicked', user_input)
        return jsonify({"message": "Click logged"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@recommendation_bp.route('/api/recommend-full', methods=['POST'])
@jwt_required()
def recommend_full():
    try:
        data = request.get_json()
        user_input = data.get('query', '').strip().lower()

        if not user_input:
            return jsonify({"error": "No input provided"}), 400

         
        user_vector = tfidf.transform([user_input])
        text_scores = cosine_similarity(user_vector, tfidf_matrix).flatten()

        category_scores = np.max(combined_similarities, axis=1)
        color_scores = np.max(combined_similarities, axis=1)

        weight_text = 0.6
        weight_category = 0.2
        weight_color = 0.2

        combined_scores = (
            weight_text * text_scores +
            weight_category * category_scores +
            weight_color * color_scores
        )

        # Boost ESG scores if score ≥ 70 by 10% and 20% if score >= 85
        eco_boost_1 = 1.1
        eco_boost_2 = 1.2   # For highly sustainable products
        for i in range(len(combined_scores)):
            if df.loc[i, 'esg_score'] >= 85:
                combined_scores[i] *= eco_boost_2
            elif df.loc[i, 'esg_score'] >= 70:
                combined_scores[i] *= eco_boost_1
                
        # Sort by updated combined score
        scores_with_indices = sorted(enumerate(combined_scores), key=lambda x: x[1], reverse=True)
        
        # Filter out products with ESG < 50
        filtered = [(i, score) for i, score in scores_with_indices if df.loc[i, 'esg_score'] >= 50]
        
        # Take top 3
        top_3 = filtered[:3]

        print("Top 3 after ESG filtering:")
        for i, score in top_3:
            print(f"Title: {df.loc[i, 'title']}, ESG Score: {df.loc[i, 'esg_score']}, Final Score: {score}")

        # scores_with_indices = sorted(enumerate(combined_scores), key=lambda x: x[1], reverse=True)
        # top_3 = scores_with_indices[:3]

        product_ids = [df.loc[i, 'product_id'] for i, _ in top_3]

        # Log events
        for pid in product_ids:
            title = df[df['product_id'] == pid]['title'].values[0]
            log_event(pid, title, 'shown', user_input)

        # Convert IDs to integers 
        int_ids = [int(pid) for pid in product_ids]

    
        products = Product.query.filter(Product.id.in_(int_ids)).all()

       
        score_map = {int(pid): float(score) for pid, score in zip(product_ids, [s for _, s in top_3])}

        results = []
        for p in products:
            results.append({
                "id": p.id,
                "title": p.title,
                "brand": p.brand,
                "category": p.category,
                "description": p.description,
                "sale_price": p.sale_price,
                "discount": p.discount,
                "colors": p.colors.split(",") if p.colors else [],
                "sizes": p.sizes.split(",") if p.sizes else [],
                "tags": p.tags.split(",") if p.tags else [],
                "publish_date": p.publish_date.isoformat() if p.publish_date else None,
                "score": score_map.get(p.id, 0.0),
                "esg_score": float(df_row['esg_score']),
                "esg_badge": df_row['esg_badge']
            })

        if not results:
            return jsonify({"error": "No matching products found in database."}), 404

        return jsonify({"results": results})

    except Exception as e:
        print("Error in /api/recommend-full:", e)
        return jsonify({"error": str(e)}), 500