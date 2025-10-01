import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend', 'app')))
from backend.app.utils.esg_module import process_product_esg

# Define 10 test products
test_products = [
    {'water_use': 30, 'carbon_emission': 40, 'ethical_rating': 70, 'material': 'leather'},
    {'water_use': 10, 'carbon_emission': 20, 'ethical_rating': 90, 'material': 'organic cotton'},
    {'water_use': 50, 'carbon_emission': 60, 'ethical_rating': 20, 'material': 'polyester'},
    {'water_use': 0, 'carbon_emission': 0, 'ethical_rating': 0, 'material': 'nylon'},
    {'water_use': 25, 'carbon_emission': 35, 'ethical_rating': 50, 'material': 'bamboo'},
    {'water_use': 100, 'carbon_emission': 90, 'ethical_rating': 10, 'material': 'acrylic'},
    {'water_use': 15, 'carbon_emission': 25, 'ethical_rating': 85, 'material': 'cotton'},
    {'water_use': 20, 'carbon_emission': 20, 'ethical_rating': 80, 'material': ''},
    {'water_use': 20, 'carbon_emission': 20, 'ethical_rating': 80}, 
    {'water_use': None, 'carbon_emission': 20, 'ethical_rating': 80, 'material': 'denim'}, 
]

# Run the tests
for i, product in enumerate(test_products, 1):
    try:
        result = process_product_esg(product)
        print(f"Test {i}: {result}")
    except Exception as e:
        print(f"Test {i} failed: {e}")