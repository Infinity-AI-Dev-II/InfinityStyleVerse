from flask import Blueprint, jsonify, render_template
# from backend.app.decorators import role_required 
from backend.app.Decorators.RoleDecorator import role_required 
from flask_jwt_extended import jwt_required

from dotenv import load_dotenv
import os

API_URL = os.getenv("API_URL")

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def home():
    return render_template('index.html')

@main_bp.route('/about')
def about():
    return render_template('about.html')

@main_bp.route('/contact')
def contact():
    return render_template('contact.html')

@main_bp.route('/profile')
def profile():
    return render_template('profile.html',api_url=API_URL)

@main_bp.route('/login')
def login_page():
    return render_template('login.html',api_url=API_URL)

@main_bp.route('/register')
def register_page():
    return render_template('register.html',api_url=API_URL)

@main_bp.route('/onboarding')
def onboarding():
    return render_template('onboarding.html',api_url=API_URL)
    
@main_bp.route('/creator/dashboard')
def creator_dashboard():
    return render_template('creator/creator_dashboard.html',api_url=API_URL)

@main_bp.route('/creator/display')
def creator_display():
    return render_template('creator/display.html',api_url=API_URL)

@main_bp.route('/admin/dashboard')
@jwt_required()
@role_required('admin')
def admin_area():
    return jsonify({"msg": "Welcome, admin!"})
    return render_template('admin/admindashboard.html')
 
@main_bp.route('/customer/dashboard')
def customer_dashboard():
    return render_template('customer/customer_dashboard.html',api_url=API_URL)

@main_bp.route('/staff/dashboard')
def staff_dashboard():
    return render_template('staffdashboard.html',api_url=API_URL)

@main_bp.route('/index')
def index():
    return render_template('index.html')

@main_bp.route('/addproduct')
def addproduct():
    return render_template('addproduct.html',api_url=API_URL)

@main_bp.route('/profileview')
def profileview():
    return render_template('profile.html',api_url=API_URL)