#run.py
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.app import create_app
from backend.app.config.settings import settings

app= create_app()

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8000)
    