from backend.app import create_app
from backend.app.config.settings import settings

app = create_app()

if __name__ == '__main__':
    app.run(debug=True)
