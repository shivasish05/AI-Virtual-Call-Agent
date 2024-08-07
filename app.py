

# call_center/app.py

from quart import Quart
import warnings
from config import Config
from logger import setup_logging
from routes import register_routes
from handlers import initialize_rag

warnings.filterwarnings('ignore')
setup_logging()

app = Quart(__name__)
register_routes(app)

@app.before_serving
async def setup():
    await initialize_rag()

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5002)  # Ensure host set for external access



