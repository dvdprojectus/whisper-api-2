from dotenv import find_dotenv, load_dotenv
from os import environ as env
import logging

# Loading env variables
ENV_FILE = find_dotenv()
if ENV_FILE:
    load_dotenv(ENV_FILE)
    
# setting up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check environment
if env.get("ENVIRONMENT") != "local":
    # Set logging level to WARNING or higher for non-local environments
    logger.setLevel(logging.WARNING)