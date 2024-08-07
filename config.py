# call_center/config.py

import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
    TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
    PDF_PATH = 'https://www.england.nhs.uk/revalidation/wp-content/uploads/sites/10/2014/05/rev-faqs-v6.pdf'
    LOCAL_VECTOR_STORE = 'CallAgentDB'
