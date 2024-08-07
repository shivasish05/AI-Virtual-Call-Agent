# Contact Center Application

## Overview

This project is a contact center application built using the Quart web framework, Twilio for voice response, and a Retrieval-Augmented Generation (RAG) model to provide intelligent responses to user inquiries. The application answers user questions based on the content of a provided PDF document, and it interacts with users via voice input and output using Twilio's capabilities additionally it is capable for scheduling meeting during call.

## Features

- Voice interaction using Twilio
- Retrieval-Augmented Generation (RAG) for intelligent responses
- Support for various speech speeds
- Health check endpoint
- Configurable via environment variables
  

## Project Structure
```plaintext
.
├── app.py
├── config.py
├── handlers.py
├── logger.py
├── rag.py
├── routes.py
├── requirements.txt
└── README.md 
```

#### app.py
This is the main entry point of the application. It initializes the Quart app, sets up logging, and registers the routes.

#### config.py
Contains configuration settings for the application, loaded from environment variables.

#### handlers.py
Contains the main logic for handling Twilio webhooks, processing user input, and generating responses using the RAG model.

#### logger.py
Sets up logging for the application.

#### rag.py
Sets up the Retrieval-Augmented Generation (RAG) model using a PDF document as the data source.

#### routes.py
Registers the URL routes for the application.

Setup
Prerequisites
Python 3.9+
Virtualenv (optional but recommended)

## Installation

1. Clone the repository:
git clone https://github.com/yourusername/contact-center-app.git
cd contact-center-app
2. Create and activate a virtual environment:
3. Install the dependencies:(requirements.txt)
4. create .env:
- OPENAI_API_KEY=your_openai_api_key
- TWILIO_ACCOUNT_SID=your_twilio_account_sid
- TWILIO_AUTH_TOKEN=your_twilio_auth_token
- ELEVEN_LABS_API_KEY=your_eleven_labs_api_key
- ELEVENLABS_VOICE_ID=your_elevenlabs_voice_id


## Endpoints
- /twilio
Handles the initial Twilio webhook and prompts the user for input.

- /gather
Handles the speech input gathered from the user, processes it using the RAG model, and responds accordingly.

- /continue
Handles continuation of the conversation.

- /health_check
Returns a simple "Working Fine!" message to indicate that the server is running.

## Logging
Logging is configured in logger.py and logs messages with the format:

-YYYY-MM-DD HH:MM:SS LEVEL Message

## Handlers
### twilio_webhook
- Initializes a voice response and prompts the user.
- Configures speech gathering with options like language and speech timeout.
#### gather
- Processes user input from Twilio.
- Handles empty input cases and retrieves answers using the RAG model.
- Adds user and assistant messages to the conversation log.
#### retrieve_answer
- Retrieves answers from the RAG model.
- Can respond with different speech rates based on user input.
#### convo
- Prepares for continuation of the conversation.
#### health_check
- Returns a simple status message.
#### handle_empty_input
- Handles cases where user input is empty.
#### handle_error
- Handles error cases by returning an appropriate Twilio response.


### We are currently using ngrok:
##### Open your terminal 
- https ngrok 80000:


  <img width="563" alt="Screenshot 2024-07-23 at 5 00 12 PM" src="https://github.com/user-attachments/assets/3c175e77-43a1-40b7-a914-236f0b421805">


- Copy the domain without https:


  <img width="688" alt="Screenshot 2024-07-23 at 5 01 25 PM" src="https://github.com/user-attachments/assets/ce271e5f-f4f1-4294-9216-af5f049a0ae5">


- Open twilio concole and past it with the endpoint /twilio:


<img width="707" alt="Screenshot 2024-07-23 at 5 02 14 PM" src="https://github.com/user-attachments/assets/d9a84ea0-f145-4413-8ac6-48057c04921f">



