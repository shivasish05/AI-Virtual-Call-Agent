
import openai
import pyaudio
import wave
import os
from dotenv import load_dotenv
import json
from pydub import AudioSegment
from pydub.playback import play
import io
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import OpenAIChat
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from datetime import datetime
import logging
import warnings

import queue
import threading
# Load OpenAI API key and ElevenLabs API key
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')
elevenlabs_api_key = os.getenv('ELEVENLABS_API_KEY')

pdf_path = '/Users/vishalprajapati/Documents/R&D/aicallagent/OBF - Virtual Call Agent Knowledge.pdf'
local_vector_store = 'OBF-CallAgentDB'


# pdf_path = 'https://www.england.nhs.uk/revalidation/wp-content/uploads/sites/10/2014/05/rev-faqs-v6.pdf'
# local_vector_store = 'Agilisys - CallAgentDB'

if os.path.exists(local_vector_store):
    print('Local vector DB not found.')
    # Load the document
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    # Chunk the documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=350,separators=["\n\n", "\n", ". ", ""])
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=60, chunk_overlap=0,separators=["\n\n", "\n", ". ", ""])
    document = text_splitter.split_documents(docs)
    # Create a vector store
    db = Chroma.from_documents(document, OpenAIEmbeddings() , persist_directory=local_vector_store)
    db.persist()
    print("Local vector DB Created.")
else:
    print('Local vector DB found. Data Loaded.')
    # Load the persisted Chroma vector store
    db = Chroma(persist_directory=local_vector_store, embedding_function=OpenAIEmbeddings())

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

warnings.filterwarnings("ignore")



if not elevenlabs_api_key:
    raise ValueError("ELEVENLABS_API_KEY environment variable not set")

client = ElevenLabs(api_key=elevenlabs_api_key)

# Function to save audio data to a WAV file
def save_audio_to_wav(audio_data, filename, sample_rate, sample_width, channels):
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data)

# Function to record audio using PyAudio
def record_audio(duration, filename):
    sample_rate = 16000
    sample_width = 2
    channels = 1
    format = pyaudio.paInt16

    audio = pyaudio.PyAudio()
    stream = audio.open(format=format, channels=channels,
                        rate=sample_rate, input=True,
                        frames_per_buffer=1024)

    print("Recording...")
    frames = []

    for _ in range(0, int(sample_rate / 1024 * duration)):
        data = stream.read(1024)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    audio.terminate()

    audio_data = b''.join(frames)
    save_audio_to_wav(audio_data, filename, sample_rate, sample_width, channels)

# Function to transcribe audio using Whisper
def transcribe_with_whisper(audio_file_path):
    with open(audio_file_path, 'rb') as speech:
        transcript = openai.Audio.transcribe(
            model="whisper-1",
            file=speech
        )
    return transcript['text']

# Function to convert text to speech using ElevenLabs API (streaming)
def text_to_speech_streaming(text):
    logging.info("Calling TTS stream conversion")
    response_stream = client.text_to_speech.convert_as_stream(
        voice_id="ThT5KcBeYPX3keUQqHPh",  # Adam pre-made voice
        optimize_streaming_latency=1,
        output_format="mp3_22050_32",
        text=text,
        model_id="eleven_turbo_v2",
        voice_settings=VoiceSettings(
            stability=0.3,
            similarity_boost=0.4,
            style=0.5,
            use_speaker_boost=True,
        ),
    )
    # logging.info("Sending chunks to audio player.")
    return response_stream



# Function to play audio in real time as it is generated
def play_audio_streaming(audio_stream):
    audio_buffer = io.BytesIO()
    chunk_size = 128000  # Buffer size to accumulate larger portions of audio

    for chunk in audio_stream:
        audio_buffer.write(chunk)
        if len(audio_buffer.getvalue()) > chunk_size:
            audio_buffer.seek(1)
            audio_segment = AudioSegment.from_file(audio_buffer, format="mp3")
            play(audio_segment)
            audio_buffer.seek(1)
            audio_buffer.truncate(1)
    logging.info("Playing audio from buffer")
    # Play any remaining audio data in the buffer
    if audio_buffer.getvalue():
        audio_buffer.seek(0)
        audio_segment = AudioSegment.from_file(audio_buffer, format="mp3")
        play(audio_segment)

# Example usage of streaming text-to-speech and audio playback

# def handle_response(response_text):
#     audio_stream = text_to_speech_streaming(response_text)
#     play_audio_streaming(audio_stream)


# def handle_response_thread(response_text):
#     def thread_a(inputs, output_queue):
#         for item in inputs:
#             print('-'*10 , 'Sent to Elevenlabs : ',item)
#             result = text_to_speech_streaming(item)
#             output_queue.put(result)

#     def thread_b(input_queue, results):
#         while True:
#             item = input_queue.get()
#             if item is None:  # None is the signal to stop processing
#                 break
#             result = play_audio_streaming(item)
#             results.append(result)
#             input_queue.task_done()

#     input_text_list = [text.strip()+'.' for text in response_text.split('.') if text != '']

#     output_queue = queue.Queue()
#     results = []

#     thread_a = threading.Thread(target=thread_a, args=(input_text_list, output_queue))
#     thread_b = threading.Thread(target=thread_b, args=(output_queue, results))

#     # Starting threads
#     thread_a.start()
#     thread_b.start()

#     thread_a.join()

#     # Adding None to the queue to signal thread_b to stop processing
#     output_queue.put(None)

#     thread_b.join()
#     print('-'*100)

def handle_response_thread(response_text):
    input_text_list = [text.strip()+'.' for text in response_text.split('.') if text != '']

    queue_a_to_b = queue.Queue()
    results = []

    # Sample function a
    def function_a(input_data, output_queue):
        # Processing the input data
        result = text_to_speech_streaming(input_data)
        output_queue.put(result)

    # Sample function b
    def function_b(input_queue):
        while True:
            input_data = input_queue.get()
            if input_data is None:  # End of processing signal
                break
            result = play_audio_streaming(input_data)
            results.append(result)
            input_queue.task_done()

    # Creating threads for functions a and b
    thread_a = threading.Thread(target=lambda: [function_a(input_data, queue_a_to_b) for input_data in input_text_list])
    thread_b = threading.Thread(target=function_b, args=(queue_a_to_b,))

    # Starting the threads
    thread_a.start()
    thread_b.start()

    # Joining the threads
    thread_a.join()

    # Sending a signal to stop the function b thread
    queue_a_to_b.put(None)

    thread_b.join()
    print('-'*100)

def get_cur_time():
    return datetime.now().time().isoformat()


logging.info("DB population.")

prompt = ChatPromptTemplate.from_template(
    """
As a seasoned contact center representative and call assistant, your goal is to provide accurate and helpful information based on the context provided. When responding to user inquiries, please adhere to the following instructions:

1. Use British English in all responses .
3. Begin your replies directly with the information the user needs but First address the question in short, without any initial greetings or identifiers like "response" or "answer."
4. Maintain a natural and conversational tone, as if you're speaking to the customer in real-time.
5. Keep your responses polite and professional.
6. Ensure your answers are concise and to the point in short.

Context:
{context}

User Inquiry:
{input}
    """
)

prompt2 = ChatPromptTemplate.from_template(
    """
    You are an helpful call assistant, who is responsible for talking on calls like customer agent.
    """
)

messages = [{"role": "system", "content": f"{prompt2}"}]

logging.info("Calling openai model")
# Create an instance of OpenAI for language model operations
llm = OpenAIChat(model='gpt-4o')
logging.info("loading docs in RAG")
document_chain = create_stuff_documents_chain(llm, prompt)
# retriever = db.as_retriever(k=2)
retriever = VectorStoreRetriever(vectorstore=db)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

logging.info("Sending audio to TTS")
# Speak the introductory line before starting the main conversation loop
intro_audio_stream = text_to_speech_streaming("Hello, I am your contact center representative. How can I help you?")
play_audio_streaming(intro_audio_stream)

# Main conversation loop
while True:
    try:
        audio_file_path = 'speech.wav'
        record_audio(duration=4, filename=audio_file_path)
        user_input = transcribe_with_whisper(audio_file_path)

        if not user_input.strip():
            continue

        messages.append({"role": "user", "content": user_input})
        logging.info(f"You said: {user_input}")

        logging.info("Retrieving data from RAG")
        response = retrieval_chain.invoke({'input': user_input})

        messages.append({"role": "assistant", "content": response['answer']})
        logging.info(f"Assistant: {response['answer']}")

        # Handle response with streaming
        handle_response_thread(response['answer'])

    except Exception as e:
        print(f"Error: {e}")
        continue