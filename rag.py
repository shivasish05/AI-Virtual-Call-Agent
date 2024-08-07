#
#
#
# import os
# import logging
# import asyncio
# # import random  # Commented out to remove fillers
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import OpenAIEmbeddings
# from langchain_community.vectorstores import Chroma
# from langchain_community.llms import OpenAIChat
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.vectorstores import VectorStoreRetriever
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.chains import create_retrieval_chain
# import openai
# from config import Config
#
# openai.api_key = Config.OPENAI_API_KEY
#
# # fillers = [  # Commented out to remove fillers
# #     "Sure, give me a sec while I provide you with the info.",
# #     "One moment, please.",
# #     "Just a moment, I'll have that information for you shortly.",
# #     "Let me check that for you.",
# #     "Please hold on while I find the details."
# # ]
#
# async def load_vector_store():
#     if not os.path.exists(Config.LOCAL_VECTOR_STORE):
#         loader = PyPDFLoader(Config.PDF_PATH)
#         docs = loader.load()
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=250, separators=["\n\n", "\n", ". ", ""])
#         document = text_splitter.split_documents(docs)
#         db = Chroma.from_documents(document, OpenAIEmbeddings(), persist_directory=Config.LOCAL_VECTOR_STORE)
#         db.persist()
#         logging.info("Local vector DB Created.")
#     else:
#         logging.info('Local vector DB found. Data Loaded.')
#         db = Chroma(persist_directory=Config.LOCAL_VECTOR_STORE, embedding_function=OpenAIEmbeddings())
#     return db
#
# async def setup_rag():
#     db = await load_vector_store()
#     retriever = VectorStoreRetriever(vectorstore=db)
#     llm = OpenAIChat(model='gpt-4o')
#     prompt = ChatPromptTemplate.from_template(
#         """
#         As a seasoned contact center representative and call assistant, your goal is to provide accurate and helpful information based on the context provided. When responding to user inquiries, please adhere to the following instructions:
#
#         1. Use British English in all responses.
#         2. Begin your replies directly with the information the user needs but first address the question in short, without any initial greetings or identifiers like "response" or "answer."
#         3. Maintain a natural and conversational tone, as if you're speaking to the customer in real-time.
#         4. Keep your responses polite and professional.
#         5. Ensure your answers are concise and to the point in short.
#         6. Respond appropriately to the user greetings and goodbyes. Do not unnecessarily add any extra information if it is not asked for along with the greetings.
#         7. Do NOT ANSWER any question unrelated to the document , if asked reply with a follow up question.
#         8. Also when asked to schedule a meeting take the input such as Name date and time  and store it in a json formate , "Name : Date: Time:" and say meeting scheduled after input taken properly.
#         Context:
#         {context}
#
#         User Inquiry:
#         {input}
#         """
#     )
#     document_chain = create_stuff_documents_chain(llm, prompt)
#     retrieval_chain = create_retrieval_chain(retriever, document_chain)
#     return retrieval_chain
#
# # async def generate_filler():  # Commented out to remove fillers
# #     await asyncio.sleep(1)
# #     return random.choice(fillers)
#
# async def retrieve_with_filler(user_input, retrieval_chain, messages):
#     # filler_task = asyncio.create_task(generate_filler())  # Commented out to remove fillers
#     retrieval_task = asyncio.create_task(retrieve_answer(user_input, retrieval_chain, messages))
#
#     done, pending = await asyncio.wait(
#         {retrieval_task},
#         return_when=asyncio.FIRST_COMPLETED
#     )
#
#     response = await retrieval_task
#
#     return response
#
# async def retrieve_answer(user_input, retrieval_chain, messages):
#     try:
#         logging.info("Retrieving data from RAG")
#
#         # Use asyncio.to_thread to run synchronous code in a separate thread
#         response = await asyncio.to_thread(retrieval_chain.invoke, {'input': user_input})
#
#         messages.append({"role": "assistant", "content": response['answer']})
#         logging.info(f"Assistant: {response['answer']}")
#
#         return response['answer']
#
#     except Exception as e:
#         logging.error(f"Error during retrieval: {e}")
#         return "An error occurred while retrieving the information from (RAG). Please try again later."
import os
import logging
import asyncio
import json
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import OpenAIChat
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import openai
from config import Config

openai.api_key = Config.OPENAI_API_KEY

meetings = []  # To store meeting details

async def load_vector_store():
    if not os.path.exists(Config.LOCAL_VECTOR_STORE):
        loader = PyPDFLoader(Config.PDF_PATH)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=250,
                                                       separators=["\n\n", "\n", ". ", ""])
        document = text_splitter.split_documents(docs)
        db = Chroma.from_documents(document, OpenAIEmbeddings(), persist_directory=Config.LOCAL_VECTOR_STORE)
        db.persist()
        logging.info("Local vector DB Created.")
    else:
        logging.info('Local vector DB found. Data Loaded.')
        db = Chroma(persist_directory=Config.LOCAL_VECTOR_STORE, embedding_function=OpenAIEmbeddings())
    return db

async def setup_rag():
    db = await load_vector_store()
    retriever = VectorStoreRetriever(vectorstore=db)
    llm = OpenAIChat(model='gpt-4o')
    prompt = ChatPromptTemplate.from_template(
        """
        As a seasoned contact center representative and call assistant, your goal is to provide accurate and helpful information based on the context provided. When responding to user inquiries, please adhere to the following instructions:

        1. Use British English in all responses.
        2. Begin your replies directly with the information the user needs but first address the question in short, without any initial greetings or identifiers like "response" or "answer."
        3. Maintain a natural and conversational tone, as if you're speaking to the customer in real-time.
        4. Keep your responses polite and professional.
        5. Ensure your answers are concise and to the point in short.
        6. Respond appropriately to the user greetings and goodbyes. Do not unnecessarily add any extra information if it is not asked for along with the greetings.
        7. Do NOT ANSWER any question unrelated to the document, if asked reply with a follow up question.
        8. Also when asked to schedule a meeting take the input such as Name, date, and time and store it in a json format, "Name: Date: Time:" and say meeting scheduled after input taken properly.
        Context:
        {context}

        User Inquiry:
        {input}
        """
    )
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    return retrieval_chain

import asyncio
import logging
import json

# Define meetings list globally or pass it as an argument
import asyncio
import logging
import json
import os

# Define meetings list globally or pass it as an argument
meetings = []


async def retrieve_answer(user_input, retrieval_chain):
    try:
        logging.info("Retrieving data from RAG")

        response = await asyncio.to_thread(retrieval_chain.invoke, {'input': user_input})

        answer = response.get('answer', '').strip()
        logging.info(f"Assistant response: {answer}")

        # Check if the response contains "meeting scheduled" in a case-insensitive manner
        if "Meeting scheduled" in answer.lower():
            print('entered the MS')

            # Extract the JSON part from the response
            try:
                # Extract JSON part from the response
                json_start = answer.find('```json\n') + len('```json\n')
                json_end = answer.find('\n```')
                json_part = answer[json_start:json_end].strip()
                logging.debug(f"Extracted JSON part: {json_part}")

                # Parse the extracted JSON part
                response_json = json.loads(json_part)

                # Construct meeting details
                details = {
                    'name': response_json.get('Name'),
                    'date': response_json.get('Date'),
                    'time': response_json.get('Time')
                }

                # Append details to meetings list
                meetings.append(details)
                print('meeting appended')

                meetings_file_path = 'meetings.json'  # Use relative path for simplicity
                try:
                    # Ensure directory exists and write the meetings list to the JSON file
                    os.makedirs(os.path.dirname(meetings_file_path), exist_ok=True)
                    with open(meetings_file_path, 'w') as f:
                        json.dump(meetings, f, indent=4)
                    logging.info(f"Meeting scheduled and stored: {details}")
                    return f"Meeting scheduled for {details['name']} on {details['date']} at {details['time']}"
                except IOError as e:
                    logging.error(f"Error writing to file {meetings_file_path}: {e}")
                    return "An error occurred while saving the meeting details. Please try again later."

            except (IndexError, json.JSONDecodeError) as e:
                logging.error(f"Error extracting or decoding JSON from response: {e}")
                return "An error occurred while processing the meeting details. Please try again later."

        return answer

    except Exception as e:
        logging.error(f"Error during retrieval: {e}")
        return "An error occurred while retrieving the information. Please try again later."


def extract_meeting_details(user_input):
    details = {"name": "", "date": "", "time": ""}

    name_match = re.search(r'schedule a meeting for (\w+)', user_input, re.IGNORECASE)
    date_match = re.search(r'on (\d{1,2} [a-zA-Z]+)', user_input, re.IGNORECASE)
    time_match = re.search(r'at (\d{1,2}:\d{2} [APM]{2})', user_input, re.IGNORECASE)

    if name_match:
        details["name"] = name_match.group(1).strip()
    if date_match:
        details["date"] = date_match.group(1).strip()
    if time_match:
        details["time"] = time_match.group(1).strip()

    logging.info(f"Extracted meeting details: {details}")
    return details if any(details.values()) else None
