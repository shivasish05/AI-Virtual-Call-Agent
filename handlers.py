# #
# #
from quart import Response, request
from twilio.twiml.voice_response import VoiceResponse
import logging
from rag import setup_rag
import asyncio

retrieval_chain = None
messages = []

async def initialize_rag():
    global retrieval_chain
    retrieval_chain = await setup_rag()

async def twilio_webhook():
    response = VoiceResponse()
    gather = response.gather(
        input='speech',
        action='/gather',
        method='POST',
        enhanced=True,
        speech_timeout='auto',
        speech_model="phone_call",
        language='en-GB',
        barge_in=True
    )
    gather.say(
        "<speak>Hello! <break time='500ms'/> I'm your contact center representative. How can I assist you today?</speak>",
        language='en-GB',
        voice='Google.en-GB-Neural2-C'
    )
    return Response(str(response), mimetype='text/xml')

async def gather():
    try:
        user_input = (await request.form).get('SpeechResult')

        if not user_input.strip():
            return await handle_empty_input()

        logging.info(f"You said: {user_input}")

        retrieval_task = asyncio.create_task(retrieve_answer(user_input))

        messages.append({"role": "user", "content": user_input})

        return await retrieval_task

    except Exception as e:
        logging.error(f"Error: {e}")
        return await handle_error()

async def retrieve_answer(user_input):
    try:
        logging.info("Retrieving data from RAG")

        response = await asyncio.to_thread(retrieval_chain.invoke, {'input': user_input})

        answer = response.get('answer', '').strip()
        logging.info(f"Assistant response: {answer}")

        if "speak slow" in user_input.lower():
            return await create_twilio_response_slow(answer)
        elif "speak fast" in user_input.lower():
            return await create_twilio_response_fast(answer)
        else:
            return await create_twilio_response(answer)

    except Exception as e:
        logging.error(f"Error during retrieval: {e}")
        return await handle_error()

async def convo():
    response = VoiceResponse()
    gather = response.gather(input='speech', action='/gather', method='POST')
    gather.say("")
    return Response(str(response), mimetype='text/xml')

async def health_check():
    return 'Working Fine!'

async def handle_empty_input():
    response = VoiceResponse()
    response.say(
        "<speak><prosody rate='medium'>I’m sorry, I didn’t quite catch that. Could you please repeat?</prosody></speak>",
        language='en-GB',
        voice='Google.en-GB-Neural2-C'
    )
    response.redirect('/twilio')
    return Response(str(response), mimetype='text/xml')

async def create_twilio_response(answer):
    response = VoiceResponse()
    gather = response.gather(
        input='speech',
        action='/gather',
        enhanced=True,
        method='POST',
        speech_timeout='auto',
        speech_model="phone_call",
        language='en-GB',
        barge_in=True
    )
    gather.say(
        f"<speak>{answer}</speak>",
        language="en-GB",
        voice="Google.en-GB-Neural2-C"
    )
    return Response(str(response), mimetype='text/xml')

async def create_twilio_response_slow(answer):
    response = VoiceResponse()
    gather = response.gather(
        input='speech',
        action='/gather',
        enhanced=True,
        method='POST',
        speech_timeout='auto',
        speech_model="phone_call",
        language='en-GB',
        barge_in=True
    )
    gather.say(
        f"<speak><prosody rate='slow'>{answer}</prosody></speak>",
        language="en-GB",
        voice="Google.en-GB-Neural2-C"
    )
    return Response(str(response), mimetype='text/xml')

async def create_twilio_response_fast(answer):
    response = VoiceResponse()
    gather = response.gather(
        input='speech',
        action='/gather',
        enhanced=True,
        method='POST',
        speech_timeout='auto',
        speech_model="phone_call",
        language='en-GB',
        barge_in=True
    )
    gather.say(
        f"<speak><prosody rate='fast'>{answer}</prosody></speak>",
        language="en-GB",
        voice="Google.en-GB-Neural2-D"
    )
    return Response(str(response), mimetype='text/xml')

async def handle_error():
    response = VoiceResponse()
    response.say(
        "<speak>An error occurred. Please try again later.</speak>",
        language='en-GB',
        voice='Google.en-GB-Neural2-C'
    )
    response.hangup()
    return Response(str(response), mimetype='text/xml')
