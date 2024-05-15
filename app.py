from flask import Flask, render_template, request
import os
import openai
import threading
import sounddevice as sd
import numpy as np
import wavio
from playsound import playsound
import ipywidgets as widgets
from IPython.display import display

app = Flask(__name__)

# Set up OpenAI API credentials
openai.api_key = 'YOUR_OPENAI_API_KEY'


# Define the default route to return the index.html file
@app.route("/")
def index():
    return render_template("index.html")


# Function to get and configure the conversation chain
def get_chain():
    # Initialize the OpenAI model with a specified temperature.
    # Temperature set to 0 for deterministic, consistent responses.
    llm = openai.LanguageModel.create(
        engine="text-davinci-003",
        temperature=0
    )

    # Create a conversation buffer memory to keep track of the conversation.
    # This includes prefixes to distinguish between the AI tutor and the human user.
    memory = openai.ConversationBufferMemory(
        ai_prefix="AI Tutor:",
        human_prefix="Human:"
    )

    # Define a template for the conversation prompt.
    # This template sets the context for the conversation and instructions for the AI.
    prompt_template = """
    The following is a friendly conversation between a human and an AI.
    The AI a top-tier English tutor with years of experience.
    The AI is talking to a student who wants to practice speaking English. 
    The AI is to help the student practice speaking English by having a conversation. 

    The AI should feel free to correct the student's grammar and pronunciation and/or suggest different words or phrases to use whenever the AI feels needed.
    And when the AI corrects the student, the AI must start the sentence with "it is better to put it this way"
    But even when you correct the student, try to make a conversation first, and then correct the student

    Current conversation:
    {history}
    Human: {input}
    AI Tutor:"""

    # Create a PromptTemplate object with the defined prompt template.
    # This template includes variables for the conversation history and the latest human input.
    conversation_prompt = openai.PromptTemplate(input_variables=["history", "input"], template=prompt_template)

    # Initialize the conversation chain.
    # This chain uses the defined prompt, the language model (llm), and the conversation memory.
    conversation_chain = openai.ConversationChain(
        prompt=conversation_prompt,
        llm=llm,
        verbose=True,
        memory=memory
    )

    # Return the configured conversation chain.
    return conversation_chain


def get_transcript(file_path):
    # Open the audio file in binary read mode
    audio_file = open(file_path, "rb")

    # Use the OpenAI Whisper model to transcribe the audio
    transcript = openai.AudioTranscription.create(
        model="whisper-1",  # Specifies the Whisper model to use
        file=audio_file,  # Passes the audio file to the API
        response_format="text"  # Requests the transcription in text format
    )

    # Return the transcription
    return transcript


def get_gpt_response(transcript):
    # Talk to the AI Tutor via langchain
    conversation = get_chain()
    answer = conversation.predict(input=transcript)

    # Return the AI's message content
    return answer


def play_gpt_response_with_tts(gpt_response):
    # Generate speech from the GPT response using TTS

    response = openai.AudioSpeech.create(
        model="tts-1",  # Specifies the TTS model to use
        voice="alloy",  # Chooses a specific voice for the TTS
        input=gpt_response  # The text input to be converted to speech
    )

    # Stream the audio to a file
    response.stream_to_file(speech_file_path)
    # response.write_to_file(speech_file_path)

    # Play the generated speech audio
    playsound(speech_file_path)

    # Remove the temporary speech file to clean up
    os.remove(speech_file_path)


def talk_to_gpt(file_path):
    # Transcribe user speech to text
    user_transcript = get_transcript(file_path)

    # Get the GPT tutor's response to the user's transcript
    # Uses only the last 10 messages in history for context
    gpt_response = get_gpt_response(user_transcript)

    # Play the GPT response using OpenAI's TTS API
    play_gpt_response_with_tts(gpt_response=gpt_response)


# Path to temporarily store the generated speech file
speech_file_path = "./speech.wav"

# Import necessary classes from the langchain and langchain_openai libraries
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate
from langchain_openai import OpenAI

# Initialize the OpenAI model with a specified temperature.
# Temperature set to 0 for deterministic, consistent responses.
llm = OpenAI(
    temperature=0
)

# Create a conversation buffer memory to keep track of the conversation.
# This includes prefixes to distinguish between the AI tutor and the human user.
memory = ConversationBufferMemory(
    ai_prefix="AI Tutor:",
    human_prefix="Human:"
)


@app.route("/api", methods=["POST"])
def api():
    # Get the audio file from the POST request
    file = request.files['file']
    file_path = f"./{file.filename}"
    file.save(file_path)
    # Process the audio file through the AI tutor
    talk_to_gpt(file_path)
    os.remove(file_path)  # Remove the temporary audio file
    return "OK"


if __name__ == '__main__':
    app.run(debug=True)
