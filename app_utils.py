from transformers import pipeline
from pypdf import PdfReader # used to extract text from pdf
from langchain.text_splitter import CharacterTextSplitter # split text in smaller snippets
#import os # read API key from environment variables. Not required if you are specifying the key in notebook.
from openai import OpenAI # used to access openai api
import json # used to create a json to store snippets and embeddings
from numpy import dot # used to match user questions with snippets.
import openai
import glob
import streamlit as st

EXTRACTED_TEXT_FILE_PATH = "pdf_text.txt" # text extracted from pdf
EXTRACTED_JSON_PATH = "extracted.json" # snippets and embeddings
OPENAI_API_KEY = st.secrets("OPENAI")#"sk-proj-QZ0dcGXN1l89JwEmQzXOT3BlbkFJxDduyUsfDNE7e3cO91U8"#"use own API token" # replace this with your openai api key or store the api key in env
EMBEDDING_MODEL = "text-embedding-ada-002" # embedding model used

GPT_MODEL = "gpt-3.5-turbo-0125" # gpt model used. alternatively you can use gpt-4 or other models.
CHUNK_SIZE = 1000 # chunk size to create snippets
CHUNK_OVERLAP = 200 # check size to create overlap between snippets
CONFIDENCE_SCORE = 0.75 # specify confidence score to filter search results. [0,1] prefered: 0.75
K=5 #max number of relevant snippets to consider
pdf_description = """ User Guide"""





@st.cache_resource
def load_pdf(folder_path="."):

  #get all pdf files
  pdf_files = glob.glob(folder_path+"/*.pdf")
  # Initialize a variable to store all the text
  full_text = ""

  #iterate over each pdf file
  for pdf_file in pdf_files:
    with open(pdf_file, 'rb') as f:
        reader = PdfReader(f)

        # Iterate over each page
        for page in reader.pages:
          page_text = page.extract_text()
          if page_text.strip():
                full_text += page_text + "\n"
  full_text += "\n".join(line for line in full_text.splitlines() if line.strip())  # Extract text and add a newline after each page

  text_file_path = "pdf_text.txt"

  with open(text_file_path, "w", encoding="utf-8") as f:
              f.write(full_text)
  return




@st.cache_resource
# Initialize the question answering pipeline
def create_embeddings(file_path="pdf_text.txt"):

    # Initialize a list to store text snippets
    snippets = []
    # Initialize a CharacterTextSplitter with specified settings
    text_splitter = CharacterTextSplitter(separator="\n",
                                         chunk_size=CHUNK_SIZE,
                                         chunk_overlap=CHUNK_OVERLAP,
                                         length_function=len)

    # Read the content of the file specified by file_path
    with open(file_path, "r", encoding="utf-8") as f:
            file_text = f.read()

    # Split the text into snippets using the specified settings
    snippets = text_splitter.split_text(file_text)



    # Initialize OpenAI Client
    client = openai.OpenAI(api_key=OPENAI_API_KEY)


    # Request embeddings for the snippets using the specified model
    response = client.embeddings.create(input=snippets,model=EMBEDDING_MODEL)

    # Extract embeddings from the API response
    embedding_list = [response_object.embedding for response_object in response.data]

    # Create a JSON object containing embeddings and snippets
    embedding_json = {
        'embeddings': embedding_list,
        'snippets': snippets
    }

    # Convert the JSON object to a formatted JSON string
    json_object = json.dumps(embedding_json, indent=4)

    # Write the JSON string to a file specified by EXTRACTED_JSON_PATH
    with open(EXTRACTED_JSON_PATH, 'w', encoding="utf-8") as f:
        f.write(json_object)


def get_embeddings():

    # Open the JSON file containing embeddings and snippets
    with open(EXTRACTED_JSON_PATH,'r') as file:
        # Load the JSON data into a Python dictionary
        embedding_json = json.load(file)

    # Return the embeddings and snippets from the loaded JSON
    return embedding_json['embeddings'], embedding_json['snippets']



def user_question_embedding_creator(question):

    client = openai.OpenAI(api_key=OPENAI_API_KEY)

    # Request embedding for the provided question using the specified model
    response = client.embeddings.create(input=question,model=EMBEDDING_MODEL)

    # Extract and return the embedding from the API response
    return response.data[0].embedding




def answer_users_question(user_question):

    try:
        # Create an embedding for the user's question
        user_question_embedding = user_question_embedding_creator(user_question)
    except Exception as e:
        # Handle any exception that occurred while using Embedding API.
        return f"An error occurred while creating embedding: {str(e)}"

    embeddings, snippets = get_embeddings()
    # Calculate cosine similarities between the user's question embedding and the document embeddings
    cosine_similarities = []
    for embedding in embeddings:
        cosine_similarities.append(dot(user_question_embedding,embedding))

    # Pair snippets with their respective cosine similarities and sort them by similarity
    scored_snippets = zip(snippets, cosine_similarities)
    sorted_snippets = sorted(scored_snippets, key=lambda x: x[1], reverse=True)

    # Filter snippets based on a confidence score and select the top 5 results
    formatted_top_results = [snipps for snipps, _score in sorted_snippets if _score > CONFIDENCE_SCORE]
    if len(formatted_top_results) > K:
        formatted_top_results = formatted_top_results[:K]

    # Create the chatbot system using pdf_description provided by the user.
    chatbot_system = f"""You are provided with SEARCH RESULTS from a pdf. This pdf is a {pdf_description}. You need to generate answer to the user's question based on the given SEARCH RESULTS. SEARCH RESULTS as a python list. SEARCH RESULTS and USER's QUESTION are delimited by ``` \n If there is no information available, or question is irrelevent respond with - "Sorry! I can't help you." """

    # Create the prompt using results and user's question.
    prompt = f"""\
    SEARCH RESULTS:
    ```
    {formatted_top_results}
    ```
    USER'S QUESTION:
    ```
    {user_question}
    ```

    """

    # Prepare the chat conversation and use GPT model for generating a response
    messages = [{'role':'system', 'content':chatbot_system},
                {'role':'user', 'content':prompt}]

    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        completion = client.chat.completions.create(model=GPT_MODEL,
                                             messages=messages,
                                             temperature=0,
                                             stream=False)
    except Exception as e:
        # Handle exception while communicating with ChatCompletion API
        return f"An error occurred with chatbot: {str(e)}"

    # Return the chatbot response.
    return completion.choices[0].message.content



