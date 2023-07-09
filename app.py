import os, streamlit as st

from llama_index import GPTSimpleVectorIndex, SimpleDirectoryReader, LLMPredictor, PromptHelper, ServiceContext
from langchain.llms.openai import OpenAI

# Uncomment to specify your OpenAI API key here, or add corresponding environment variable (recommended)
# os.environ['OPENAI_API_KEY']= "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"

# Provide openai key from the frontend if you are not using the above line of code to seet the key
openai_api_key = st.sidebar.text_input(
    label="#### Your OpenAI API key 👇",
    placeholder="Paste your openAI API key, sk-",
    type="password")

directory_path = st.sidebar.text_input(
    label="#### Your data directory path 👇",
    placeholder="C:\data",
    type="default")

def get_response(query,directory_path,openai_api_key):
    # This example uses text-davinci-003 by default; feel free to change if desired. 
    # Skip openai_api_key argument if you have already set it up in environment variables (Line No: 7)
    llm_predictor = LLMPredictor(llm=OpenAI(openai_api_key=openai_api_key, temperature=0, model_name="text-davinci-003"))

    # Configure prompt parameters and initialise helper
    max_input_size = 4096
    num_output = 256
    max_chunk_overlap = 20

    prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

    if os.path.isdir(directory_path): 
        # Load documents from the 'data' directory
        documents = SimpleDirectoryReader(directory_path).load_data()
        service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)
        index = GPTSimpleVectorIndex.from_documents(documents, service_context=service_context)
        
        response = index.query(query)
        if response is None:
            st.error("Oops! No result found")
        else:
            st.success(response)
    else:
        st.error(f"Not a valid directory: {directory_path}")

# Define a simple Streamlit app
st.title("DocExtractor")
query = st.text_input("What would you like to ask?", "")

# If the 'Submit' button is clicked
if st.button("Submit"):
    if not query.strip():
        st.error(f"Please provide the search query.")
    else:
        try:
            if len(openai_api_key) > 0:
                get_response(query,directory_path,openai_api_key)
            else:
                st.error(f"Enter a valid openai key")
        except Exception as e:
            st.error(f"An error occurred: {e}")
