from abc import ABC, abstractmethod
from pathlib import Path
from langchain.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import chainlit as cl

# Abstract base class for JSON applications
class AbstractJSONApplication(ABC):
    @abstractmethod
    def clean_json(self, file_path):
        pass

    @abstractmethod
    def run_chatbot(self, file_path):
        pass

# Subclass for cleaning JSON files
class JSONCleaner(AbstractJSONApplication):
    def clean_json(self, file_path):
        # Logic to clean and correct JSON formatting
        fixed_chars = []
        in_string = False
        escape_next = False
        i = 0
        data_string = Path(file_path).read_text()
        while i < len(data_string):
            char = data_string[i]
            if char == "'" and not in_string:
                fixed_chars.append('"')
            elif char == '"' and not escape_next:
                in_string = not in_string
                fixed_chars.append(char)
            elif char == '\\' and not escape_next:
                escape_next = True
                fixed_chars.append(char)
            elif escape_next:
                escape_next = False
                fixed_chars.append(char)
            elif not in_string and data_string[i:i+4] == 'None':
                fixed_chars.append('null')
                i += 3  # Skip the next 3 characters
            else:
                fixed_chars.append(char)
            i += 1
        fixed_string = ''.join(fixed_chars)
        end_of_json = fixed_string.rfind('}')
        return fixed_string[:end_of_json + 1]

    def run_chatbot(self, file_path):
        pass  # Not implemented in this subclass

# Subclass for running Chainlit chatbot
class ChainlitChatbot(AbstractJSONApplication):
    def clean_json(self, file_path):
        pass  # Not implemented in this subclass

    def run_chatbot(self, file_path):
        pass  # Logic is handled by Chainlit callbacks

# Chainlit callbacks
@cl.on_chat_start
async def start():
    chain = create_qa_chain()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, What is your query?"
    await msg.update()
    cl.user_session.set("chain", chain)

@cl.on_message
async def handle_message(message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=['FINAL', 'ANSWER']
    )
    message_text = message.content if hasattr(message, 'content') else str(message)
    res = await chain.acall(message_text, callbacks=[cb])
    answer = res["result"]
    await cl.Message(content=answer).send()

def load_and_split_json(file_path, jq_schema):
    loader = JSONLoader(file_path=file_path, jq_schema=jq_schema, text_content=False)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=300)
    return text_splitter.split_documents(data)

def initialize_embeddings_and_faiss(DBS_PATH, model_name, model_kwargs):
    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
    texts = load_and_split_json('cleanjson.json', '.')
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DBS_PATH)
    return embeddings, db

def create_qa_chain():
    DBS_PATH = "dbs/documentation/faiss_index"
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    model_kwargs = {'device': 'cpu'}
    embeddings, db = initialize_embeddings_and_faiss(DBS_PATH, model_name, model_kwargs)
    # Make sure the model path is correct for your system!
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    llm = LlamaCpp(
        model_path="llama-2-7b-chat.Q6_K.gguf",
        n_gpu_layers=1,
        n_batch=512,
        n_ctx=2048,
        f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
        callback_manager=callback_manager,
        verbose=True,
    )
    template = """Based on the provided document content in the JSON file, answer the following question. Only provide me with the most helpful answer'
    {context}
    Question: {question}
    Answer:"""
    prompt = PromptTemplate(input_variables=["context", "question"], template=template)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 1})
    return RetrievalQA.from_chain_type(llm, retriever=retriever, chain_type="stuff",  chain_type_kwargs={"prompt": prompt})

# Main function to run the application
def main():
    file_path = 'cleanjson.json'
    cleaner = JSONCleaner()
    cleaned_json = cleaner.clean_json('dirtyjson.json')
    with open(file_path, 'w') as f:
        f.write(cleaned_json)
    cl.run()

if __name__ == "__main__":
    main()
