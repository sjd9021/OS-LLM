import os
from langchain.document_loaders import PyPDFLoader, JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain,RetrievalQA
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import LlamaCpp


DBS_PATH = "dbs/documentation/faiss_index"
pdf = "cleanjson.json"
loader = JSONLoader('cleanjson.json', '.', text_content=False)
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 200
)
texts = text_splitter.split_documents(docs)
# s
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})
db = FAISS.from_documents(texts, embeddings)
db.save_local(DBS_PATH)

# db = FAISS.load_local("dbs/documentation/faiss_index", embeddings)

# custom_prompt_template = """Use the following pieces of information to answer the user's question. If you don't
# know the answer, please just say you don't know the answer; don't try to make up an answer.

# {context}
# Question: {question}

# Only return the helpful answer below and nothing else.
# """
# prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])

# retriever=db.as_retriever(search_type="similarity", search_kwargs={"k": 1})
# n_gpu_layers = 1  # Metal set to 1 is enough.
# n_batch = 512  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
# callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# # Make sure the model path is correct for your system!
# llm = LlamaCpp(
#     model_path="llama-2-7b-chat.Q6_K.gguf",
#     n_gpu_layers=1,
#     n_batch=512,
#     n_ctx=2048,
#     f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
#     callback_manager=callback_manager,
#     verbose=True,
# )


# qa_chain = RetrievalQA.from_chain_type(
#     llm=llm,
#     chain_type="stuff",
#     retriever= retriever,
#     return_source_documents=True,
#     chain_type_kwargs={'prompt': prompt}
# )    

# question = str('Tell me something about this pdf')
# qa_result = qa_chain({"query": question})
# print(qa_result['result'])