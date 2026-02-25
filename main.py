import os

import gradio as gr
import wget
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI


class DocumentProcessor:
    file_name = "companyPolicies.txt"
    # Initialize history as an empty list

    def __init__(self):
        self.llm = ChatOpenAI(model_name="gpt-5-nano", temperature=1)
        self.history = []

    def upload_document(self):
        url = (
            "https://cf-courses-data.s3.us.cloud-object-storage."
            "appdomain.cloud/6JDbUb_L3egv_eOkouY71A.txt"
        )

        # Check if file already exists
        if not os.path.exists(self.file_name):
            # Use wget to download the file
            wget.download(url, out=self.file_name)
            print("\n file downloaded")
        else:
            print(f"\n file {self.file_name} already exists")

        # with open(FILE_NAME, 'r') as file:
        #     # Read the contents of the file
        #     contents = file.read()
        #     print(contents)

    def split_embedding_and_storing_document(self):
        loader = TextLoader(self.file_name)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)
        print(len(texts))

        embeddings = HuggingFaceEmbeddings()
        # store the embedding in docsearch using Chromadb
        self.docsearch = Chroma.from_documents(
            texts,
            embeddings,
            # persist_directory="./chroma_data"
        )
        print("document ingested")

    def make_conversational_retrieval_chain(self):
        memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True, output_key="answer"
        )
        self.qa = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.docsearch.as_retriever(),
            memory=memory,
            get_chat_history=lambda h: h,
            return_source_documents=True,
        )

    def ask_question_chain_type(self, query: str):

        prompt_template = """Use the information from the document to answer the question at the end. If you don't know the answer, just say that you don't know, definately do not try to make up an answer.

        {context}

        Question: {question}
        """

        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        chain_type_kwargs = {"prompt": PROMPT}

        qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.docsearch.as_retriever(),
            chain_type_kwargs=chain_type_kwargs,
            return_source_documents=False,
        )
        data = qa.invoke(query)

        return data

    def ask_question_llm(self, query: str):

        result = self.qa.invoke(
            {"question": query},
            # {"chat_history": self.history},
        )
        # self.history.append((query, result["answer"]))

        return result

    def process(self):
        self.upload_document()
        self.split_embedding_and_storing_document()
        self.make_conversational_retrieval_chain()
        # self.ask_question_chain_type("what is mobile policy?")
        # self.ask_question_chain_type("Can you summarize the document for me?")
        # self.ask_question_chain_type("Can I eat in company vehicles?")

        # self.ask_question_llm("Пожалуйста, в следующий раз ответ переводи на русский")
        # self.ask_question_llm("what is mobile policy?")
        # self.ask_question_llm("List points in it?")
        # self.ask_question_llm("What is the aim of it?")

        # history = []
        # while True:
        #     query = input("Question: ")

        #     if query.lower() in ["quit", "exit", "bye"]:
        #         print("Answer: Goodbye!")
        #         break

        #     result = self.ask_question_llm(query)

        #     print("Answer: ", result["answer"])

    def generate_response(self, query: str):
        return self.ask_question_llm(query)["answer"]


if __name__ == "__main__":
    doc_proc = DocumentProcessor()
    doc_proc.process()
    chat_application = gr.Interface(
        fn=doc_proc.generate_response,
        inputs=gr.Textbox(
            label="Input", lines=2, placeholder="Type your question here..."
        ),
        outputs=gr.Textbox(label="Output"),
        title="Watsonx.ai Chatbot",
        description="Ask any question and the chatbot will try to answer.",
    )
    chat_application.launch(server_name="127.0.0.1", server_port=7860)
