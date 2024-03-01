import os
from dotenv import load_dotenv
from langchain.document_loaders.text import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter

load_dotenv()


if __name__ == "__main__":
    print("Hello World")
    loader = TextLoader(
        "/Users/sahilmobaidin/Desktop/myprojects/vectordb-db-intro/medium-blogs/mediumblog.txt"
    )
    document = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    text = text_splitter.split_documents(document)
    print(len(text))

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
