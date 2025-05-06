import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import weaviate.classes as wvc
from weaviate.util import generate_uuid5
from dotenv import load_dotenv

load_dotenv()


def extract_text_from_pdf(pdf_path):

    reader = PdfReader(pdf_path)
    all_text = ""
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            all_text += text + "\n"
            print(f"Page {i + 1}: Text extracted")
        else:
            print(f"Page {i + 1}: No text found")
    return all_text


def ChunkData(text):

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.create_documents([text])
    return docs


def CreateDataObjects(docs, collection):

    objects = []
    for i, chunk in enumerate(docs):
        props = {
            "body": chunk.page_content,
        }
        chunk_id = generate_uuid5(i)
        data_object = wvc.data.DataObject(properties=props, uuid=chunk_id)
        objects.append(data_object)

    collection.data.insert_many(objects)
    print("Data inserted into Weaviate collection.")
