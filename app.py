import gradio as gr
import requests
import tempfile
from connection import connect_db, huggingFace_vectorizer, LLM_pipeline
from chunks import extract_text_from_pdf, ChunkData, CreateDataObjects
import weaviate.classes.config as wc
import weaviate.classes as wvc
import os

# Gradio frontend and logic
chatbot_name = "Lumina: Your HR Policy Assistant"
client = connect_db()
collection_name = "HR_doc"

# Ensure collection exists
if collection_name not in client.collections.list_all():
    client.collections.create(
        name=collection_name,
        vectorizer_config=huggingFace_vectorizer(),
        properties=[wc.Property(name="body", data_type=wc.DataType.TEXT)],
    )
collection = client.collections.get(collection_name)


# Function to upload PDF and extract text
def upload_pdf(file_path):
    try:
        if not file_path:
            return "No file selected"

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp:
            with open(file_path, "rb") as f:
                temp.write(f.read())
            text = extract_text_from_pdf(temp.name)
            chunks = ChunkData(text)
            CreateDataObjects(chunks, collection)

        return "PDF processed and data stored successfully."
    except Exception as e:
        return f"Error processing file: {str(e)}"


# Function to interact with the chatbot
def chat_with_bot(user_input, history):
    try:
        # Query the Weaviate collection
        response = collection.query.near_text(
            query=user_input,
            limit=1,
            return_metadata=wvc.query.MetadataQuery(score=True, explain_score=True),
        )

        if response.objects:
            context = response.objects[0].properties["body"]
            prompt = f"Context: {context}\n\nQuestion: {user_input}"
            answer = LLM_pipeline(prompt)
            history.append((user_input, answer))
            return "", history
        else:
            return "I'm sorry, I couldn't find relevant information.", history
    except Exception as e:
        return f"Error in chat: {str(e)}", history


# Function to end the chat and delete the collection
def end_chat():
    try:
        if collection_name in client.collections.list_all():
            client.collections.delete(collection_name)
            return f"Collection '{collection_name}' deleted successfully."
        else:
            return f"Collection '{collection_name}' does not exist."
    except Exception as e:
        return f"Error ending chat: {str(e)}"


# Set up Gradio interface
with gr.Blocks() as demo:
    gr.Markdown(f"# {chatbot_name}")

    with gr.Row():
        with gr.Column():
            pdf_file = gr.File(label="Upload HR Policy PDF", type="filepath")
            upload_button = gr.Button("Upload")
            upload_status = gr.Textbox(label="Upload Status")
        with gr.Column():
            chatbot = gr.Chatbot()
            msg = gr.Textbox(label="Your Question")
            send_button = gr.Button("Send")
            end_button = gr.Button("End Chat")
            end_status = gr.Textbox(label="Chat Status")

    # Define interaction logic
    upload_button.click(upload_pdf, inputs=pdf_file, outputs=upload_status)
    send_button.click(chat_with_bot, inputs=[msg, chatbot], outputs=[msg, chatbot])
    end_button.click(end_chat, outputs=end_status)

# Launch Gradio interface
demo.launch(share=True, inbrowser=True, server_name="0.0.0.0", server_port=7860)
