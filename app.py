from flask import Flask, request, jsonify
from connection import connect_db, huggingFace_vectorizer, LLM_pipeline
from chunks import extract_text_from_pdf, ChunkData, CreateDataObjects
import weaviate.classes.config as wc
import weaviate.classes as wvc
import tempfile
import atexit
import gradio as gr
import requests

app = Flask(__name__)
client = connect_db()
atexit.register(lambda: client.close())

collection_name = "HR_doc"
if collection_name not in client.collections.list_all():
    client.collections.create(
        name=collection_name,
        vectorizer_config=huggingFace_vectorizer(),
        properties=[wc.Property(name="body", data_type=wc.DataType.TEXT)],
    )
collection = client.collections.get(collection_name)


@app.route("/upload", methods=["POST"])
def upload():
    try:
        if "file" not in request.files:
            return jsonify({"message": "No file part"}), 400
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"message": "No selected file"}), 400
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp:
            file.save(temp.name)
            text = extract_text_from_pdf(temp.name)
            chunks = ChunkData(text)
            CreateDataObjects(chunks, collection)
        return jsonify({"message": "PDF processed and data stored successfully."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        query = data.get("query", "")
        response = collection.query.near_text(
            query=query,
            limit=1,
            return_metadata=wvc.query.MetadataQuery(score=True, explain_score=True),
        )
        if response.objects:
            context = response.objects[0].properties["body"]
            prompt = f"Context: {context}\n\nQuestion: {query}"
            answer = LLM_pipeline(prompt)
            return jsonify({"response": answer})
        else:
            return jsonify(
                {"response": "I'm sorry, I couldn't find relevant information."}
            )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/end_chat", methods=["POST"])
def end_chat():
    try:
        if collection_name in client.collections.list_all():
            client.collections.delete(collection_name)
            return jsonify(
                {"message": f"Collection '{collection_name}' deleted successfully."}
            )
        else:
            return (
                jsonify({"message": f"Collection '{collection_name}' does not exist."}),
                404,
            )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# === Gradio UI setup ===

chatbot_name = "Lumina: Your HR Policy Assistant"


def chat_with_bot(user_input, history):
    response = requests.post("http://localhost:5000/chat", json={"query": user_input})
    bot_reply = response.json().get("response", "")
    history.append((user_input, bot_reply))
    return "", history


def upload_pdf(file_path):
    with open(file_path, "rb") as f:
        files = {"file": (file_path, f, "application/pdf")}
        response = requests.post("http://localhost:5000/upload", files=files)
    return response.json().get("message", "Upload failed.")


def end_chat_gradio():
    response = requests.post("http://localhost:5000/end_chat")
    return response.json().get("message", "Failed to end chat.")


# Gradio interface defined inside Flask route
@app.route("/", methods=["GET"])
def gradio_interface():
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
                send = gr.Button("Send")
                end = gr.Button("End Chat")
                end_status = gr.Textbox(label="Chat Status")

        upload_button.click(upload_pdf, inputs=pdf_file, outputs=upload_status)
        send.click(chat_with_bot, inputs=[msg, chatbot], outputs=[msg, chatbot])
        end.click(end_chat_gradio, outputs=end_status)

    demo.launch(share=True, inbrowser=True, server_name="0.0.0.0", server_port=7860)

    return "Gradio interface is running at /"


# === Run Flask ===
if __name__ == "__main__":
    app.run(debug=True, port=5000)
