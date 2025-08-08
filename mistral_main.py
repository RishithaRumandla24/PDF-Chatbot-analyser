from flask import Flask, render_template, request, jsonify, session
from flask_session import Session
import ollama
from llama_index.core import Settings, load_index_from_storage, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os
import json
import uuid
import re
import traceback
import time
from dotenv import load_dotenv
 
# === Load environment ===
load_dotenv()
 
# === Flask Config ===
app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)
 
# === LlamaIndex Settings ===
os.environ["LLAMA_INDEX_LLM"] = "none"
os.environ["LLAMA_INDEX_EMBEDDING_MODEL"] = "huggingface"
hf_embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5")
Settings.embed_model = hf_embed_model
Settings.llm = None
 
suggested_questions = {
    "Patient Engage": [
        "How to run the database replication setup scripts",
        "Explain about Epic API Settings clearly",
        "Mention the Message Pruning Settings",
        "Explain the different setting in Data Pruner Settings Tab",
        "Explain about IVR Exit Points",
        "Describe about the Pharmacy Settings",
        "Explain about the Call Steering functionality",
    ],
    "Patient Notify": [
        " Configuring Patient Notify with Avaya",
        " Configuring Patient Notify with Five9",
        " Configuring Patient Notify with IMI",
        " Configuring Patient Notify with Pinpoint",
        " Configuring Patient Notify with Twilio",
        "Disposition Codes and Error Codes",
        "Troubleshooting guide - PN",
    ],
    " toy": [
        "AWS Deployment of Core PAC Components",
        "How to create User - Service Account in Five9 Integration",
        "Configuring Patient Notify with Nice CxOne",
        "Configuring Patient Notify with Twilio using SpinSci Twilio Tenant",
        "Configuring Patient Notify with WebEx Connect",
        "Purpose Of Mirth Crons Setup",
        "Configuring Patient Notify with Nice CxOne Dialer",
    ]
}
 
# === Globals ===
query_engine = None
selected_index = "None"
 
# === Helper: Load index from persisted_index folder ===
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
 
def load_index(index_name):
    persist_dir = os.path.join("persisted_index", index_name)
    chroma_client = chromadb.PersistentClient(path=persist_dir)
    collection = chroma_client.get_or_create_collection("default")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(
        persist_dir=persist_dir,
        vector_store=vector_store
    )
    return load_index_from_storage(storage_context).as_query_engine()
 
# === Routes ===
 
@app.route('/')
def home():
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    return render_template('new_interface.html', selected_pickle=selected_index)
 
@app.route('/healthcheck')
def health():
    return "Success"
 
@app.route('/get_pickles')
def get_indexes():
    folders = [f.name for f in os.scandir('persisted_index') if f.is_dir()]
    return jsonify(folders)
 
@app.route('/select_pickle', methods=['POST'])
def select_index():
    global query_engine, selected_index
    data = request.get_json()
    selected_index = data.get('file')
    try:
        query_engine = load_index(selected_index)
        questions = suggested_questions.get(selected_index, [])
        return jsonify({"success": True, "selected": selected_index, "questions": questions})
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500
 
@app.route('/chat', methods=['POST'])
def chat():
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
 
    user_input = request.json.get('message')
    if not user_input:
        return jsonify({'error': 'No message provided'}), 400
 
    try:
        # Query the index
        response_data = query_engine.query(user_input)
 
        # Check if the query engine returned relevant results
        has_source_nodes = hasattr(response_data, 'source_nodes') and len(response_data.source_nodes) > 0
        has_valid_response = hasattr(response_data, 'response') and response_data.response and response_data.response.strip() and "no relevant information" not in str(response_data).lower()
 
        if not (has_source_nodes or has_valid_response):
            return jsonify({
                'response': "I don't know the answer to this query as it is not covered by the available documents.",
                'user_id': session['user_id'],
                'followup_questions': [],
                'images': [],
                'urls': []
            })
 
        # Extract source document information from source_nodes
        source_docs = []
        if has_source_nodes:
            for node in response_data.source_nodes:
                doc_info = node.node.metadata.get('document_name', 'Unknown Document')
                page = node.node.metadata.get('page', 'Unknown Page')
                source_docs.append({
                    'document_name': doc_info,
                    'page': page
                })
 
        # Load SharePoint URL mappings
        with open('url.json', 'r') as f:
            url_data = json.load(f)
 
        # Find related resources (URLs) based on source documents
        url_path = []
        for doc in url_data.get("documents", []):
            doc_name = doc.get("document_name")
            if doc_name and any(doc_name in source_doc['document_name'] for source_doc in source_docs):
                url_path.append({
                    "document_name": doc_name,
                    "sharepoint_url": doc.get("sharepoint_url")
                })
 
        # Process only if relevant results are found
        prompt = f"""
        You are an AI assistant designed to help internal developers with well-structured, context-aware answers based **only** on the provided document context. Do not use any external knowledge or general information beyond what is provided in the context.
 
        Your task is to:
        - Read and understand the user query
        - Use **only** the provided context (which may include both document and OCR image data)
        - Deliver a helpful, clear, step-by-step human-readable explanation
        - If the context does not contain relevant information, state: "The provided documents do not contain information to answer this query."
        - Include references to the source documents (document name and page number) when applicable.
 
        ### Response Format
 
        **Your asking about:**  
        Briefly restate the user's intent.
 
        **Your intention:**  
        Clarify what the user seems to want from the query.
 
        **Here what I get:**  
        Provide a detailed explanation based **only** on the retrieved context.
        - Use numbered steps for processes or procedures
        - Use bullet points for key items or features
        - Use **bold** for important technical terms or keywords
        - Reference the source documents as: **Document Name** (Page X)
        - If the context is insufficient, state: "The provided documents do not contain information to answer this query."
 
        **User Query:**
        {user_input}
 
        **Context (from documents):**
        {response_data}
 
        **Source Documents:**
        {json.dumps(source_docs, indent=2)}
 
        **Image Insights (if available):**  
        - If image text is included in the context, summarize useful points in a readable format.
        - Ignore meaningless numbers or noise.
        - If OCR content is not interpretable, say: "Image text is not clearly interpretable."
        """
        response = invoke_with_retry(prompt)  # Use retry logic
        response_content = response if isinstance(response, str) else str(response)
 
        # Suggest follow-up questions
        followup_prompt = f"""
        You are an assistant helping developers explore technical topics based **only** on the provided response content.
 
        Based on this assistant response:
        \"\"\"
        {response_content}
        \"\"\"
 
        Suggest exactly 3 relevant and helpful follow-up technical questions the user might ask next, based **only** on the response content.
 
        Return only a JSON list like:
        [
            "What is the next step after this?",
            "Can you give an example of how this works?",
            "What are some common mistakes with this process?"
        ]
        """
        followup_response = invoke_with_retry(followup_prompt)  # Use retry logic
        followup_questions = []
        try:
            cleaned = followup_response.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned.replace("```json", "").replace("```", "").strip()
            followup_questions = json.loads(cleaned)
            if not isinstance(followup_questions, list):
                followup_questions = []
        except Exception as e:
            print("Follow-up parse failed:", e)
 
        # Find any referenced images
        image_paths = re.findall(r'images\\(.+\.(?:png|jpg|jpeg|gif|bmp|tiff|webp))', response_content)
 
        # --- NEW: Extract document names from the response_content ---
        doc_names_in_response = set()
        # Example: matches "Installing+Patient+Engage+-+Configuring+Agenta+Server.pdf" and "Pharmacy FAQ Questions.pdf"
        doc_names_in_response.update(re.findall(r'([A-Za-z0-9\-\+\(\) ]+\.(?:pdf|docx|xlsx))', response_content))
 
        # Add any document URLs from url.json that are referenced in the response
        for doc in url_data.get("documents", []):
            doc_name = doc.get("document_name")
            if doc_name and (doc_name in doc_names_in_response) and not any(u['document_name'] == doc_name for u in url_path):
                url_path.append({
                    "document_name": doc_name,
                    "sharepoint_url": doc.get("sharepoint_url")
                })
 
        return jsonify({
            'response': response_content,
            'user_id': session['user_id'],
            'followup_questions': followup_questions,
            'images': image_paths,
            'urls': url_path  # Now includes all referenced document links
        })
 
    except Exception as e:
        print("Error in /chat:\n", traceback.format_exc())
        return jsonify({'error': f"Chat failed: {str(e)}"}), 500
 
def invoke_with_retry(prompt, retries=3, delay=1):
    for attempt in range(retries):
        try:
            response = ollama.chat(model='mistral:latest', messages=[
                {
                    'role': 'user',
                    'content': prompt,
                },
            ])
            return response['message']['content']
        except Exception as e:
            print(f"Ollama request failed. Retrying in {delay} seconds... Error: {str(e)}")
            time.sleep(delay)
            delay *= 2  # Exponential backoff
    raise Exception("Failed after multiple retries due to Ollama connection issues.")
 
if __name__ == '__main__':
    app.run(debug=True)