import requests
import streamlit as st

from src.config import settings

API_BASE_URL=f"http://localhost:{settings.api_port}"
st.set_page_config(page_title="RAG Document Search", page_icon="📄")


with st.sidebar:
    st.header("upload document")
    uploded_file=st.file_uploader("Choose a PDF or TXT file", type=["pdf", "txt"])
    if uploded_file is not None:
        if st.button("Upload & Index"):
            with st.spinner("Uploading and processing..."):
                files={"file":(uploded_file.name,uploded_file.getvalue())}
                try:
                    response=requests.post(f"{API_BASE_URL}/upload",files=files)
                    response.raise_for_status()
                    result=response.json()
                    st.success(f"Index'{result['filename']}' '({result['chunk_count']} chunks)'")

                except requests.exceptions.RequestException as e:
                    st.error(f"upload file:{e}")

    st.divider()
    st.header("Index document")
    try:
        doc_response=requests.get(f"{API_BASE_URL}/documents")
        doc_response.raise_for_status()
        documents=doc_response.json()
        if documents:
            for doc in documents:
                icon = {"processed": "✅", "pending": "⏳"}.get(doc["status"], "❌")
                st.write(f"{icon} {doc['filename']} ({doc['chunk_count']} chunks)")
        else:
            st.caption("No documents indexed yet.")
    except requests.exceptions.RequestException as e:
        st.error(f"Could not reach API: {e}")

st.subheader("Ask a question about your documents")
query_text = st.text_input("Your question")

if st.button("Search") and query_text:
    with st.spinner("Searching..."):
        try:
            response = requests.post(f"{API_BASE_URL}/query", json={"query": query_text})
            response.raise_for_status()
            answer = response.json()["answer"]
            st.markdown("### Answer")
            st.write(answer)
        except requests.exceptions.RequestException as e:
            st.error(f"Query failed: {e}")
    
            