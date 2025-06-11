# app.py
import streamlit as st
from datetime import datetime
from crewai import Crew
from agents import Agents
from tasks import Tasks
from crewai import LLM
import os
import tempfile

import logging

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("rag_app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Streamlit page configuration
st.set_page_config(page_title="RAG PDF Assistant", layout="wide")
st.title("📄 RAG-based Document Assistant")

class RAGApp:
    def __init__(self):
        # Initialize the RAGApp, LLM, placeholders, agents, and tasks
        logger.info("Initializing RAGApp")
        self.llm = LLM(model="gemini/gemini-2.0-flash")
        self.output_placeholder = st.empty()
        self.agents = Agents()
        self.tasks = Tasks()

    def run(self):
        # Sidebar UI for file upload and query input
        st.sidebar.header("📎 Upload and Query")
        
        file = st.sidebar.file_uploader("Upload PDF Document", type=["pdf"])
        user_query = st.sidebar.text_area("🔍 Enter your query", height=150)
        submit = st.sidebar.button("Search")

        if submit:
            logger.info("Search button clicked")
            # Check if file and query are provided
            if not file or not user_query.strip():
                logger.warning("No file uploaded or query is empty")
                st.warning("Please upload a document and enter a query.")
                return
            
            # Save uploaded file temporarily
            try:
                # Save uploaded file to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(file.read())
                    tmp_file_path = tmp_file.name  # Absolute file path
                logger.info("Uploaded file saved as temp_uploaded_doc.pdf")
            except Exception as e:
                logger.error(f"Failed to save uploaded file: {e}")
                st.error(f"Failed to save uploaded file: {e}")
                return

            # Initialize DocumentSearchTool with uploaded file path
            try:
                logger.info("Setting RAG tool with uploaded document")
                self.agents.set_rag_tool(tmp_file_path)

                # Create retriever and responder agents
                retriever = self.agents.retriever_agent()
                responder = self.agents.response_agent()
                
                # Define retrieval and response tasks
                retrieval_task = self.tasks.retrieval_task(retriever)
                response_task = self.tasks.response_task(responder)

                # Create Crew with agents and tasks
                crew = Crew(
                    agents=[retriever, responder],
                    tasks=[retrieval_task, response_task],
                    verbose=True
                )

                logger.info("Starting Crew kickoff")
                # Run the Crew workflow with user query as input
                result = crew.kickoff(inputs={"query": user_query})
                logger.info("Crew kickoff completed successfully")
                # Display the result in the app
                self.output_placeholder.markdown(f"### 🧾 Response\n\n{result}")

            except Exception as e:
                logger.error(f"Error during RAG process: {str(e)}")
                st.error(f"Error during RAG process: {str(e)}")

# Initialize and run the app
if __name__ == "__main__":
    logger.info("Starting RAG PDF Assistant app")
    app = RAGApp()
    app.run()
