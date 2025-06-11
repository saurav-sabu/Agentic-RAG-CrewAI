import logging
from crewai import LLM, Agent
from tool.rag_tools import DocumentSearchTool

logger = logging.getLogger(__name__)

class Agents():
    def __init__(self):
        # Initialize the Agents class and set up the LLM model
        logger.info("Initializing Agents...")
        self.llm = LLM(model="gemini/gemini-2.0-flash")
        self.rag_tool = None  # Will be set later with a document search tool
        logger.info("Agents initialized.")

    def set_rag_tool(self, file_path: str):
        """
        Dynamically initialize the DocumentSearchTool with a file path.
        This allows the agent to search documents specified at runtime.
        """
        logger.info(f"Setting RAG tool with file: {file_path}")
        self.rag_tool = DocumentSearchTool(file_path)

    def retriever_agent(self):
        """
        Create and return an agent responsible for retrieving relevant information
        from the document or other sources using the RAG tool.
        """
        if not self.rag_tool:
            logger.error("RAG tool is not initialized. Call set_rag_tool(file_path) first.")
            raise ValueError("RAG tool is not initialized. Call set_rag_tool(file_path) first.")
        
        logger.info("Creating retriever agent.")
        return Agent(
            role="Retrieve the relevant information from the document or text",
            goal=(
                "Retrieve the most relevant information from the available sources "
                "for the user query: {query}. Always try to use the PDF search tool first. "
                "If you are not able to retrieve the information from the PDF search tool, "
                "then try to use the web search tool."
            ),
            backstory=(
                "You're a meticulous analyst with a keen eye for detail. "
                "You're known for your ability to understand user queries: {query} "
                "and retrieve knowledge from the most suitable knowledge base."
            ),
            verbose=True,
            tools=[self.rag_tool],  # Use the initialized RAG tool for document search
            llm=self.llm,           # Use the specified LLM model
            allow_delegation=False  # Do not allow delegation to other agents
        )

    def response_agent(self):
        """
        Create and return an agent responsible for synthesizing a response
        based on the information retrieved by the retriever agent.
        """
        if not self.rag_tool:
            logger.error("RAG tool is not initialized. Call set_rag_tool(file_path) first.")
            raise ValueError("RAG tool is not initialized. Call set_rag_tool(file_path) first.")
        
        logger.info("Creating response agent.")
        return Agent(
            role="Response synthesizer agent for the user query: {query}",
            goal=(
                "Synthesize the retrieved information into a concise and coherent response "
                "based on the user query: {query}. If you are not able to retrieve the "
                'information then respond with "I\'m sorry, I couldn\'t find the information '
                'you\'re looking for."'
            ),
            backstory=(
                "You're a skilled communicator with a knack for turning "
                "complex information into clear and concise responses."
            ),
            verbose=True,
            tools=[self.rag_tool],  # Use the initialized RAG tool for context
            llm=self.llm,           # Use the specified LLM model
            allow_delegation=False  # Do not allow delegation to other agents
        )
