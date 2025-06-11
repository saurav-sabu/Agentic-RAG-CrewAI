import logging
from crewai import Task
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Configure logging to display INFO level and above
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Tasks():
    """
    A class to define and create tasks for information retrieval and response synthesis.
    """

    def __init__(self):
        # Create a logger specific to this class
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Tasks class initialized.")

    def retrieval_task(self, agent):
        """
        Creates a retrieval task for the given agent.

        Args:
            agent: The agent responsible for executing the retrieval task.

        Returns:
            Task: An instance of the Task class configured for information retrieval.
        """
        # Log the creation of the retrieval task
        self.logger.info("Creating retrieval task for agent: %s", agent)
        # Define the retrieval task with a description and expected output
        retrieval_task = Task(
            description=(
                "Retrieve the most relevant information from the available "
                "sources for the user query: {query}"
            ),
            expected_output=(
                "The most relevant information in the form of text as retrieved "
                "from the sources."
            ),
            agent=agent
        )
        # Debug log with the created task details
        self.logger.debug("Retrieval task created: %s", retrieval_task)
        return retrieval_task

    def response_task(self, agent):
        """
        Creates a response synthesis task for the given agent.

        Args:
            agent: The agent responsible for generating the final response.

        Returns:
            Task: An instance of the Task class configured for response synthesis.
        """
        # Log the creation of the response task
        self.logger.info("Creating response task for agent: %s", agent)
        # Define the response task with a description and expected output
        response_task = Task(
            description=(
                "Synthesize the final response for the user query: {query}"
            ),
            expected_output=(
                "A concise and coherent response based on the retrieved information "
                "from the right source for the user query: {query}. If you are not "
                "able to retrieve the information, then respond with: "
                '"I\'m sorry, I couldn\'t find the information you\'re looking for."'
            ),
            agent=agent
        )
        # Debug log with the created task details
        self.logger.debug("Response task created: %s", response_task)
        return response_task
