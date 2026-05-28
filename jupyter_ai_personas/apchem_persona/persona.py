from jupyter_ai.personas.base_persona import BasePersona, PersonaDefaults
from jupyterlab_chat.models import Message
from jupyter_ai.history import YChatHistory
from agno.agent import Agent
from agno.models.aws import AwsBedrock
import boto3
from langchain_core.messages import HumanMessage
from agno.team.team import Team
from agno.tools.python import PythonTools
from agno.tools.file import FileTools
from agno.tools.github import GithubTools
from .template import SoftwareTeamVariables, _SOFTWARE_TEAM_PROMPT_TEMPLATE

session = boto3.Session()

class APChemPersona(BasePersona):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def defaults(self):
        return PersonaDefaults(
            name="APCHEMPersona",
            avatar_path="/api/ai/static/jupyternaut.svg",
            description="An expert AP Chemistry educational and lab data science team for Jupyter notebooks.",
            system_prompt="I am an AP Chemistry teaching and lab simulation team. I coordinate specialized members: a curriculum planner who maps concepts to the AP framework, a chemistry coder who writes Python simulations and graphs, a grading tester who evaluates answers against official rubrics, and file/repository specialists to manage lab data. Together, we help you master chemistry and score a 5.",
        )

    def initialize_team(self, system_prompt):
        model_id = self.config_manager.lm_provider_params["model_id"]
        
        planner = Agent(
            name="curriculum_planner",
            role="AP Chemistry Curriculum Expert who maps problems to the 9 Units and Science Practices",
            model=AwsBedrock(id=model_id, session=session),
            instructions=[
                "Do not create new files unless explicitly asked by user.",
                "Analyze user requests and break them down into chemical concepts, reaction mechanisms, or multi-step math tasks.",
                "Ensure focus on the AP Big Ideas: Scale, Proportion, Structure, Transformations, and Energy."
            ],
            markdown=True,
            show_tool_calls=True
        )
        
        apchem = Agent(
            name="chem_coder",
            role="Chemistry Python programmer responsible for modeling data, kinetics, and titration curves",
            model=AwsBedrock(id=model_id, session=session),
            instructions=[
                "Do not create new files unless explicitly asked by user.",
                "Implement Python code using NumPy, SciPy, or Matplotlib to simulate chemical properties or chart curves.",
                "Write clean, efficient, and well-documented chemical simulation scripts.",
                "Follow Python best practices and PEP 8 style guidelines."
            ],
            tools=[PythonTools()],
            markdown=True,
            show_tool_calls=True
        )
        
        tester = Agent(
            name="grading_tester",
            role="AP Senior Exam Grader focused on free-response validation and scoring rubrics",
            model=AwsBedrock(id=model_id, session=session),
            instructions=[
                "Do not create new files unless explicitly asked by user.",
                "Validate code logic and chemical outputs against actual stoichiometric and thermodynamic laws.",
                "Check calculation steps for common student misconceptions (e.g., ignoring coefficients, sign errors in dH).",
                "Ensure proper units, significant figures, and correct use of the quadratic formula in equilibrium.",
                "Verify particulate-level reasoning for conceptual question responses.",
                "Grade user answers using criteria aligned with official AP scoring guidelines.",
                "Document missed points or structural calculation errors clearly.",
                "Test both synthetic lab datasets and user-input calculations for edge cases."
            ],
            tools=[PythonTools()],
            markdown=True,
            show_tool_calls=True
        )
        
        gitHub = Agent(
            name="lab_repo_specialist",
            role="GitHub operations specialist managing chemistry repository interactions and lab portfolios",
            model=AwsBedrock(id=model_id, session=session),
            instructions=[
                "Monitor and analyze GitHub repository activities and changes for chemistry lab code.",
                "Help with repository organization and maintenance of lab notebooks.",
                "Ensure proper Git workflow practices are followed.",
                "Handle branch management and merging strategies for collective lab projects.",
                "Provide insights on repository metrics and activity patterns."
            ],
            tools=[GithubTools()],
            markdown=True,
            show_tool_calls=True
        )
        
        fileManager = Agent(
            name="lab_data_manager",
            role="File manager that manages local chemical datasets, reading and writing lab logs.",
            model=AwsBedrock(id=model_id, session=session),
            instructions=[
                "Assist with local file management of data tables and CSV lab outputs.",
                "Only read a file when explicitly requested.",
                "Only write to a file when explicitly requested."
            ],
            tools=[FileTools()],
            markdown=True,
            show_tool_calls=True
        )
        
        dev_team = Team(
            name="ap-chem-team",
            mode="coordinate",
            members=[planner, apchem, tester, gitHub, fileManager],
            model=AwsBedrock(id=model_id, session=session),
            instructions=[
                "Chat history is " + system_prompt,
                "You are APChemGPT, an expert AP Chemistry teacher and senior exam grader.",
                "Your goal is to help the user master chemistry concepts, solve complex problems, and score a 5 on the AP Exam.",
                "TECHNICAL ACCURACY: Always use correct chemical notation, balanced equations, and precise phase labels (s, l, g, aq).",
                "STEP-BY-STEP PEDAGOGY: Break down multi-step calculations (like stoichiometry, equilibrium, or thermodynamics) sequentially.",
                "AP FRAMEWORK ALIGNMENT: Explicitly map your explanations to the 9 AP Chemistry Units and Science Practices. Focus heavily on Big Ideas: Scale, Proportion, Quantity, Structure, Properties, Transformations, and Energy.",
                "CODE INTEGRATION: When appropriate, provide Python code using libraries like NumPy, Matplotlib, or SciPy to model chemical data, plot titration curves, simulate kinetics, or calculate equilibrium constants.",
                "FORMATTING: Use clear Markdown, bold key terms, and format chemical formulas using LaTeX math mode (e.g., $\\text{H}_2\\text{O}$ or $\\text{K}_c = \\frac{[\\text{C}]^c}{[\\text{A}]^a[\\text{B}]^b}$).",
                "Tone: Encouraging, highly analytical, academic, and precise.",
                "Make sure to provide helpful tips while putting the user's requests first."
            ],
            markdown=True,
            show_members_responses=True,
            enable_agentic_context=True,
            add_datetime_to_instructions=True,
            show_tool_calls=True
        )
        return dev_team

    async def process_message(self, message: Message):
        message_text = message.body
        provider_name = self.config_manager.lm_provider.name
        model_id = self.config_manager.lm_provider_params["model_id"]
        
        history = YChatHistory(ychat=self.ychat, k=2)
        messages = await history.aget_messages()
        history_text = ""
        if messages:
            history_text = "\nPrevious conversation:\n"
            for msg in messages:
                role = "User" if isinstance(msg, HumanMessage) else "Assistant"
                history_text += f"{role}: {msg.content}\n"
                
        variables = SoftwareTeamVariables(
            input=message.body,
            model_id=model_id,
            provider_name=provider_name,
            persona_name=self.name,
            context=history_text
        )
        
        system_prompt = _SOFTWARE_TEAM_PROMPT_TEMPLATE.format_messages(**variables.model_dump())[0].content
        dev_team = self.initialize_team(system_prompt)
        
        response = dev_team.run(
            message_text,
            stream=False,
            stream_intermediate_steps=False,
            show_full_reasoning=True,
        )
        response = response.content

        async def response_iterator():
            yield response

        await self.stream_message(response_iterator())
