import os
from langtrace_python_sdk import langtrace
langtrace.init(api_key = os.getenv("LANGTRACE_API_KEY"))
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai_tools import SerperDevTool,ScrapeWebsiteTool
from typing import List

# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class CompanyProfiler():
    """CompanyProfiler crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    def __init__(self, output_file=None):
        self.llm = LLM(
            model=os.getenv("MODEL", "gemini/gemini-2.5-flash"),
            api_key=os.getenv("GEMINI_API_KEY")
        )
        self.groq_llm = LLM(
            model=os.getenv("THINKING", "openrouter/qwen/qwen3-235b-a22b:free"),
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            min_p=0
        )


    @agent
    def strategist(self) -> Agent:
        return Agent(
            config=self.agents_config['strategist'], # type: ignore[index]
            verbose=True,
            llm=self.groq_llm,
            tools=[
                SerperDevTool(),
                ScrapeWebsiteTool()
            ],
            inject_date=True,
            reasoning=True,
            allow_delegation=True,
            max_reasoning_attempts=10,
            max_rpm=5,
            max_iter=10
        )

    @agent
    def analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['analyst'], # type: ignore[index]
            verbose=True,
            llm=self.groq_llm,
            tools=[
                SerperDevTool(),
                ScrapeWebsiteTool()
            ],
            inject_date=True,
            reasoning=True,
            allow_delegation=False,
            max_reasoning_attempts=10,
            max_rpm=5,
            max_iter=10
        )

    @agent
    def marketer(self) -> Agent:
        return Agent(
            config=self.agents_config['marketer'], # type: ignore[index]
            verbose=True,
            llm=self.groq_llm,
            tools=[
                SerperDevTool(),
                ScrapeWebsiteTool()
            ],
            inject_date=True,
            reasoning=True,
            allow_delegation=False,
            max_reasoning_attempts=10,
            max_rpm=5,
            max_iter=10
        )

    @agent
    def product_manager(self) -> Agent:
        return Agent(
            config=self.agents_config['product_manager'], # type: ignore[index]
            verbose=True,
            llm=self.groq_llm,
            tools=[
                SerperDevTool(),
                ScrapeWebsiteTool()
            ],
            inject_date=True,
            reasoning=True,
            allow_delegation=False,
            max_reasoning_attempts=10,
            max_rpm=5,
            max_iter=10
        )

    @agent
    def sales(self) -> Agent:
        return Agent(
            config=self.agents_config['sales'], # type: ignore[index]
            verbose=True,
            llm=self.groq_llm,
            tools=[
                SerperDevTool(),
                ScrapeWebsiteTool()
            ],
            inject_date=True,
            reasoning=True,
            allow_delegation=False,
            max_reasoning_attempts=10,
            max_rpm=5,
            max_iter=10
        )


    @task
    def project_task(self) -> Task:
        return Task(
            config=self.tasks_config['project_task'], # type: ignore[index]
        )

    @task
    def competitor_identification(self) -> Task:
        return Task(
            config=self.tasks_config['competitor_identification'], # type: ignore[index]
        )

    @task
    def metrics_identification(self) -> Task:
        return Task(
            config=self.tasks_config['metrics_identification'], # type: ignore[index]
        )

    @task
    def financial_market_data_collection(self) -> Task:
        return Task(
            config=self.tasks_config['financial_market_data_collection'], # type: ignore[index]
        )

    @task
    def porduct_and_feature_data(self) -> Task:
        return Task(
            config=self.tasks_config['porduct_and_feature_data'], # type: ignore[index]
        )

    @task
    def social_media_data(self) -> Task:
        return Task(
            config=self.tasks_config['social_media_data'], # type: ignore[index]
        )

    @task
    def customer_and_sales_data(self) -> Task:
        return Task(
            config=self.tasks_config['customer_and_sales_data'], # type: ignore[index]
        )

    @task
    def data_analysis_and_swot(self) -> Task:
        return Task(
            config=self.tasks_config['data_analysis_and_swot'], # type: ignore[index]
        )

    @task
    def report_generation(self) -> Task:
        return Task(
            config=self.tasks_config['report_generation'], # type: ignore[index]
            output_file='src/company_profiler/Reports/Report.md'
        )

    @crew
    def crew(self) -> Crew:
        """Creates the CompanyProfiler crew""" 

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.hierarchical,
            verbose=True,
            manager_agent=self.strategist(), # type: ignore[index]
            max_rpm=30,
            planning=True,
            cache=True,
            embedder={      #If you're using groq or open router keep embedder otherwise remove it
            "provider": "huggingface",
            "config": {
                "model": 'sentence-transformers/all-MiniLM-L6-v2'
                        }
            }
            
        )
