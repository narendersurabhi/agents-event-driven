from agents.jd_analysis import JDAnalysisAgent
from core.llm_client import OpenAILLMClient
from core.config import get_default_model

jd_text = """
We are looking for a Senior Machine Learning Engineer at Netflix to build
and scale personalization and recommendation systems using Python, PySpark,
and AWS. Experience with MLOps and experimentation platforms is required.
Nice to have: Kubernetes, feature stores, GenAI/LLM tooling.
"""

llm = OpenAILLMClient()
agent = JDAnalysisAgent(llm=llm, model=get_default_model())

result = agent.analyze(jd_text)
print(result.model_dump_json(indent=2))
