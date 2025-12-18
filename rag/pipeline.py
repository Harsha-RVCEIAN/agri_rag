from typing import Dict
from rag.retriever import Retriever
from rag.prompt_builder import PromptBuilder
from rag.answer_generator import AnswerGenerator


class RAGPipeline:
    """
    End-to-end RAG runtime pipeline.
    """

    def __init__(self):
        self.retriever = Retriever()
        self.prompt_builder = PromptBuilder()
        self.answer_generator = AnswerGenerator()

    def run(
        self,
        query: str,
        intent: str | None = None,
        language: str | None = None
    ) -> Dict:
        # 1. Retrieve
        retrieval = self.retriever.retrieve(
            query=query,
            intent=intent,
            language=language
        )

        chunks = retrieval.get("chunks", [])

        # 2. Build prompt
        prompt_bundle = self.prompt_builder.build(
            query=query,
            retrieved_chunks=chunks
        )

        # 3. Generate answer
        result = self.answer_generator.generate(prompt_bundle)

        # 4. Attach diagnostics
        result["diagnostics"] = retrieval.get("diagnostics", {})

        return result
