from modules.preprocessors import BasePreprocessor
from modules.templates import CONDENSE_QUESTION_TEMPLATE
from utils import create_collection, create_save_collection

import os
import langchain
from typing import Optional, Any, Dict, Union
from langchain.schema import BaseDocumentTransformer
from langchain.schema.prompt_template import BasePromptTemplate
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.vectorstore import VectorStore
from langchain.document_loaders.base import BaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.cache import InMemoryCache
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from pydantic import BaseModel
from datetime import datetime
from wandb.sdk.data_types.trace_tree import Trace


class BaseBot:
    langchain.llm_cache = InMemoryCache()
    os.environ["LANGCHAIN_WANDB_TRACING"] = "true"
    os.environ["WANDB_PROJECT"] = "langchain"

    def __init__(
        self,
        # prompts: Optional[CustomPrompts] = None,
        llm: Optional[BaseLanguageModel] = None,
        condense_question_llm: Optional[BaseLanguageModel] = None,
        condense_question_prompt: Optional[BasePromptTemplate] = None,
        vectorstore: Optional[VectorStore] = None,
        docs_chain_type: str = "stuff",
        docs_chain_kwargs: Optional[Dict] = None,
        wandb_tracer: Optional[Trace] = None,
        configs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Args:
            - prompts: dict of prompts to use for each chain type. If not given, default prompts will be used. Different sets of prompts are required for different chain types.
            For example, `stuff` chain_type requires `qa_prompt` and `condense_question_prompt` prompts, while `map_reduce` chain_type requires `condense_question_prompt`, `question_prompt` and `combine_prompt` prompts.
        """
        # prompts
        # if prompts is not None:
        #     _, self.docs_chain_kwargs = self._validate_docs_chain_and_prompts(
        #         prompts, docs_chain_type, docs_chain_kwargs
        #     )
        # else:
        #     self.condense_question_prompt = CONDENSE_QUESTION_TEMPLATE
        self.condense_question_prompt = (
            condense_question_prompt or CONDENSE_QUESTION_TEMPLATE
        )

        # llm for doc-chain
        self.llm = (
            ChatOpenAI(
                model_name="gpt-3.5-turbo-0613",  # "gpt-4"
                temperature=0,
                verbose=True,
            )
            if llm is None
            else llm
        )
        self.vectorstore = (
            Chroma(
                collection_name="default",
            )
            if vectorstore is None
            else vectorstore
        )
        self.retriever = self.vectorstore.as_retriever()
        self.condense_question_llm = (
            ChatOpenAI(
                model_name="gpt-3.5-turbo-0613",
                temperature=0,
            )
            if condense_question_llm is None
            else condense_question_llm
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            output_key="answer",  # ☑️ required if return_source_documents=True
            return_messages=True,  # ☑️ required if return_source_documents=True
        )

        self.wandb_tracer = wandb_tracer or Trace(
            name="ConverationalRetrievalChain",
            kind="chain",
            start_time_ms=round(datetime.now().timestamp() * 1000),
            metadata={
                "llm": self.llm.__dict__["model_name"],
                "temperature": self.llm.__dict__["temperature"],
                "condense_question_llm": self.condense_question_llm.__dict__[
                    "model_name"
                ],
                "condense_question_temperature": self.condense_question_llm.__dict__[
                    "temperature"
                ],
            },
        )

        # build a chain with the given components
        self.chain = ConversationalRetrievalChain.from_llm(
            # https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/chains/conversational_retrieval/base.py#L268
            # chain_type:
            # "stuff": default; to use all of the text from the documents in the prompt
            # "map_reduce": to batchify docs and feeds each batch with the question to LLM, and come up with the final answer based on the answers
            # "refine": to batchify docs and feeds the first batch to LLM, and then feeds the second batch with the answer from the first one, and so on
            # "map-rerank": to batchify docs and feeds each batch, return a score and come up with the final answer based on the scores
            llm=self.llm,
            retriever=self.retriever,
            memory=self.memory,
            chain_type=docs_chain_type,
            condense_question_llm=self.condense_question_llm,
            condense_question_prompt=self.condense_question_prompt,
            combine_docs_chain_kwargs=docs_chain_kwargs,
            rephrase_question=False,  # default: True; Will pass the new generated question for retrieval
            return_source_documents=True,
            get_chat_history=None,  # default: None -> will use default;
            response_if_no_docs_found="잘 모르겠습니다.",
            verbose=True,
        )

    def __call__(self, question: str):
        response = self.chain(question)
        self.wandb_tracer.add_inputs_and_outputs(
            inputs={"question": question}, outputs={"response": response}
        )
        return response

    # def _validate_docs_chain_and_prompts(
    #     self, prompts, docs_chain_type: str, docs_chain_kwargs: Dict
    # ):
    #     assert docs_chain_type in [
    #         "stuff",
    #         "map_reduce",
    #         "refine",
    #         "map-rerank",
    #     ], f"docs_chain_type must be one of ['stuff', 'map_reduce', 'refine', 'map-rerank'], but got {docs_chain_type}"

    #     if docs_chain_type == "stuff":
    #         assert (
    #             prompts.combine_prompt is None
    #             and prompts.collapse_prompt is None
    #             and prompts.refine_prompt is None
    #         )
    #         prompts["prompt"] = prompts.pop("qa_prompt")
    #     elif docs_chain_type == "map-rerank":
    #         assert (
    #             prompts.combine_prompt is None
    #             and prompts.collapse_prompt is None
    #             and prompts.refine_prompt is None
    #         )
    #         prompts["prompt"] = prompts.pop("qa_prompt")
    #     elif docs_chain_type == "refine":
    #         assert (
    #             prompts.refine_prompt
    #             and prompts.collapse_prompt is None
    #             and prompts.combine_prompt is None
    #         )
    #         prompts["question_prompt"] = prompts.pop("qa_prompt")
    #     else:
    #         assert (
    #             prompts.refine_prompt is None
    #             and prompts.collapse_prompt
    #             and prompts.combine_prompt
    #         )
    #         prompts["question_prompt"] = prompts.pop("qa_prompt")

    #     self.condense_question_prompt = prompts.pop("condense_question_prompt")
    #     docs_chain_kwargs.update(prompts)

    #     return prompts, docs_chain_kwargs

    @staticmethod
    def __configure__(configs: Dict[str, Any]):
        """
        각 컴포넌트에 kwargs로 들어가는 인자들의 값을 설정합니다. 사용자가 설정하지 않은 값들의 기본값을 설정합니다.

        TO-DO:
        - choose size appropriate to llm context size
        """
        default_configs = {}
        default_splitter_configs = {
            "chunk_size": 1000,
            "chunk_overlap": 150,
        }
        splitter_configs = (
            configs.get(
                "splitter", default_splitter_configs
            )  # default: 4000 / 200 # TO-DO
            if configs
            else default_splitter_configs
        )
        default_configs["splitter"] = splitter_configs
        return default_configs

    @classmethod
    def from_new_collection(
        cls,
        loader: BaseLoader,
        splitter: Optional[BaseDocumentTransformer] = None,
        preprocessor: Optional[BasePreprocessor] = None,
        collection_name: str = "default",
        llm: Optional[BaseLanguageModel] = None,
        condense_question_llm: Optional[BaseLanguageModel] = None,
        condense_question_prompt: Optional[BasePromptTemplate] = None,
        # prompts: Optional[CustomPrompts] = None,
        docs_chain_type: str = "stuff",
        docs_chain_kwargs: Optional[Dict] = None,
        configs: Optional[Dict[str, Dict[str, str]]] = None,
    ):
        """Build new collection AND chain based on it"""
        configs = cls.__configure__(configs)
        data = loader.load()

        if preprocessor is None:
            splitter = splitter or RecursiveCharacterTextSplitter(
                **configs["splitter"],
            )
            print(
                "💥The default text-splitter `RecursiveCharacterTextSplitter` will be used."
            )
            docs = splitter.split_documents(data)
        else:
            if splitter:
                print(
                    "💥The given text-splitter will be overriden by that of the given preprocessor."
                )
            docs = preprocessor.preprocess_and_split(
                docs=data,
                fn=configs.get("preprocessing_fn", None),
            )

        vectorstore = create_save_collection(
            collection_name=collection_name,
            docs=docs,
        )
        return cls(
            # prompts=prompts,
            llm=llm,
            vectorstore=vectorstore,
            condense_question_llm=condense_question_llm,
            condense_question_prompt=condense_question_prompt,
            docs_chain_type=docs_chain_type,
            docs_chain_kwargs=docs_chain_kwargs,
            configs=configs,
        )
