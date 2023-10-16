"""
chain 타입에 따라 template의 포맷과 variable이 달라짐.
`ConversationalRetrievalChain`의 경우, quenstion-condensing을 위한 `CONDENSE_QUESTION_PROMPT` template과 answer-generation을 위한 `QA_PROMPT` template이 필요함. (https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/chains/conversational_retrieval/prompts.py#L10)
`ConversationalRetrievalChain.from_llm()`의 경우 
1. `CONDENSE_QUESTION_PROMPT` template은 인자 `condense_question_prompt`로 커스텀할 수 있으며, default template은 `langchain.chains.conversational_retrieval.prompts.py`에 지정되어 있음.
2. `QA_PROMPT`는 `from_llm()` 내부 `load_qa_chain()`에서 `chain_type`("stuff" by default)을 받고 그 종류에 따라 `langchain.chains.question_answering`에 지정된 default prompt를 사용함.
    커스텀하려면 `combine_docs_chain_kwargs`에 해당 prompt_type 명시해야함 (참고: https://github.com/langchain-ai/langchain/blob/c2d1d903fa35b91018b4d777db2b008fcbaa9fbc/langchain/chains/question_answering/__init__.py#L134)
"""
from langchain.prompts import PromptTemplate


rules = """[규칙] \
1. [최종 답변]은 온전한 문장으로 작성하라. \
2. [최종 답변]은 한국어로 작성하라. \
3. [최종 답변]은 반말은 사용하지 말고 존댓말로 격식있게 하라. \
4. [최종 답변]을 근거없이 지어내지 마라. \
5. [최종 답변]에 욕, 비속어, 인종차별, 성차별 기타 소수자 혐오 발언은 하지 마라.\
"""

template_en = """Observe the following rules to answer the question at the end.\
    1. Answer the question in a complete sentence.\
    2. Answer in Korean.\
    3. Answer in a polite manner with honorifics. \
    4. If you don't know the answer, just type "잘 모르겠습니다".\
    5. DO NOT swear or use offensive language.\
    Given the rules, the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
    chat history: {chat_history}\
    question: {question}\
    answer:"""

template_kor = """{rules} \
[대화내역] {chat_history} \
[질문] {question} \
위 [규칙]과 [대화내역]을 참고하여 [질문]에 대한 [답변]을 작성하라. \
[답변] """

CONDENSE_QUESTION_TEMPLATE = PromptTemplate.from_template(
    " ".join(
        (
            "[대화 내역]을 기반으로 [사용자 질문]을 고치거나 보강하여 한국어로 [수정된 사용자 질문]을 작성하라",
            "[대화 내역] {chat_history}",
            "[사용자 질문] {question}",
            "[수정된 사용자 질문] ",
        )
    )
)

STUFF_QA_TEMPLATE = PromptTemplate.from_template(
    " ".join(
        (
            "[질문]과 [context]를 바탕으로 [최종 답변]을 작성하라. [최종 답변] 작성시 아래 [규칙]을 참고하라.",
            rules,
            "[최종 답변] ",
        )
    )
)
