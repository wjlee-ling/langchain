from modules.preprocessors.base import BasePreprocessor

from typing import List, Dict, Callable, Optional
from langchain.schema import Document
from langchain.text_splitter import MarkdownTextSplitter


class PDFPreprocessor(BasePreprocessor):
    @property
    def splitter(self):
        if self._splitter is None:
            return MarkdownTextSplitter()
        else:
            return self._splitter

    def get_default_fn(
        self,
        doc: Document,
        previous_doc_content: List = [None],
    ) -> Dict:
        """
        디폴트 전처리 함수. 현재 doc뿐만 아니라 이전 doc의 content의 뒷부분도 같이 GPT에 전달하여, 필요시 활용하게끔 함 (e.g. 페이지 넘어감).
        """
        page_content = doc.page_content

        if previous_doc_content[0] is None:
            context = ""
        else:
            context = "\n".join(previous_doc_content[0].strip().split("\n")[-2:])

        previous_doc_content[0] = page_content

        return {
            "context": context,
            "doc": page_content,
        }

    def preprocess(
        self,
        docs: List[Document],
        fn: Optional[Callable[Document, Dict]] = None,
    ) -> List[Document]:
        """
        batch로 한번에 요청하여 빠르게 전처리. 다만 API 요청 제한이 있으므로 주의.
        Args
            - fn: `Document`별 전처리 함수. 사용하는 템플렛에 맞게 전처리 함수 구현해야 함.
        """

        fn = fn or self.get_default_fn
        new_page_contents = self.chain.batch(
            list(
                map(
                    lambda doc: fn(doc) if fn else doc,
                    docs,
                )
            )
        )  # returns: e.g. [{'doc': '안   녕 하세!요', 'output': '안녕하세요!'}, {'doc': '바보\n야 꺼~~~져!', 'output': '"바보야, 꺼져!"'}]
        # .apply 보다 2배 빠름

        for i, doc in enumerate(docs):
            self.save_output(
                {
                    "original_page_content": doc.page_content,
                    "revision": new_page_contents[i]["output"],
                    "metadata": doc.metadata,
                },
            )
            doc.metadata["original_page_content"] = doc.page_content
            doc.page_content = new_page_contents[i]["output"]

        return docs
