from modules.preprocessors import BasePreprocessor
from modules.templates_notion import TEMPLATE_NOTION_DEFAULT

import re
import glob
from typing import List, Tuple, Dict, Union, Callable, Optional
from langchain.schema import (
    BasePromptTemplate,
    Document,
    BaseDocumentTransformer,
)
from langchain.document_loaders import NotionDirectoryLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter


class NotionPreprocessor(BasePreprocessor):
    @property
    def splitter(self):
        if self._splitter is None:
            return MarkdownHeaderTextSplitter(
                headers_to_split_on=[
                    ("#", "#"),
                    ("##", "##"),
                ],  # (current_header, reformat_header)
            )
        else:
            return self._splitter

    def _get_file_path(self, filename: str, directory: str = None):
        directory = "data/notion/DV-PromptTown"  # 💥 TODO: guess from the full path
        files = glob.glob(f"{directory}/**/*{filename}", recursive=True)
        return files

    def _extract_doc_id(self, doc: Document):
        """Notion이 자동으로 붙인 파일/폴더 id (소문자 + 숫자 + (.확장자)) 추출."""
        return doc.metadata["source"].split(" ")[-1]

    def _handle_links(
        self,
        doc: Document,
        file_formats: Tuple[str] = ("md", "csv"),
    ) -> Document:
        """마크다운 문서에서 포함된 하이퍼링크 스트링이 임베딩 되지 않게 "(link@{num})"으로 변환하고 메타데이터 리스트(인덱스{num})에 저장 (key는 'links')"""
        page_content = doc.page_content
        num = 0
        while match := re.search(
            r"(?<=\])\(%[A-Za-z0-9\/\(\)%\.]+",
            page_content,
        ):
            (match_start_idx, non_match_start_idx) = match.span()

            if match.group().endswith(file_formats):
                # 링크 스트링 메타 데이터에 추가
                doc.metadata["links"] = doc.metadata.get("links", []).append(
                    match.group().strip("()")
                )

                # 링크 스트링 삭제
                page_content = (
                    page_content[: match_start_idx - 1]
                    + f"(link@{num})"
                    + page_content[non_match_start_idx:]
                )

                num += 1
            else:
                ## .png 등은 그냥 링크 삭제
                page_content = (
                    page_content[: match_start_idx - 1]
                    + page_content[non_match_start_idx:]
                )

        doc.page_content = page_content

        return doc

    def preprocess(
        self,
        docs: List[Document],
    ) -> List[Document]:
        """
        if the splitter is `MarkdownTextSplitter` add the headers

        """
        ## page_content내 링크를 meta 데이터로 추가
        docs = [self._handle_links(doc) for doc in docs]
        return docs

    def postprocess(
        self,
        chunks: List[Document],
    ) -> List[Document]:
        """
        `TextSplitter`가 자른 문서가 1. 길이가 너무 짧고 2. 같은 부모 디렉토리를 갖을 때 합치기 (+ 메타데이터 소스 수정)
        """
        prev_chunk = None
        new_chunks = []
        for chunk in chunks:
            if prev_chunk is None:
                # 맨 처음 chunk는 바로 prev_chjunk
                prev_chunk = chunk
                continue

            if (
                len(prev_chunk.page_content) + len(chunk.page_content) < 500
                and prev_chunk.metadata["source"].split("/")[:-1]
                == chunk.metadata["source"].split("/")[:-1]
            ):
                chunk.page_content = prev_chunk.page_content + "\n" + chunk.page_content
                ## 동일 부모 폴더의 파일들을 합치는 경우 "부모폴더/파일1&&파일2" 로 메타 데이터 저장
                chunk.metadata["source"] = (
                    prev_chunk.metadata["source"]
                    + "&&"
                    + chunk.metadata["source"].split("/")[-1]
                )
            else:
                new_chunks.append(prev_chunk)

            prev_chunk = chunk

        if prev_chunk != new_chunks[-1]:
            new_chunks.append(prev_chunk)

        return new_chunks

    def _split(
        self,
        docs: List[Document],
    ) -> List[Document]:
        """`.split_text(doc.page_content)` 한 결과물에 메타데이터로 헤더값 본문에 추가"""

        new_chunks = []
        for doc in docs:
            src = doc.metadata["source"]
            src_id = self._extract_doc_id(doc)
            page_content = doc.page_content

            ## 정한 헤더 레벨에 따라 chunking
            chunks = self.splitter.split_text(page_content)

            headers = self.splitter.headers_to_split_on
            for _, header in headers:
                if header in doc.metadata:
                    doc.page_content = (
                        header
                        + " "
                        + doc.metadata.pop(header)
                        + "\n"
                        + doc.page_content
                    )
            new_chunks.append(doc)

    def preprocess_and_split(
        self,
        docs: List[Document],
    ) -> List[Document]:
        docs = self.preprocess(docs)
        chunks = self._split(docs)
        chunks = self.postprocess(chunks)

        return chunks


loader = NotionDirectoryLoader("data/notion/[DV-프롬프트타운]")
docs = loader.load()
processor = NotionPreprocessor(prompt=TEMPLATE_NOTION_DEFAULT)
docs = processor.preprocess_and_split(docs)
