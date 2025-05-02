import logging
import re
from typing import List, Dict, Tuple, Optional, Any

from dataclasses import dataclass, field
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseRetriever, Document
from langchain.callbacks.manager import CallbackManagerForRetrieverRun

from core.utils import (
    num_tokens_from_string,
    smart_chunk_selection,
)
from core.ocr import is_gibberish, StatisticalTextCorrector

# Configure logging
logger = logging.getLogger(__name__)

# ================================
# Configuration
# ================================

@dataclass
class RAGConfig:
    embedding_model: str = "all-MiniLM-L6-v2"
    llm_model: str = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
    temperature: float = 0.5  # Reduced for more deterministic outputs
    max_tokens_response: int = 2048  # Increased for more comprehensive answers
    max_tokens_context: int = 8192  # Increased to leverage Llama-4's context window
    retrieval_k: int = 8  # Increased for more comprehensive knowledge retrieval
    page_filtering: bool = True
    handle_low_quality: bool = True  # Enable special handling for low-quality text
    use_structure: bool = True  # Enable structure-aware retrieval

# ================================
# Token Management
# ================================

class TokenManager:
    def __init__(self, config: RAGConfig):
        self.config = config

    def get_available_tokens(self, query: str) -> int:
        query_tokens = num_tokens_from_string(query)
        reserved = 500
        return max(500, self.config.max_tokens_context - query_tokens - reserved)

    def optimize_context(self, contexts: List[str], query: str) -> List[str]:
        available = self.get_available_tokens(query)
        context_tokens = sum(num_tokens_from_string(c) for c in contexts)
        if context_tokens <= available:
            return contexts

        logger.info(f"Optimizing {context_tokens} tokens to fit {available} tokens.")
        relevance_scores = [1.0 - (i / len(contexts)) for i in range(len(contexts))]
        return smart_chunk_selection(contexts, query, relevance_scores, available)

# ================================
# Quality Assessment
# ================================

class TextQualityAssessor:
    """Assess and improve text quality for retrieval."""
    
    def __init__(self):
        self.text_corrector = StatisticalTextCorrector()
        self.quality_threshold = 0.4
        
    def assess_document_quality(self, doc: Document) -> Tuple[float, Document]:
        """
        Assess document quality and improve if needed.
        Returns quality score and potentially improved document.
        """
        text = doc.page_content
        quality_score = self._calculate_quality_score(text)
        
        # If document has metadata indicating it was already corrected, trust it
        if doc.metadata.get("corrected", False):
            return quality_score, doc
            
        # Apply correction if quality is low
        if quality_score < self.quality_threshold:
            corrected_text = self.text_corrector.correct_text(text)
            
            # Create new document with corrected text
            improved_doc = Document(
                page_content=corrected_text,
                metadata={**doc.metadata, "corrected": True, "original_quality": quality_score}
            )
            return quality_score, improved_doc
            
        return quality_score, doc
        
    def _calculate_quality_score(self, text: str) -> float:
        """Calculate a quality score between 0-1 for the text."""
        if not text or len(text) < 20:
            return 0.0
            
        # Check for gibberish and unusual patterns
        if is_gibberish(text):
            return 0.2
            
        # Count unusual character sequences
        unusual_chars = sum(1 for c in text if ord(c) > 127 or c in "❑✬✚✲✭✮✯")
        unusual_ratio = unusual_chars / len(text)
        
        # Check for well-formed sentences with proper punctuation
        sentences = re.findall(r'[A-Z][^.!?]*[.!?]', text)
        sentence_ratio = min(1.0, len(sentences) / max(1, text.count('.') + text.count('!') + text.count('?')))
        
        # Word to character ratio (to detect meaningless character strings)
        words = re.findall(r'\b\w+\b', text)
        word_char_ratio = min(1.0, len(words) * 5 / len(text))
        
        # Calculate final score
        score = (1.0 - unusual_ratio) * 0.3 + sentence_ratio * 0.4 + word_char_ratio * 0.3
        return max(0.0, min(1.0, score))

# ================================
# RAG Engine
# ================================

class RAGEngine:
    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        self.token_manager = TokenManager(self.config)
        self.quality_assessor = TextQualityAssessor() if self.config.handle_low_quality else None
        self.embedding_model = None
        self.vectorstore = None
        self.memory = None
        self.chain = None

    def initialize(self):
        try:
            self.embedding_model = HuggingFaceEmbeddings(model_name=self.config.embedding_model)
            logger.info(f"Loaded embedding model: {self.config.embedding_model}")
        except Exception as e:
            logger.error(f"Embedding model load failed: {e}")
            raise

        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
        )
        return self

    def create_knowledge_base(self, chunks: List[Dict]):
        if not chunks:
            raise ValueError("No chunks provided.")
        
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [self._prepare_metadata(chunk) for chunk in chunks]

        try:
            self.vectorstore = FAISS.from_texts(
                texts=texts,
                embedding=self.embedding_model,
                metadatas=metadatas,
            )
            logger.info(f"Knowledge base created with {len(chunks)} chunks.")
        except Exception as e:
            logger.error(f"Knowledge base creation failed: {e}")
            raise

    def _prepare_metadata(self, chunk: Dict) -> Dict:
        metadata = {
            "tokens": chunk.get("tokens", num_tokens_from_string(chunk["text"])),
            "section": chunk.get("section", "default"),
            "page": chunk.get("metadata", [{}])[0].get("page", 0) if chunk.get("metadata") else 0,
            "chunk_index": chunk.get("chunk_index", 0),
            "confidence": chunk.get("confidence", 1.0),
            "corrected": chunk.get("corrected", False)
        }
        return metadata

    def setup_retrieval_chain(self, groq_api_key: str):
        if not self.vectorstore:
            raise ValueError("Knowledge base missing.")
        
        # Enhance prompt to handle low quality text
        template = """
        You are a helpful assistant. 
        
        Important: Some of the document context may contain OCR errors or text recognition issues. 
        Provide the Document Overview only if the user asks for it.
        Do your best to understand the meaning despite these errors. 
        If you're uncertain about particular text, indicate this in your response.
        When referring to specific parts of the document, mention page numbers if available.
        Format your response in a clear, structured way that's easy to read.
        Maintain a professional, helpful tone throughout
        Document Context:
        {context}
        
        Chat History:
        {chat_history}
        
        Human: {question}
        
        Assistant:
        """

        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "chat_history", "question"]
        )

        llm = ChatGroq(
            model_name=self.config.llm_model,
            temperature=self.config.temperature,
            groq_api_key=groq_api_key,
            max_tokens=self.config.max_tokens_response,
        )

        retriever = self._create_retriever()
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": prompt},
            return_source_documents=True,
            output_key="answer",
            verbose=True,
        )
        logger.info("Retrieval chain setup complete.")

    def _create_retriever(self):
        base_retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.config.retrieval_k}
        )

        class EnhancedRetriever(BaseRetriever):
            base_retriever: BaseRetriever = field()
            token_manager: TokenManager = field()
            config: RAGConfig = field()
            quality_assessor: Optional[TextQualityAssessor] = field(default=None)

            class Config:
                arbitrary_types_allowed = True

            def _get_relevant_documents(
                self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
            ) -> List[Document]:
                docs = self.base_retriever.get_relevant_documents(query)
                page_number = self._extract_page_number(query)

                if page_number and self.config.page_filtering:
                    docs = [doc for doc in docs if doc.metadata.get("page") == page_number]
                    logger.info(f"Filtered docs for page {page_number}: {len(docs)} found.")
                
                # Process retrieved documents for quality if enabled
                if self.quality_assessor:
                    improved_docs = []
                    for doc in docs:
                        _, improved_doc = self.quality_assessor.assess_document_quality(doc)
                        improved_docs.append(improved_doc)
                    docs = improved_docs

                texts = [doc.page_content for doc in docs]
                optimized_texts = self.token_manager.optimize_context(texts, query)

                selected_docs = [
                    doc for doc in docs if doc.page_content in optimized_texts
                ]
                return selected_docs

            def _extract_page_number(self, query: str) -> Optional[int]:
                match = re.search(r"page\s+(\d+)", query.lower())
                return int(match.group(1)) if match else None

        return EnhancedRetriever(
            base_retriever=base_retriever,
            token_manager=self.token_manager,
            config=self.config,
            quality_assessor=self.quality_assessor,
        )

    def query(self, query: str) -> Tuple[str, List[Dict]]:
        if not self.chain:
            raise ValueError("Retrieval chain not initialized.")

        try:
            result = self.chain.invoke({"question": query})
            answer = result.get("answer", "")
            sources = [
                {"content": doc.page_content, "metadata": doc.metadata}
                for doc in result.get("source_documents", [])
            ]
            
            # Add a note about potential OCR issues if low confidence sources were used
            low_confidence_sources = [s for s in sources if s["metadata"].get("confidence", 1.0) < 0.7]
            if low_confidence_sources and len(low_confidence_sources) > len(sources) / 2:
                note = "\n\nNote: Some source material may contain OCR or text recognition errors."
                if not note in answer:
                    answer += note
                    
            return answer, sources

        except Exception as e:
            logger.error(f"Query error: {e}")
            return f"I encountered an error: {e}", []
            
    def query_direct(self, query: str) -> str:
        """
        Query the LLM directly without document context.
        Used when no document has been uploaded.
        """
        try:
            # Setup a minimal context template for direct queries
            template = """
            You are a helpful assistant. Answer the user's question to the best of your ability.
            
            Chat History:
            {chat_history}
            
            Human: {question}
            
            Assistant:
            """
            
            prompt = PromptTemplate(
                template=template,
                input_variables=["chat_history", "question"]
            )
            
            # Get the groq_api_key that was passed to setup_retrieval_chain
            groq_api_key = None
            
            # Try to get the API key from existing chain
            if hasattr(self, 'chain') and self.chain:
                try:
                    # Access the LLM directly from the chain if possible
                    llm = self.chain.llm
                    return self._run_direct_query(query, llm, prompt)
                except Exception as e:
                    logger.warning(f"Couldn't access LLM from chain: {e}")
                    # Continue to create new LLM
            
            # If we reach here, we need to create a new LLM
            if not groq_api_key:
                # The API key must be provided when calling the method
                raise ValueError("Missing Groq API key for direct query. Run setup_retrieval_chain first.")
                
            # Create a new LLM
            llm = ChatGroq(
                model_name=self.config.llm_model,
                temperature=self.config.temperature,
                groq_api_key=groq_api_key,
                max_tokens=self.config.max_tokens_response,
            )
            
            return self._run_direct_query(query, llm, prompt)
            
        except Exception as e:
            logger.error(f"Direct query error: {e}")
            return f"I encountered an error: {e}"
            
    def _run_direct_query(self, query: str, llm, prompt):
        """Helper method to run a direct query with a given LLM and prompt."""
        # Create a simple chain for direct queries
        from langchain.chains import LLMChain
        
        direct_chain = LLMChain(
            llm=llm,
            prompt=prompt,
            memory=self.memory,
            verbose=True,
        )
        
        # Run the query
        result = direct_chain.invoke({"question": query})
        answer = result.get("text", "")
        
        return answer
