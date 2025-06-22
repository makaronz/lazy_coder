"""
LazyCoder File Processing Agent
Handles upload, analysis, and processing of various file types using AI.
"""

import os
import uuid
import asyncio
import mimetypes
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import traceback

from loguru import logger

# Optional imports - will be handled gracefully if not available
try:
    import openai
except ImportError:
    openai = None

try:
    import cv2
    import numpy as np
    from PIL import Image
    import pytesseract
except ImportError:
    cv2 = None
    np = None
    Image = None
    pytesseract = None

try:
    import whisper
except ImportError:
    try:
        import openai_whisper as whisper  # type: ignore
    except ImportError:
        whisper = None

try:
    import librosa
except ImportError:
    librosa = None

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

try:
    from docx import Document
except ImportError:
    Document = None

try:
    import spacy
except ImportError:
    spacy = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

from ...core.config import LazyCoderConfig

import httpx

# --- Monkey patch to ignore unsupported 'proxies' kwarg -----------------------
_orig_asyncclient_init = httpx.AsyncClient.__init__  # type: ignore

def _patched_asyncclient_init(self, *args, **kwargs):  # type: ignore
    # OpenAI >=1.38 przekazuje 'proxies' do httpx.AsyncClient, który już go nie wspiera
    kwargs.pop("proxies", None)
    return _orig_asyncclient_init(self, *args, **kwargs)

if not getattr(httpx.AsyncClient.__init__, "_is_patched", False):
    _patched_asyncclient_init._is_patched = True  # type: ignore
    httpx.AsyncClient.__init__ = _patched_asyncclient_init  # type: ignore
# -----------------------------------------------------------------------------

class FileType(Enum):
    """Supported file types"""
    IMAGE = "image"
    DOCUMENT = "document"
    AUDIO = "audio"
    VIDEO = "video"
    CODE = "code"
    UNKNOWN = "unknown"


@dataclass
class FileInfo:
    """Basic file information"""
    name: str
    path: str
    size: int
    type: FileType
    mime_type: str
    extension: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContentAnalysis:
    """Results of content analysis"""
    text_content: str = ""
    summary: str = ""
    entities: List[Dict[str, Any]] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    sentiment: Optional[Dict[str, float]] = None
    embeddings: Optional[np.ndarray] = None
    
    # Code-specific analysis
    functions: List[Dict[str, Any]] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    complexity: float = 0.0
    
    # Additional insights
    insights: List[str] = field(default_factory=list)
    action_items: List[str] = field(default_factory=list)


@dataclass
class MediaAnalysis:
    """Results of media-specific analysis"""
    # Image analysis
    ocr_text: str = ""
    objects_detected: List[Dict[str, Any]] = field(default_factory=list)
    ui_elements: List[Dict[str, Any]] = field(default_factory=list)
    
    # Audio analysis
    transcript: str = ""
    language: str = ""
    confidence: float = 0.0
    audio_features: Dict[str, Any] = field(default_factory=dict)
    
    # Video analysis
    frames_analyzed: int = 0
    scene_changes: List[float] = field(default_factory=list)


@dataclass
class ProcessingResult:
    """Complete file processing result"""
    file_id: str
    file_info: FileInfo
    content_analysis: ContentAnalysis
    media_analysis: Optional[MediaAnalysis] = None
    processing_time: float = 0.0
    status: str = "completed"
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)


class FileProcessingAgent:
    """
    AI-powered file processing agent that can analyze various file types
    and extract meaningful insights for development workflows.
    """
    
    def __init__(self, config: LazyCoderConfig):
        self.config = config
        self.file_config = config.get_file_processing_config()
        
        # Initialize AI clients
        self.openai_client = None
        self.whisper_model = None
        self.nlp_model = None
        self.sentence_transformer = None
        
        # Supported file extensions
        self.supported_extensions = {
            FileType.IMAGE: ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'],
            FileType.DOCUMENT: ['.pdf', '.docx', '.doc', '.txt', '.md', '.rtf'],
            FileType.AUDIO: ['.mp3', '.wav', '.flac', '.m4a', '.ogg', '.aac'],
            FileType.VIDEO: ['.mp4', '.avi', '.mov', '.mkv', '.webm'],
            FileType.CODE: ['.py', '.js', '.ts', '.html', '.css', '.json', '.xml', '.yaml', '.yml']
        }
        
        # One-time flags to avoid spammy logs
        self._whisper_warning_emitted = False
        self._spacy_warning_emitted = False
        
        logger.info("FileProcessingAgent initialized")
    
    async def initialize(self):
        """Initialize AI models and clients"""
        try:
            # Initialize OpenAI client
            if openai is not None:
                openai_config = self.config.get_ai_provider_config('openai')
                self.openai_client = openai.AsyncOpenAI(api_key=openai_config.api_key)
                logger.info("OpenAI client initialized")
            else:
                logger.warning("OpenAI not available - text analysis will be limited")
            
            # Initialize Whisper for audio transcription
            if self.file_config.use_whisper and whisper is not None and hasattr(whisper, 'load_model'):
                self.whisper_model = whisper.load_model(self.file_config.whisper_model)
                logger.info("Whisper model loaded")
            else:
                if self.file_config.use_whisper:
                    logger.warning("Whisper not available or incompatible - audio transcription disabled")
            
            # Initialize NLP model
            if spacy is not None:
                try:
                    self.nlp_model = spacy.load("en_core_web_sm")
                    logger.info("spaCy model loaded")
                except OSError:
                    logger.warning("spaCy model 'en_core_web_sm' not found - NLP analysis disabled")
            else:
                logger.warning("spaCy not available - NLP analysis disabled")
            
            # Initialize sentence transformer for embeddings
            if SentenceTransformer is not None:
                try:
                    self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
                    logger.info("Sentence transformer loaded")
                except Exception as e:
                    logger.warning(f"Failed to load sentence transformer: {e}")
            else:
                logger.warning("SentenceTransformer not available - embeddings disabled")
            
            # Create upload directory
            upload_path = Path(self.file_config.local_path)
            upload_path.mkdir(parents=True, exist_ok=True)
            
            logger.info("FileProcessingAgent initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize FileProcessingAgent: {e}\n{traceback.format_exc()}")
            raise
    
    async def process_file(self, file_path: str, context: str = "") -> ProcessingResult:
        """Main file processing workflow"""
        start_time = datetime.now()
        
        try:
            # Analyze file
            file_info = self._analyze_file(file_path)
            
            # Extract content
            content_analysis = await self._analyze_content(file_path, file_info)
            
            # Media-specific processing
            media_analysis = None
            if file_info.type in [FileType.IMAGE, FileType.AUDIO, FileType.VIDEO]:
                media_analysis = await self._analyze_media(file_path, file_info)
            
            # Generate development insights
            await self._generate_insights(content_analysis, media_analysis, context)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = ProcessingResult(
                file_id=str(uuid.uuid4()),
                file_info=file_info,
                content_analysis=content_analysis,
                media_analysis=media_analysis,
                processing_time=processing_time
            )
            
            logger.info(f"File processed successfully: {file_path}")
            return result
            
        except Exception as e:
            logger.error(f"File processing failed for {file_path}: {e}")
            return ProcessingResult(
                file_id=str(uuid.uuid4()),
                file_info=FileInfo("", file_path, 0, FileType.UNKNOWN, "", ""),
                content_analysis=ContentAnalysis(),
                status="failed",
                error=str(e)
            )
    
    def _analyze_file(self, file_path: str) -> FileInfo:
        """Analyze basic file information"""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Get file info
        stat = path.stat()
        mime_type, _ = mimetypes.guess_type(file_path)
        extension = path.suffix.lower()
        
        # Determine file type
        file_type = FileType.UNKNOWN
        for ftype, extensions in self.supported_extensions.items():
            if extension in extensions:
                file_type = ftype
                break
        
        return FileInfo(
            name=path.name,
            path=str(path.absolute()),
            size=stat.st_size,
            type=file_type,
            mime_type=mime_type or "unknown",
            extension=extension,
            metadata={
                "created": datetime.fromtimestamp(stat.st_ctime),
                "modified": datetime.fromtimestamp(stat.st_mtime)
            }
        )
    
    async def _analyze_content(self, file_path: str, file_info: FileInfo) -> ContentAnalysis:
        """Analyze file content based on type"""
        analysis = ContentAnalysis()
        
        try:
            # Extract text content
            if file_info.type == FileType.DOCUMENT:
                analysis.text_content = await self._extract_document_text(file_path, file_info.extension)
            elif file_info.type == FileType.CODE:
                analysis.text_content = await self._extract_code_content(file_path)
            elif file_info.type == FileType.IMAGE:
                analysis.text_content = await self._extract_image_text(file_path)
            
            # Perform NLP analysis if we have text
            if analysis.text_content:
                await self._perform_nlp_analysis(analysis)
            
            # Code-specific analysis
            if file_info.type == FileType.CODE:
                await self._analyze_code(analysis, file_info.extension)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Content analysis failed: {e}")
            analysis.insights.append(f"Content analysis failed: {str(e)}")
            return analysis
    
    async def _extract_document_text(self, file_path: str, extension: str) -> str:
        """Extract text from document files"""
        try:
            if extension == '.pdf':
                if PyPDF2 is None:
                    logger.warning("PyPDF2 not available - PDF extraction disabled")
                    return ""
                with open(file_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text() + "\n"
                    return text
            
            elif extension in ['.docx', '.doc']:
                if Document is None:
                    logger.warning("python-docx not available - DOCX extraction disabled")
                    return ""
                doc = Document(file_path)
                return "\n".join([paragraph.text for paragraph in doc.paragraphs])
            
            elif extension in ['.txt', '.md', '.rtf']:
                with open(file_path, 'r', encoding='utf-8') as file:
                    return file.read()
            
            return ""
            
        except Exception as e:
            logger.error(f"Document text extraction failed: {e}")
            return ""
    
    async def _extract_code_content(self, file_path: str) -> str:
        """Extract content from code files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            logger.error(f"Code content extraction failed: {e}")
            return ""
    
    async def _extract_image_text(self, file_path: str) -> str:
        """Extract text from images using OCR"""
        try:
            if cv2 is None or pytesseract is None:
                logger.warning("OpenCV or pytesseract not available - OCR disabled")
                return ""
                
            image = cv2.imread(file_path)
            if image is None:
                return ""
            
            # Use pytesseract for OCR
            text = pytesseract.image_to_string(image)
            return text.strip()
            
        except Exception as e:
            logger.error(f"Image text extraction failed: {e}")
            return ""
    
    async def _perform_nlp_analysis(self, analysis: ContentAnalysis):
        """Perform NLP analysis on text content"""
        try:
            if not analysis.text_content:
                return
            
            # Process with spaCy if available
            if self.nlp_model is not None:
                doc = self.nlp_model(analysis.text_content[:1000000])  # Limit text length
                
                # Extract entities
                analysis.entities = [
                    {
                        "text": ent.text,
                        "label": ent.label_,
                        "description": spacy.explain(ent.label_) if spacy else ent.label_
                    }
                    for ent in doc.ents
                ]
                
                # Extract keywords (noun phrases)
                analysis.keywords = [
                    chunk.text.lower()
                    for chunk in doc.noun_chunks
                    if len(chunk.text) > 2
                ][:20]  # Limit to top 20
            else:
                logger.warning("spaCy model not available - entity and keyword extraction skipped")
            
            # Generate summary using OpenAI
            if self.openai_client and len(analysis.text_content) > 100:
                analysis.summary = await self._generate_summary(analysis.text_content)
            
            # Generate embeddings
            if self.sentence_transformer:
                analysis.embeddings = self.sentence_transformer.encode(analysis.text_content[:512])
            else:
                logger.warning("Sentence transformer not available - embeddings skipped")
            
        except Exception as e:
            logger.error(f"NLP analysis failed: {e}")
    
    async def _generate_summary(self, text: str) -> str:
        """Generate summary using OpenAI"""
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that creates concise summaries of documents. Focus on key points and actionable information."
                    },
                    {
                        "role": "user", 
                        "content": f"Please provide a concise summary of the following text:\n\n{text[:4000]}"
                    }
                ],
                max_tokens=200,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return ""
    
    async def _analyze_code(self, analysis: ContentAnalysis, extension: str):
        """Analyze code-specific features"""
        try:
            code = analysis.text_content
            
            if extension == '.py':
                await self._analyze_python_code(analysis, code)
            elif extension in ['.js', '.ts']:
                await self._analyze_javascript_code(analysis, code)
            elif extension in ['.html', '.xml']:
                await self._analyze_markup_code(analysis, code)
            
        except Exception as e:
            logger.error(f"Code analysis failed: {e}")
    
    async def _analyze_python_code(self, analysis: ContentAnalysis, code: str):
        """Analyze Python code"""
        import ast
        
        try:
            tree = ast.parse(code)
            
            # Extract functions
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    analysis.functions.append({
                        "name": node.name,
                        "line": node.lineno,
                        "args": [arg.arg for arg in node.args.args],
                        "docstring": ast.get_docstring(node)
                    })
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        analysis.imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    for alias in node.names:
                        analysis.imports.append(f"{module}.{alias.name}")
            
            # Calculate complexity (simplified)
            analysis.complexity = len([n for n in ast.walk(tree) 
                                    if isinstance(n, (ast.If, ast.For, ast.While, ast.Try))])
            
        except Exception as e:
            logger.error(f"Python code analysis failed: {e}")
    
    async def _analyze_javascript_code(self, analysis: ContentAnalysis, code: str):
        """Analyze JavaScript/TypeScript code"""
        # Simple regex-based analysis for now
        import re
        
        # Extract function declarations
        func_pattern = r'function\s+(\w+)\s*\([^)]*\)'
        arrow_func_pattern = r'const\s+(\w+)\s*=\s*\([^)]*\)\s*=>'
        
        functions = re.findall(func_pattern, code) + re.findall(arrow_func_pattern, code)
        analysis.functions = [{"name": func, "type": "function"} for func in functions]
        
        # Extract imports
        import_pattern = r'import\s+.*?\s+from\s+[\'"]([^\'"]+)[\'"]'
        require_pattern = r'require\([\'"]([^\'"]+)[\'"]\)'
        
        imports = re.findall(import_pattern, code) + re.findall(require_pattern, code)
        analysis.imports = imports
    
    async def _analyze_markup_code(self, analysis: ContentAnalysis, code: str):
        """Analyze HTML/XML markup"""
        try:
            if BeautifulSoup is None:
                logger.warning("BeautifulSoup not available - markup analysis skipped")
                return
                
            soup = BeautifulSoup(code, 'html.parser')
            
            # Extract elements
            elements = {}
            for tag in soup.find_all():
                if tag.name in elements:
                    elements[tag.name] += 1
                else:
                    elements[tag.name] = 1
            
            analysis.insights.append(f"HTML elements found: {elements}")
            
        except Exception as e:
            logger.error(f"Markup analysis failed: {e}")
    
    async def _analyze_media(self, file_path: str, file_info: FileInfo) -> MediaAnalysis:
        """Analyze media files (images, audio, video)"""
        analysis = MediaAnalysis()
        
        try:
            if file_info.type == FileType.IMAGE:
                await self._analyze_image(file_path, analysis)
            elif file_info.type == FileType.AUDIO:
                await self._analyze_audio(file_path, analysis)
            elif file_info.type == FileType.VIDEO:
                await self._analyze_video(file_path, analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Media analysis failed: {e}")
            return analysis
    
    async def _analyze_image(self, file_path: str, analysis: MediaAnalysis):
        """Analyze image content"""
        try:
            # OCR text extraction
            if cv2 is not None and pytesseract is not None:
                image = cv2.imread(file_path)
                if image is not None:
                    analysis.ocr_text = pytesseract.image_to_string(image)
            else:
                logger.warning("OpenCV or pytesseract not available - OCR skipped")
            
            # Use OpenAI Vision API if available
            if self.openai_client and self.file_config.use_openai_vision:
                await self._analyze_image_with_openai(file_path, analysis)
            
        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
    
    async def _analyze_image_with_openai(self, file_path: str, analysis: MediaAnalysis):
        """Analyze image using OpenAI Vision API"""
        try:
            # Convert image to base64
            import base64
            with open(file_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Analyze this image and describe any UI elements, text, or development-related content you see. Focus on actionable insights for software development."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=300
            )
            
            description = response.choices[0].message.content
            analysis.ui_elements.append({
                "description": description,
                "source": "openai_vision"
            })
            
        except Exception as e:
            logger.error(f"OpenAI Vision analysis failed: {e}")
    
    async def _analyze_audio(self, file_path: str, analysis: MediaAnalysis):
        """Analyze audio content"""
        try:
            if self.whisper_model and whisper is not None:
                # Transcribe with Whisper
                result = self.whisper_model.transcribe(file_path)
                analysis.transcript = result['text']
                analysis.language = result.get('language', 'unknown')
                
                # Extract audio features if librosa is available
                if librosa is not None and np is not None:
                    y, sr = librosa.load(file_path)
                    analysis.audio_features = {
                        'duration': librosa.get_duration(y=y, sr=sr),
                        'tempo': float(librosa.beat.tempo(y=y, sr=sr)[0]),
                        'spectral_centroid': float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
                    }
                else:
                    logger.warning("librosa not available - audio feature extraction skipped")
            else:
                logger.warning("Whisper not available - audio transcription skipped")
            
        except Exception as e:
            logger.error(f"Audio analysis failed: {e}")
    
    async def _analyze_video(self, file_path: str, analysis: MediaAnalysis):
        """Analyze video content"""
        try:
            if cv2 is None:
                logger.warning("OpenCV not available - video analysis skipped")
                return
                
            # Basic video analysis - extract frames and analyze
            cap = cv2.VideoCapture(file_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Sample a few frames for analysis
            sample_frames = min(5, frame_count)
            for i in range(sample_frames):
                frame_pos = i * (frame_count // sample_frames)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
                ret, frame = cap.read()
                
                if ret:
                    analysis.frames_analyzed += 1
                    # Could add frame-specific analysis here
            
            cap.release()
            
        except Exception as e:
            logger.error(f"Video analysis failed: {e}")
    
    async def _generate_insights(self, content_analysis: ContentAnalysis, 
                                media_analysis: Optional[MediaAnalysis], context: str):
        """Generate development-focused insights"""
        insights = []
        
        # Content-based insights
        if content_analysis.text_content:
            if len(content_analysis.functions) > 0:
                insights.append(f"Found {len(content_analysis.functions)} functions")
            
            if len(content_analysis.imports) > 0:
                insights.append(f"Uses {len(content_analysis.imports)} imports/dependencies")
            
            if content_analysis.complexity > 10:
                insights.append(f"High complexity code (score: {content_analysis.complexity})")
        
        # Media-based insights
        if media_analysis:
            if media_analysis.ocr_text:
                insights.append(f"Extracted {len(media_analysis.ocr_text)} characters of text from image")
            
            if media_analysis.transcript:
                insights.append(f"Audio transcription available ({len(media_analysis.transcript)} characters)")
        
        # Context-based insights
        if context:
            insights.append(f"Processing in context: {context}")
        
        content_analysis.insights = insights
    
    async def execute_action(self, action: str, parameters: Dict[str, Any]) -> Any:
        """Execute file processing actions"""
        try:
            if action == "process_file":
                file_path = parameters.get("file_path")
                context = parameters.get("context", "")
                return await self.process_file(file_path, context)
            
            elif action == "analyze_content":
                file_path = parameters.get("file_path")
                file_info = self._analyze_file(file_path)
                return await self._analyze_content(file_path, file_info)
            
            else:
                raise ValueError(f"Unknown action: {action}")
                
        except Exception as e:
            logger.error(f"Action execution failed: {e}")
            raise
    
    async def health_check(self) -> bool:
        """Check agent health"""
        try:
            if self.file_config.use_whisper and not self.whisper_model:
                if not self._whisper_warning_emitted:
                    logger.warning("Whisper model not loaded - audio transcription disabled")
                    self._whisper_warning_emitted = True
            if not self.nlp_model:
                if not self._spacy_warning_emitted:
                    logger.warning("spaCy model not loaded - NLP features limited")
                    self._spacy_warning_emitted = True
            return True
        except Exception:
            return True
    
    def get_capabilities(self) -> List[str]:
        """Get list of agent capabilities"""
        return [
            "process_file",
            "analyze_content", 
            "extract_text",
            "generate_summary",
            "analyze_code",
            "transcribe_audio",
            "analyze_image"
        ]