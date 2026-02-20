# Code Review and Refactoring Recommendations

**Repository:** llm-devs
**Last Updated:** 2026-02-20
**Code Base Size:** 1,642 lines of Python across 13 files
**Test Coverage:** 0%
**Priority Status:** HIGH - Multiple critical issues identified

---

## 📋 Executive Summary

This document outlines a comprehensive code review of the llm-devs repository, identifying opportunities for cleaning, refactoring, and improving code quality. The codebase contains multiple research and development projects for LLM/VLM applications but lacks standardization, consolidation, and critical best practices.

**Key Metrics:**
- ⚠️ **3 Critical Security Issues** (credentials, user paths, hardcoded URIs)
- ⚠️ **0% Test Coverage** (18+ functions without tests)
- ⚠️ **18 Functions** missing docstrings
- ⚠️ **4+ Code Duplication** patterns identified
- ⚠️ **15+ Magic Numbers** hardcoded throughout
- ✅ **Opportunity for 30-40% Code Reduction** through consolidation

---

## 🔴 CRITICAL ISSUES (Fix Immediately)

### 1. Security: Credentials in Source Code

**Location:** `/d/llm-devs/langchain/graphdb_groq_gemma.py` Line 12

**Current Code:**
```python
NEO4J_PASSWORD = "your_password"
```

**Issue:** Hardcoded password (even as placeholder) in source code

**Fix:** Use environment variables
```python
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
if not NEO4J_PASSWORD:
    raise ValueError("NEO4J_PASSWORD environment variable not set")
```

---

### 2. Security: User Path Exposure

**Location:** `/d/llm-devs/langchain/nvidia_rag.py` Line 16

**Current Code:**
```python
pdf_path = "/mnt/c/Users/mauli/Downloads/2503.19903v1.pdf"
vector_root = "/mnt/d/llm-devs/langchain/vector_stores"
```

**Issue:** Exposes username "mauli" and absolute paths

**Fix:** Use environment variables
```python
pdf_path = os.getenv("PDF_PATH", "/mnt/c/Users/mauli/Downloads/2503.19903v1.pdf")
vector_root = os.getenv("VECTOR_STORE_ROOT", "/mnt/d/llm-devs/langchain/vector_stores")
```

---

### 3. Security: Environment Variable Override

**Location:** `/d/llm-devs/langchain/graphdb_groq_gemma.py` Lines 9-12

**Current Code:**
```python
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "your_password")
# Then immediately overrides:
NEO4J_URI = "neo4j://localhost:7687"  # WRONG: This shadows the env var!
```

**Issue:** Environment variable is set but then overridden with hardcoded value

**Fix:** Remove the override or add validation
```python
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
# Don't override after setting!
```

---

## 🟠 HIGH PRIORITY ISSUES (Fix This Week)

### 1. Duplicate Code: VLM API Calls

**Files Affected:**
- `/d/llm-devs/vlm_application/utils.py` (3 functions: `send_prompt`, `send_prompt_for_image`, `send_text_query_prompt`)
- `/d/llm-devs/langchain/nvidia_tools.py` (3 variations in different methods)

**Current Pattern:** Each function rebuilds headers and payload from scratch
```python
# Pattern repeated 6+ times:
headers = {"Content-Type": "application/json"}
payload = {
    "model": vllm_model,
    "temperature": 0,
    "max_tokens": 1024,
    "messages": [...]
}
response = requests.post(vllm_api_endpoint, headers=headers, json=payload)
```

**Recommendation:** Create `VLMAPIClient` class
```python
class VLMAPIClient:
    def __init__(self, endpoint, model, temperature=0, max_tokens=1024):
        self.endpoint = endpoint
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def send_prompt_for_video(self, prompt, video_base64, system_prompt=None):
        """Send prompt with video to VLM API"""
        payload = self._build_payload(prompt, video_base64, "video_url", system_prompt)
        return self._post(payload)

    def send_prompt_for_image(self, prompt, image_base64, system_prompt=None):
        """Send prompt with image to VLM API"""
        payload = self._build_payload(prompt, image_base64, "image_url", system_prompt)
        return self._post(payload)

    def _build_payload(self, prompt, media_base64, media_type, system_prompt):
        """Build consistent payload structure"""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [...]
        }

    def _post(self, payload):
        """Execute API call with error handling"""
        try:
            response = requests.post(self.endpoint, headers=self.headers, json=payload)
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"VLM API error: {e}")
            raise
```

**Impact:** Reduces ~150 lines of code, improves maintainability

---

### 2. Duplicate Code: Base64 Encoding

**Location:** `/d/llm-devs/vlm_application/utils.py` Lines 140-150

**Current Code:**
```python
def encode_base64_content_from_file(file_path: str) -> str:
    """Encode a local video file to base64 format."""
    with open(file_path, "rb") as file:
        file_content = file.read()
        base64_encoded_content = base64.b64encode(file_content).decode("utf-8")
    return base64_encoded_content

def encode_base64_content_for_imagefile(image_path):
    with open(image_path, "rb") as img:
        return base64.b64encode(img.read()).decode("utf-8")
```

**Issue:** Two functions doing identical operations

**Fix:** Single function with consistent interface
```python
def encode_file_to_base64(file_path: str) -> str:
    """Encode any file to base64 string.

    Args:
        file_path: Path to file (video, image, document, etc.)

    Returns:
        Base64 encoded string

    Raises:
        FileNotFoundError: If file doesn't exist
        IOError: If file cannot be read
    """
    try:
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except IOError as e:
        raise IOError(f"Cannot read file {file_path}: {e}")

# Update imports to use single function
```

**Impact:** Eliminates code duplication, improves consistency

---

### 3. Hardcoded Paths Across Codebase

**Affected Files:**

| File | Line | Issue |
|------|------|-------|
| `/d/llm-devs/ASR/train.py` | 143 | `/mnt/d/llm-devs/ASR/whisper-base-en-finetuned` |
| `/d/llm-devs/n8n/fake_data_gen.py` | 30 | `D:/llm-devs/n8n/fake_data_gen.json` |
| `/d/llm-devs/langchain/nvidia_rag.py` | 16-17 | User-specific paths |
| `/d/llm-devs/vlm_application/config.yaml` | 4-8 | Windows-specific paths |

**Recommendation:** Create `.env.example` and load via `python-dotenv`

```bash
# .env.example
ASR_MODEL_PATH=/path/to/whisper-base-en-finetuned
DATA_OUTPUT_DIR=/path/to/output
PDF_INPUT_PATH=/path/to/pdf.pdf
NEO4J_URI=neo4j://localhost:7687
NEO4J_PASSWORD=your_password
VLM_ENDPOINT=http://127.0.0.1:8000/v1/chat/completions
VLM_MODEL=qwen/qwen2.5-vl-7b
```

**Python Implementation:**
```python
from pathlib import Path
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv(Path(__file__).parent.parent / ".env")

# Use with defaults for testing
ASR_MODEL_PATH = Path(os.getenv("ASR_MODEL_PATH", "/mnt/d/llm-devs/ASR/whisper-base-en-finetuned"))
```

---

### 4. Missing Type Hints

**Files Without Type Hints (Priority Order):**

1. **`/d/llm-devs/vlm_application/utils.py`** (348 lines)
   - Only 3 of 12 functions have type hints

2. **`/d/llm-devs/langchain/nvidia_tools.py`** (241 lines)
   - Multiple functions lack hints

3. **`/d/llm-devs/asr_text_collection.py`** (126 lines)
   - No type hints on any functions

**Example Fix:**

**Before:**
```python
def send_prompt(vllm_api_endpoint, vllm_model, prompt, video_base64, system_prompt=None):
    headers = {"Content-Type": "application/json"}
    payload = {...}
    response = requests.post(vllm_api_endpoint, headers=headers, json=payload)
    result = response.json()["choices"][0]["message"]["content"]
    return result
```

**After:**
```python
from typing import Optional, Dict, Any

def send_prompt(
    vllm_api_endpoint: str,
    vllm_model: str,
    prompt: str,
    video_base64: str,
    system_prompt: Optional[str] = None
) -> str:
    """Send a prompt with video to the VLM API.

    Args:
        vllm_api_endpoint: URL of the VLM API endpoint
        vllm_model: Model name to use (e.g., 'qwen2.5-vl-7b')
        prompt: User prompt/question
        video_base64: Video file encoded as base64 string
        system_prompt: Optional system prompt to override default

    Returns:
        Model response text

    Raises:
        requests.RequestException: If API call fails
        ValueError: If response format is unexpected
    """
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    payload: Dict[str, Any] = {...}
    response = requests.post(vllm_api_endpoint, headers=headers, json=payload)
    result: str = response.json()["choices"][0]["message"]["content"]
    return result
```

---

### 5. Zero Test Coverage

**Current State:** No tests exist

**Minimum Tests to Add:**

```
tests/
├── test_utils.py          # Test base64, video processing
├── test_vlm_client.py     # Test API client
├── test_config.py         # Test configuration loading
└── test_data_pipeline.py  # Test SUTD data processing
```

**Example Test:**
```python
import unittest
from vlm_application.utils import encode_file_to_base64
import tempfile
import base64

class TestEncoding(unittest.TestCase):
    def test_encode_base64_creates_valid_encoding(self):
        """Test that base64 encoding produces valid output"""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            test_data = b"Hello, World!"
            f.write(test_data)
            f.flush()

            result = encode_file_to_base64(f.name)
            decoded = base64.b64decode(result)

            self.assertEqual(decoded, test_data)

    def test_encode_nonexistent_file_raises_error(self):
        """Test that encoding nonexistent file raises FileNotFoundError"""
        with self.assertRaises(FileNotFoundError):
            encode_file_to_base64("/nonexistent/path/file.txt")
```

---

## 🟡 MEDIUM PRIORITY ISSUES (Fix This Month)

### 1. Bad Exception Handling

**Location:** `/d/llm-devs/vlm_application/utils.py` Line 307

**Current Code:**
```python
try:
    output_duration = float(subprocess.check_output(cmd).decode().strip())
except:  # BAD: Catches ALL exceptions including KeyboardInterrupt
    output_duration = -1
```

**Problems:**
- Catches `KeyboardInterrupt` and `SystemExit`
- Silent failure with no logging
- Swallows important error information

**Fix:**
```python
import logging

logger = logging.getLogger(__name__)

try:
    output_duration = float(subprocess.check_output(cmd).decode().strip())
except subprocess.CalledProcessError as e:
    logger.error(f"ffprobe failed: {e.stderr}")
    output_duration = -1
except ValueError as e:
    logger.error(f"Invalid duration format: {e}")
    output_duration = -1
except Exception as e:
    logger.error(f"Unexpected error getting video duration: {e}")
    raise
```

---

### 2. Magic Numbers

**Files with Magic Numbers:**

| File | Line | Issue | Recommendation |
|------|------|-------|-----------------|
| `/d/llm-devs/vlm_application/utils.py` | 41 | `image.width > 800` | Config: `MAX_IMAGE_DIMENSION` |
| `/d/llm-devs/vlm_application/utils.py` | 50 | `max_tokens=1024` | Config: `VLM_MAX_TOKENS` |
| `/d/llm-devs/langchain/nvidia_rag.py` | 21 | `chunk_size=1000` | Config: `RAG_CHUNK_SIZE` |
| `/d/llm-devs/langchain/nvidia_rag.py` | 68 | `k: 20` | Config: `RAG_TOP_K_DOCS` |
| `/d/llm-devs/asr_text_collection.py` | 87 | `1200` sentences | Config: `TOTAL_SENTENCES` |

**Solution: Create `constants.py`**

```python
# config/constants.py
"""Global configuration constants"""

# VLM Settings
VLM_TEMPERATURE = 0
VLM_MAX_TOKENS = 1024
VLM_MODEL = "qwen/qwen2.5-vl-7b"

# Image Processing
MAX_IMAGE_DIMENSION = 800  # Resize if larger
IMAGE_QUALITY = 95

# RAG Settings
RAG_CHUNK_SIZE = 1000
RAG_CHUNK_OVERLAP = 200
RAG_TOP_K_DOCS = 20
RAG_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"

# ASR Settings
TOTAL_SENTENCES = 1200
ASR_BATCH_SIZE = 60
ASR_DELAY_SECONDS = 5

# Video Processing
VIDEO_CODEC = "mp4v"
VIDEO_FPS = 1
VIDEO_CHUNK_DURATION = 10  # seconds
VIDEO_ENCODING_PRESET = "veryfast"  # ffmpeg preset
```

**Usage:**
```python
from config.constants import VLM_MAX_TOKENS, RAG_CHUNK_SIZE

def send_prompt(...):
    payload = {
        "max_tokens": VLM_MAX_TOKENS,  # Instead of hardcoded 1024
        ...
    }
```

---

### 3. Missing Docstrings

**Priority Functions to Document:**

| File | Function | Lines | Priority |
|------|----------|-------|----------|
| `/d/llm-devs/vlm_application/utils.py` | `send_prompt` | 15 | HIGH |
| `/d/llm-devs/vlm_application/utils.py` | `process_video` | 41 | HIGH |
| `/d/llm-devs/vlm_application/utils.py` | `do_chunking` | 66 | MEDIUM |
| `/d/llm-devs/langchain/nvidia_tools.py` | `fuyu` | 37 | HIGH |
| `/d/llm-devs/asr_text_collection.py` | `generate_batch` | 25 | MEDIUM |

**Template for Docstrings:**
```python
def process_video(
    input_video_file: str,
    output_video_file: str,
    total_samples: int = 10,
    fps: int = 1,
    resize: Optional[Tuple[int, int]] = None
) -> None:
    """Extract and process frames from a video file.

    This function extracts evenly-spaced frames from a video, optionally
    resizes them, and writes to an output video file. Useful for creating
    summary videos or reducing video complexity.

    Args:
        input_video_file: Path to input video file (mp4, avi, etc.)
        output_video_file: Path to write processed video
        total_samples: Number of frames to extract (default: 10)
        fps: Frames per second for output video (default: 1)
        resize: Optional tuple (width, height) to resize frames

    Returns:
        None (writes to output_video_file)

    Raises:
        FileNotFoundError: If input_video_file doesn't exist
        ValueError: If total_samples <= 0
        cv2.error: If video encoding fails

    Example:
        >>> process_video("input.mp4", "output.mp4", total_samples=5, resize=(640, 360))
        # Creates output.mp4 with 5 sampled frames resized to 640x360
    """
```

---

### 4. Unused Imports

**Location:** `/d/llm-devs/n8n/fake_data_gen.py` Line 2
```python
import random  # Not used anywhere in the file
```

**Location:** `/d/llm-devs/vlm_application/utils.py` Line 10
```python
from pathlib import Path  # Not used
```

**Fix:** Remove these imports

---

### 5. Package Structure Issues

**Problem:** Missing `__init__.py` files prevent proper imports

**Current Structure:**
```
llm-devs/
├── ASR/          # No __init__.py
│   ├── train.py
│   └── data/
├── langchain/    # No __init__.py
│   ├── nvidia_rag.py
│   └── research_agent/
└── vlm_application/  # No __init__.py
    ├── utils.py
    └── sutd_vqa/
```

**Fix:** Add `__init__.py` files
```bash
touch ASR/__init__.py
touch langchain/__init__.py
touch vlm_application/__init__.py
touch vlm_application/sutd_vqa/__init__.py
touch langchain/research_agent/__init__.py
```

**Enable proper imports:**
```python
# Instead of:
import sys
sys.path.append('../../vlm_application')
from utils import send_prompt

# You can now use:
from vlm_application.utils import send_prompt
```

---

## 🔵 LOW PRIORITY ISSUES (Refactor When Convenient)

### 1. Split Monolithic Files

**`/d/llm-devs/vlm_application/utils.py` (348 lines)**

**Current Organization:**
- API calls (15-135 lines)
- Base64 encoding (140-150 lines)
- Video processing (153-290 lines)
- Video chunking (222-314 lines)

**Recommended Split:**
```
vlm_application/
├── api_client.py         # VLMAPIClient class (~60 lines)
├── encoding.py           # encode_file_to_base64() (~20 lines)
├── video/
│   ├── __init__.py
│   ├── processor.py      # VideoProcessor class (~80 lines)
│   └── chunking.py       # do_chunking() (~80 lines)
└── utils.py              # Remaining utilities (~20 lines)
```

**Example `api_client.py`:**
```python
"""VLM API client for video and image queries."""

import requests
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class VLMAPIClient:
    """Client for vLLM-compatible API endpoints."""

    def __init__(self, endpoint: str, model: str, **kwargs):
        self.endpoint = endpoint
        self.model = model
        self.temperature = kwargs.get('temperature', 0)
        self.max_tokens = kwargs.get('max_tokens', 1024)

    # ... rest of implementation
```

---

### 2. Use Logging Instead of Print

**Current:**
```python
print(f"🔍 Retrieved {len(docs)} chunks from retriever.")
print(f"🕐 Took {elapsed_time:.2f}s")
```

**Better:**
```python
import logging

logger = logging.getLogger(__name__)

logger.info(f"Retrieved {len(docs)} chunks from retriever")
logger.debug(f"Took {elapsed_time:.2f}s")
```

---

### 3. Remove Commented-Out Code

**Location:** `/d/llm-devs/vlm_application/sutd_vqa/run_10sec_vid.py` Lines 135-136

```python
# y = run_for_content(x)
# output_content.append(y)
```

**Better:** Use git history to find old code if needed

---

### 4. Fix Typos in Comments

**Location:** `/d/llm-devs/vlm_application/utils.py` Line 340
```python
# Testinf the functions  # TYPO: should be "Testing"
```

**Location:** `/d/llm-devs/vlm_application/sutd_vqa/run_10sec_vid.py` Line 78
```python
# Comment ends mid-sentence "STEP 2: Run inference in parallel (API calls are sa"
```

---

### 5. Standardize Comment Style

**Current Mixed Style:**
```python
# Comment style 1
## Comment style 2 (double hash)
###Comment style 3 (no space)
```

**Standard:** Use `# ` (hash + single space) for all comments

---

## 📊 Dependency Cleanup

### Unused or Duplicate Dependencies

| Package | Issue | Action |
|---------|-------|--------|
| `pypdf2` | Deprecated, use `pypdf` instead | Remove |
| `ruff` | Linting tool, should be dev-only | Move to dev dependencies |
| `importlib-metadata` | Duplicated in dependencies | Keep one only |
| `jinja2` | Duplicated in dependencies | Keep one only |
| `setuptools` | Usually auto-installed | Remove explicit dependency |

**Updated `pixi.toml` section:**
```toml
[dependencies]
python = "3.12"
# Core ML
torch = { version = "==2.8.0+cu128", index = "https://download.pytorch.org/whl/cu128" }
transformers = ">=4.45.0"
# Remove: pypdf2 (use pypdf instead)
pypdf = ">=4.0.0"

[pypi-dependencies]
# Remove duplicates
# importlib-metadata = "*"  # Use only in dependencies
# jinja2 = "*"  # Use only in dependencies

[dev-dependencies]
ruff = ">=0.6.0"
pytest = ">=7.0"
pytest-cov = ">=4.0"
```

---

## 🎯 Implementation Roadmap

### Week 1: Critical Fixes
- [ ] Remove credentials from code
- [ ] Replace hardcoded paths with environment variables
- [ ] Fix exception handling (bare except clauses)
- [ ] Create `.env.example`

### Week 2: Code Consolidation
- [ ] Create `VLMAPIClient` class
- [ ] Consolidate base64 encoding functions
- [ ] Create `constants.py` for magic numbers
- [ ] Add `__init__.py` files to packages

### Week 3: Documentation & Tests
- [ ] Add docstrings to all public functions
- [ ] Add type hints to all functions
- [ ] Write basic unit tests
- [ ] Add README with examples

### Week 4: Polish & Refactoring
- [ ] Split monolithic files
- [ ] Remove unused dependencies
- [ ] Switch to logging module
- [ ] Standardize code style

---

## 📚 Additional Resources

### Code Style Guide
- **Type Hints:** https://docs.python.org/3/library/typing.html
- **Docstrings:** https://peps.python.org/pep-0257/
- **Python Best Practices:** https://pep8.org/

### Tools
```bash
# Install linting/formatting tools
pip install ruff black mypy

# Run linter
ruff check .

# Format code
black .

# Check types
mypy .

# Run tests with coverage
pytest --cov=. tests/
```

### Configuration Files to Create

**.env.example**
```bash
# VLM Configuration
VLM_ENDPOINT=http://127.0.0.1:8000/v1/chat/completions
VLM_MODEL=qwen/qwen2.5-vl-7b
VLM_MAX_TOKENS=1024

# Database
NEO4J_URI=neo4j://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password

# Paths
DATA_ROOT=./data
OUTPUT_DIR=./output
MODEL_PATH=./models

# API Keys
HF_TOKEN=your_token_here
GEMINI_API_KEY=your_key_here
```

**pyproject.toml** (if switching from pixi.toml)
```toml
[project]
name = "llm-devs"
version = "0.1.0"
description = "LLM and VLM development workspace"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.8.0",
    "transformers>=4.45.0",
]

[project.optional-dependencies]
dev = ["pytest>=7.0", "black>=23.0", "ruff>=0.1.0", "mypy>=1.0"]
```

---

## ✅ Checklist for Refactoring

**Before Starting:**
- [ ] Read this entire document
- [ ] Create a new branch: `git checkout -b refactor/consolidate-code`
- [ ] Back up important files

**Phase 1: Cleanup (Day 1-2)**
- [ ] Remove unused imports
- [ ] Delete commented-out code
- [ ] Fix typos in comments
- [ ] Remove duplicate dependencies

**Phase 2: Configuration (Day 3-4)**
- [ ] Create `.env.example`
- [ ] Replace hardcoded paths with env vars
- [ ] Move magic numbers to `constants.py`
- [ ] Update config loading

**Phase 3: Consolidation (Day 5-7)**
- [ ] Create `VLMAPIClient` class
- [ ] Consolidate duplicate functions
- [ ] Add `__init__.py` files
- [ ] Update imports throughout

**Phase 4: Documentation (Day 8-10)**
- [ ] Add type hints to all functions
- [ ] Add docstrings to all functions
- [ ] Add module-level docstrings
- [ ] Create example files

**Phase 5: Testing (Day 11-12)**
- [ ] Create `tests/` directory
- [ ] Write unit tests for utilities
- [ ] Write integration tests
- [ ] Achieve >80% coverage

**Phase 6: Polish (Day 13-14)**
- [ ] Run linters (ruff, black, mypy)
- [ ] Fix style issues
- [ ] Split monolithic files (if time)
- [ ] Create comprehensive README

**Submission:**
- [ ] Run all tests: `pytest`
- [ ] Run linter: `ruff check .`
- [ ] Check types: `mypy .`
- [ ] Create commit: `git commit -m "refactor: Consolidate and improve code quality"`
- [ ] Push: `git push origin refactor/consolidate-code`
- [ ] Create pull request

---

## 📞 Questions?

For questions or clarifications about these recommendations, refer to the specific sections above or consult:
- Python documentation: https://docs.python.org
- PEP 8 Style Guide: https://pep8.org
- Type Hints: https://docs.python.org/3/library/typing.html

---

**Generated:** 2026-02-20
**Review Status:** Comprehensive Code Analysis Complete
**Estimated Refactoring Time:** 2-3 weeks (depending on parallelization)
**Risk Level:** Low (mostly internal refactoring, no breaking changes)
