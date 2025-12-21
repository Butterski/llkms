"""
LLKMS Doctor - System health check utility.

Checks all dependencies and configurations to ensure the system is ready to run.
"""

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

from llkms.utils.logger import logger


class CheckStatus(Enum):
    """Status of a health check."""

    OK = "✅"
    WARNING = "⚠️"
    ERROR = "❌"
    SKIP = "⏭️"


@dataclass
class CheckResult:
    """Result of a single health check."""

    name: str
    status: CheckStatus
    message: str
    fix_hint: Optional[str] = None


class LLKMSDoctor:
    """System health checker for LLKMS."""

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the doctor.

        Args:
            config: Optional configuration dictionary.
        """
        self.config = config or {}
        self.results: List[CheckResult] = []

    def run_all_checks(self) -> List[CheckResult]:
        """Run all health checks and return results."""
        self.results = []

        # Core Python dependencies
        self._check_python_version()
        self._check_core_packages()

        # External tools
        self._check_tesseract()
        self._check_poppler()

        # Optional: Ollama (if configured)
        if self.config.get("model", {}).get("provider") == "ollama":
            self._check_ollama()

        # AWS credentials
        self._check_aws_credentials()

        # API keys based on config
        self._check_api_keys()

        # Vector store cache
        self._check_vector_store_cache()

        return self.results

    def _add_result(self, name: str, status: CheckStatus, message: str, fix_hint: Optional[str] = None):
        """Add a check result."""
        self.results.append(CheckResult(name, status, message, fix_hint))

    def _check_python_version(self):
        """Check Python version."""
        version = sys.version_info
        version_str = f"{version.major}.{version.minor}.{version.micro}"

        if version.major == 3 and version.minor >= 9:
            self._add_result("Python Version", CheckStatus.OK, f"Python {version_str}")
        else:
            self._add_result(
                "Python Version",
                CheckStatus.ERROR,
                f"Python {version_str} (requires 3.9+)",
                "Install Python 3.9 or higher from https://python.org",
            )

    def _check_core_packages(self):
        """Check core Python packages."""
        packages = [
            ("langchain", "langchain"),
            ("langchain_community", "langchain-community"),
            ("langchain_openai", "langchain-openai"),
            ("faiss", "faiss-cpu"),
            ("boto3", "boto3"),
            ("unstructured", "unstructured"),
            ("PIL", "pillow"),
            ("pillow_heif", "pillow-heif"),
            ("pdfminer", "pdfminer.six"),
            ("yaml", "pyyaml"),
            ("questionary", "questionary"),
            ("docx", "python-docx"),
        ]

        missing = []
        for import_name, package_name in packages:
            try:
                __import__(import_name)
            except ImportError:
                missing.append(package_name)

        if not missing:
            self._add_result("Python Packages", CheckStatus.OK, "All core packages installed")
        else:
            self._add_result(
                "Python Packages",
                CheckStatus.ERROR,
                f"Missing: {', '.join(missing)}",
                f"Run: pip install {' '.join(missing)}",
            )

    def _check_tesseract(self):
        """Check Tesseract OCR installation."""
        tesseract_path = shutil.which("tesseract")

        if tesseract_path:
            try:
                result = subprocess.run(
                    ["tesseract", "--version"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                version_line = result.stdout.split("\n")[0] if result.stdout else "unknown version"
                self._add_result("Tesseract OCR", CheckStatus.OK, version_line)

                # Check for Polish language data
                self._check_tesseract_languages()
            except subprocess.TimeoutExpired:
                self._add_result(
                    "Tesseract OCR",
                    CheckStatus.WARNING,
                    "Installed but not responding",
                    "Try reinstalling Tesseract",
                )
            except Exception as e:
                self._add_result(
                    "Tesseract OCR",
                    CheckStatus.WARNING,
                    f"Found at {tesseract_path} but error: {e}",
                )
        else:
            # Check common Windows paths
            common_paths = [
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                os.path.expanduser(r"~\AppData\Local\Tesseract-OCR\tesseract.exe"),
            ]

            found_path = None
            for path in common_paths:
                if os.path.exists(path):
                    found_path = path
                    break

            if found_path:
                self._add_result(
                    "Tesseract OCR",
                    CheckStatus.WARNING,
                    f"Found at {found_path} but not in PATH",
                    f"Add to PATH: set PATH=%PATH%;{os.path.dirname(found_path)}",
                )
            else:
                self._add_result(
                    "Tesseract OCR",
                    CheckStatus.ERROR,
                    "Not installed (required for OCR/image processing)",
                    "Download from: https://github.com/UB-Mannheim/tesseract/wiki\n"
                    "   Or run: winget install UB-Mannheim.TesseractOCR",
                )

    def _check_tesseract_languages(self):
        """Check available Tesseract language packs."""
        try:
            result = subprocess.run(
                ["tesseract", "--list-langs"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            languages = result.stdout.strip().split("\n")[1:]  # Skip header

            has_eng = "eng" in languages
            has_pol = "pol" in languages

            if has_eng and has_pol:
                self._add_result(
                    "Tesseract Languages",
                    CheckStatus.OK,
                    f"Available: {', '.join(languages[:5])}{'...' if len(languages) > 5 else ''}",
                )
            elif has_eng:
                self._add_result(
                    "Tesseract Languages",
                    CheckStatus.WARNING,
                    "Polish (pol) language pack not installed",
                    "Download pol.traineddata from https://github.com/tesseract-ocr/tessdata",
                )
            else:
                self._add_result(
                    "Tesseract Languages",
                    CheckStatus.WARNING,
                    f"Only: {', '.join(languages)}",
                    "Install additional language packs as needed",
                )
        except Exception:
            pass  # Skip if tesseract not working

    def _check_poppler(self):
        """Check Poppler (PDF utilities) installation."""
        # Check for pdftoppm which is part of poppler
        pdftoppm_path = shutil.which("pdftoppm")

        if pdftoppm_path:
            self._add_result("Poppler (PDF tools)", CheckStatus.OK, f"Found: {pdftoppm_path}")
        else:
            # Check common Windows paths
            common_paths = [
                r"C:\Program Files\poppler\Library\bin\pdftoppm.exe",
                r"C:\poppler\Library\bin\pdftoppm.exe",
                os.path.expanduser(r"~\poppler\Library\bin\pdftoppm.exe"),
            ]

            found_path = None
            for path in common_paths:
                if os.path.exists(path):
                    found_path = path
                    break

            if found_path:
                self._add_result(
                    "Poppler (PDF tools)",
                    CheckStatus.WARNING,
                    f"Found at {found_path} but not in PATH",
                    f"Add to PATH: {os.path.dirname(found_path)}",
                )
            else:
                self._add_result(
                    "Poppler (PDF tools)",
                    CheckStatus.WARNING,
                    "Not found (optional, improves PDF processing)",
                    "Download from: https://github.com/oschwartz10612/poppler-windows/releases\n"
                    "   Extract and add bin folder to PATH",
                )

    def _check_ollama(self):
        """Check Ollama installation and running status."""
        ollama_path = shutil.which("ollama")

        if not ollama_path:
            self._add_result(
                "Ollama",
                CheckStatus.ERROR,
                "Not installed (required for local models)",
                "Download from: https://ollama.ai",
            )
            return

        # Check if Ollama is running
        try:
            import urllib.request

            api_base = self.config.get("model", {}).get("api_base", "http://localhost:11434")
            # Remove /v1 suffix if present
            base_url = api_base.replace("/v1", "")

            req = urllib.request.Request(f"{base_url}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=3) as response:
                if response.status == 200:
                    import json

                    data = json.loads(response.read())
                    models = [m["name"] for m in data.get("models", [])]

                    if models:
                        self._add_result(
                            "Ollama",
                            CheckStatus.OK,
                            f"Running with models: {', '.join(models[:3])}{'...' if len(models) > 3 else ''}",
                        )

                        # Check if configured model is available
                        configured_model = self.config.get("model", {}).get("model", "")
                        model_base = configured_model.split(":")[0]
                        if not any(model_base in m for m in models):
                            self._add_result(
                                "Ollama Model",
                                CheckStatus.WARNING,
                                f"Configured model '{configured_model}' not found",
                                f"Run: ollama pull {configured_model}",
                            )
                    else:
                        self._add_result(
                            "Ollama",
                            CheckStatus.WARNING,
                            "Running but no models installed",
                            "Run: ollama pull <model_name>",
                        )
        except urllib.error.URLError:
            self._add_result(
                "Ollama",
                CheckStatus.WARNING,
                "Installed but not running",
                "Start Ollama: ollama serve (or run Ollama app)",
            )
        except Exception as e:
            self._add_result(
                "Ollama",
                CheckStatus.WARNING,
                f"Error checking status: {e}",
            )

    def _check_aws_credentials(self):
        """Check AWS credentials."""
        access_key = os.getenv("AWS_ACCESS_KEY_ID") or self.config.get("aws", {}).get("access_key_id")
        secret_key = os.getenv("AWS_SECRET_ACCESS_KEY") or self.config.get("aws", {}).get("secret_access_key")

        # Don't show actual keys, just check if they exist
        access_key = access_key if access_key and not access_key.startswith("${") else None
        secret_key = secret_key if secret_key and not secret_key.startswith("${") else None

        if access_key and secret_key:
            # Mask the key for display
            masked_key = (
                access_key[:4] + "*" * (len(access_key) - 8) + access_key[-4:] if len(access_key) > 8 else "***"
            )
            self._add_result("AWS Credentials", CheckStatus.OK, f"Configured (key: {masked_key})")
        elif access_key or secret_key:
            self._add_result(
                "AWS Credentials",
                CheckStatus.ERROR,
                "Incomplete credentials",
                "Set both AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in .env",
            )
        else:
            self._add_result(
                "AWS Credentials",
                CheckStatus.ERROR,
                "Not configured",
                "Add AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY to .env file",
            )

    def _check_api_keys(self):
        """Check API keys based on configuration."""
        model_provider = self.config.get("model", {}).get("provider", "")
        embeddings_provider = self.config.get("embeddings", {}).get("provider", "")

        # Check OpenAI API key
        if model_provider == "openai" or embeddings_provider == "openai":
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key:
                masked = openai_key[:7] + "*" * 10 + openai_key[-4:] if len(openai_key) > 15 else "***"
                self._add_result("OpenAI API Key", CheckStatus.OK, f"Configured ({masked})")
            else:
                self._add_result(
                    "OpenAI API Key",
                    CheckStatus.ERROR,
                    "Not configured (required for OpenAI provider)",
                    "Set OPENAI_API_KEY in .env file",
                )

        # Check DeepSeek API key
        if model_provider == "deepseek":
            deepseek_key = os.getenv("DEEPSEEK_API_KEY")
            config_key = self.config.get("model", {}).get("api_key", "")
            config_key = config_key if config_key and not config_key.startswith("${") else None

            if deepseek_key or config_key:
                key = deepseek_key or config_key
                masked = key[:4] + "*" * 10 + key[-4:] if len(key) > 12 else "***"
                self._add_result("DeepSeek API Key", CheckStatus.OK, f"Configured ({masked})")
            else:
                self._add_result(
                    "DeepSeek API Key",
                    CheckStatus.ERROR,
                    "Not configured (required for DeepSeek provider)",
                    "Set DEEPSEEK_API_KEY in .env file",
                )

    def _check_vector_store_cache(self):
        """Check vector store cache directory."""
        cache_dir = self.config.get("app", {}).get("vector_store_cache", "vector_store_cache")

        if os.path.exists(cache_dir):
            # Check if there's an index
            index_file = os.path.join(cache_dir, "faiss_index.index")
            docs_file = os.path.join(cache_dir, "docs.pkl")

            if os.path.exists(index_file) and os.path.exists(docs_file):
                # Get file size
                index_size = os.path.getsize(index_file) / (1024 * 1024)  # MB
                self._add_result(
                    "Vector Store Cache",
                    CheckStatus.OK,
                    f"Found ({index_size:.1f} MB index)",
                )
            else:
                self._add_result(
                    "Vector Store Cache",
                    CheckStatus.WARNING,
                    "Directory exists but no valid index",
                    "Run with --reindex to rebuild",
                )
        else:
            self._add_result(
                "Vector Store Cache",
                CheckStatus.WARNING,
                "Not found (will be created on first run)",
            )

    def print_results(self):
        """Print all check results in a formatted way."""
        print("\n" + "=" * 60)
        print("🔍 LLKMS Doctor - System Health Check")
        print("=" * 60 + "\n")

        # Group by status
        errors = [r for r in self.results if r.status == CheckStatus.ERROR]
        warnings = [r for r in self.results if r.status == CheckStatus.WARNING]
        ok = [r for r in self.results if r.status == CheckStatus.OK]

        for result in self.results:
            print(f"{result.status.value} {result.name}: {result.message}")
            if result.fix_hint and result.status in (CheckStatus.ERROR, CheckStatus.WARNING):
                for line in result.fix_hint.split("\n"):
                    print(f"   💡 {line}")
            print()

        # Summary
        print("-" * 60)
        print(f"Summary: {len(ok)} OK, {len(warnings)} Warnings, {len(errors)} Errors")

        if errors:
            print("\n⛔ Some critical issues need to be fixed before running LLKMS.")
        elif warnings:
            print("\n⚠️  Some optional features may not work correctly.")
        else:
            print("\n🎉 All checks passed! LLKMS is ready to run.")

        print("=" * 60 + "\n")

        return len(errors) == 0


def run_doctor(config: Optional[Dict] = None) -> bool:
    """
    Run the LLKMS doctor and print results.

    Args:
        config: Optional configuration dictionary.

    Returns:
        True if all critical checks passed, False otherwise.
    """
    doctor = LLKMSDoctor(config)
    doctor.run_all_checks()
    return doctor.print_results()
