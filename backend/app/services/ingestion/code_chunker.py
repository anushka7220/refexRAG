# code_chunker.py
#
# Chunks SOURCE CODE files into embeddable pieces. Separate from chunker.py
# because prose and code have opposite chunking needs:
#
#   Prose (issues, PR bodies): fixed token windows WITH overlap, because
#   meaning flows across sentence boundaries and overlap preserves context.
#
#   Code: syntactic boundaries (functions, classes) with NO overlap, because
#   a half-function is meaningless to retrieve and a duplicated function
#   across two chunks pollutes similarity search with near-identical vectors.
#
# STRATEGY BY LANGUAGE:
#   Python  -> real AST parsing via the stdlib ast module. Exact function and
#              class spans, including decorators and docstrings.
#   JS/TS/Go/Java/etc -> pragmatic regex boundary detection on common
#              function/class declaration patterns.
#   Everything else -> line-window fallback (60 lines, 10 overlap).
#
# EVERY chunk gets a location header baked into the embedded text:
#   "File: app/services/rag/critic.py (lines 88-140)"
# so the vector itself carries where-am-I context, and the LLM can cite
# file and line in answers.
#
# WHY NOT TREE-SITTER:
# tree-sitter gives perfect parsing for 20+ languages but adds a compiled
# dependency and grammar management. The pragmatic tiers above cover the
# large majority of real repos. tree-sitter is the documented upgrade path
# once language coverage becomes a real user complaint, not before.

import ast
import re
import structlog
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from app.models.chunk import Chunk
from app.utils.hashing import sha256

log = structlog.get_logger(__name__)

# Fallback window for languages we cannot parse structurally.
FALLBACK_WINDOW_LINES = 60
FALLBACK_OVERLAP_LINES = 10

# A single structural unit larger than this gets re-split by the fallback,
# because one 800-line class as one chunk blows past embedding token limits.
MAX_UNIT_LINES = 120

# Declaration patterns for regex-tier languages. Deliberately loose:
# we only need boundaries, not a parse tree.
_DECL_PATTERNS = re.compile(
    r"^(\s*)("
    r"(export\s+)?(default\s+)?(async\s+)?function\s+\w+"      # JS/TS function
    r"|(export\s+)?(abstract\s+)?class\s+\w+"                  # JS/TS/Java class
    r"|(public|private|protected|static|final)\s+[\w<>,\s\[\]]+\s+\w+\s*\("  # Java/C# method
    r"|func\s+(\(\w+\s+\*?\w+\)\s+)?\w+\s*\("                  # Go func / method
    r"|fn\s+\w+"                                                # Rust
    r"|def\s+\w+"                                               # Ruby
    r")"
)


@dataclass
class SourceFile:
    """One source file pulled from the repo tarball, ready to chunk."""
    path:     str           # repo-relative path, e.g. "app/services/rag/critic.py"
    content:  str
    language: str           # "python", "javascript", "other", ...


# Extension to language mapping. Drives which chunking tier a file gets.
_EXT_LANG = {
    ".py": "python",
    ".js": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "typescript",
    ".go": "go", ".rs": "rust", ".java": "java",
    ".rb": "ruby", ".cs": "csharp", ".php": "php",
    ".c": "c", ".h": "c", ".cpp": "cpp", ".hpp": "cpp",
    ".swift": "swift", ".kt": "kotlin", ".scala": "scala",
    ".sql": "sql", ".sh": "shell",
    ".md": "markdown", ".rst": "markdown", ".txt": "markdown",
    ".yaml": "config", ".yml": "config", ".toml": "config", ".json": "config",
}

# Languages that get structural (regex) boundary detection.
_REGEX_TIER = {"javascript", "typescript", "go", "rust", "java", "ruby", "csharp", "php", "kotlin", "swift", "scala"}


def language_for(path: str) -> Optional[str]:
    """Returns the language for a file path, or None if we do not index it."""
    for ext, lang in _EXT_LANG.items():
        if path.endswith(ext):
            return lang
    return None


class CodeChunker:
    """
    Turns SourceFile objects into Chunk objects with source_type "code".

    Usage (called by IngestionOrchestrator after code fetch):
        chunker = CodeChunker(repo_id, owner, repo_name, default_branch)
        chunks = chunker.chunk_files(source_files)
    """

    def __init__(self, repo_id: str, owner: str, repo_name: str, default_branch: str):
        self.repo_id = repo_id
        self.owner = owner
        self.repo_name = repo_name
        self.default_branch = default_branch

    # ── Public entry ───────────────────────────────────────────────────────

    def chunk_files(self, files: list[SourceFile]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for f in files:
            try:
                if f.language == "python":
                    units = self._split_python(f.content)
                elif f.language in _REGEX_TIER:
                    units = self._split_by_declarations(f.content)
                elif f.language == "markdown":
                    units = self._split_markdown(f.content)
                else:
                    units = self._split_by_lines(f.content)
            except Exception as e:
                # A single unparseable file must never kill ingestion.
                log.warning("code_chunk_failed_fallback", path=f.path, error=str(e))
                units = self._split_by_lines(f.content)

            for start, end, text in units:
                if not text.strip():
                    continue
                chunks.append(self._make_chunk(f, start, end, text))

        log.info("code_chunking_complete", files=len(files), chunks=len(chunks))
        return chunks

    # ── Tier 1: Python via AST ─────────────────────────────────────────────

    def _split_python(self, content: str) -> list[tuple[int, int, str]]:
        """
        Splits a Python file at top-level function and class boundaries
        using the real parser, so spans are exact (decorators included).
        Module-level code between definitions is grouped into its own chunks.
        """
        lines = content.splitlines()
        tree = ast.parse(content)

        units: list[tuple[int, int, str]] = []
        covered: set[int] = set()

        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                # decorator_list nodes start before node.lineno; take the earliest
                start = min(
                    [node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])]
                )
                end = node.end_lineno or node.lineno
                text = "\n".join(lines[start - 1:end])

                # Oversized classes get re-split so one giant class does not
                # become one giant unembeddable chunk.
                if end - start + 1 > MAX_UNIT_LINES:
                    for s2, e2, t2 in self._split_by_lines(text):
                        units.append((start + s2 - 1, start + e2 - 1, t2))
                else:
                    units.append((start, end, text))
                covered.update(range(start, end + 1))

        # Gather uncovered module-level lines (imports, constants, top-level
        # statements) into contiguous blocks. These carry real signal, e.g.
        # HTTP_TIMEOUT = 30 lives here, which is exactly the kind of fact
        # users ask about.
        block: list[tuple[int, str]] = []
        for i, line in enumerate(lines, start=1):
            if i in covered:
                if block:
                    units.append((block[0][0], block[-1][0], "\n".join(t for _, t in block)))
                    block = []
                continue
            block.append((i, line))
        if block:
            units.append((block[0][0], block[-1][0], "\n".join(t for _, t in block)))

        units.sort(key=lambda u: u[0])
        return units

    # ── Tier 2: regex declaration boundaries ───────────────────────────────

    def _split_by_declarations(self, content: str) -> list[tuple[int, int, str]]:
        """
        Best-effort structural split for JS/TS/Go/Java-family files.
        Finds declaration lines and treats each as the start of a unit
        running until the next declaration. Not a parse, just boundaries,
        which is all chunking needs.
        """
        lines = content.splitlines()
        boundaries = [i for i, line in enumerate(lines, start=1) if _DECL_PATTERNS.match(line)]

        if not boundaries:
            return self._split_by_lines(content)

        units = []
        # Preamble before the first declaration (imports, constants)
        if boundaries[0] > 1:
            units.append((1, boundaries[0] - 1, "\n".join(lines[:boundaries[0] - 1])))

        for idx, start in enumerate(boundaries):
            end = (boundaries[idx + 1] - 1) if idx + 1 < len(boundaries) else len(lines)
            text = "\n".join(lines[start - 1:end])
            if end - start + 1 > MAX_UNIT_LINES:
                for s2, e2, t2 in self._split_by_lines(text):
                    units.append((start + s2 - 1, start + e2 - 1, t2))
            else:
                units.append((start, end, text))
        return units

    # ── Tier 3: markdown by headings ───────────────────────────────────────

    def _split_markdown(self, content: str) -> list[tuple[int, int, str]]:
        """
        READMEs and docs split at headings. The README is often the single
        highest-value file in a repo for "what does this project do", so it
        deserves clean section-level chunks.
        """
        lines = content.splitlines()
        boundaries = [i for i, line in enumerate(lines, start=1) if line.startswith("#")]
        if not boundaries:
            return self._split_by_lines(content)

        units = []
        if boundaries[0] > 1:
            units.append((1, boundaries[0] - 1, "\n".join(lines[:boundaries[0] - 1])))
        for idx, start in enumerate(boundaries):
            end = (boundaries[idx + 1] - 1) if idx + 1 < len(boundaries) else len(lines)
            units.append((start, end, "\n".join(lines[start - 1:end])))
        return units

    # ── Tier 4: line-window fallback ───────────────────────────────────────

    def _split_by_lines(self, content: str) -> list[tuple[int, int, str]]:
        lines = content.splitlines()
        units = []
        step = FALLBACK_WINDOW_LINES - FALLBACK_OVERLAP_LINES
        i = 0
        while i < len(lines):
            start = i + 1
            end = min(i + FALLBACK_WINDOW_LINES, len(lines))
            units.append((start, end, "\n".join(lines[i:end])))
            if end == len(lines):
                break
            i += step
        return units

    # ── Chunk factory ──────────────────────────────────────────────────────

    def _make_chunk(self, f: SourceFile, start: int, end: int, text: str) -> Chunk:
        """
        Builds a code Chunk. The location header is prepended to the content
        BEFORE embedding, so the vector carries file identity and answers
        can cite exact locations. The GitHub URL deep-links to the line span.
        """
        header = f"File: {f.path} (lines {start}-{end})\n"
        content = header + text

        url = (
            f"https://github.com/{self.owner}/{self.repo_name}"
            f"/blob/{self.default_branch}/{f.path}#L{start}-L{end}"
        )

        return Chunk(
            repo_id=self.repo_id,
            content=content,
            source_type="code",
            source_id=f.path,
            status="none",
            content_hash=sha256(content),
            source_created_at=datetime.now(timezone.utc),
            url=url,
            version_tag=None,
            embedding=[],
            id="",
            file_path=f.path,
            language=f.language,
            start_line=start,
            end_line=end,
        )