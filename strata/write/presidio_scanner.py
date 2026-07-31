"""Presidio-based PII scanner — production-grade detection (optional [pii] extra).

Microsoft Presidio provides ML-based PII detection with 30+ entity types
including names, addresses, dates, medical info, financial data, and more.
This scanner slots in behind the same PIIScanner protocol as the regex fallback.

Install: pip install strata-memory[pii]

Entity types detected: https://microsoft.github.io/presidio/supported_entities/
"""
from __future__ import annotations

from .pii import PIIMatch, PIIScanner


class PresidioScanner(PIIScanner):
    """Production PII detection using Microsoft Presidio.
    
    Advantages over RegexScanner:
    - 30+ entity types vs 5 basic patterns
    - Context-aware detection (reduces false positives)
    - Multilingual support
    - Customizable recognizers
    
    Example:
        >>> scanner = PresidioScanner()
        >>> matches = scanner.scan("My name is John Smith and my email is john@example.com")
        >>> [m.type for m in matches]
        ['PERSON', 'EMAIL_ADDRESS']
    """
    
    def __init__(self, language: str = "en"):
        """Initialize Presidio analyzer.
        
        Args:
            language: ISO 639-1 language code (default: 'en')
        """
        try:
            from presidio_analyzer import AnalyzerEngine
        except ImportError as e:
            raise ImportError(
                "Presidio requires the [pii] extra: pip install strata-memory[pii]"
            ) from e
        
        self.analyzer = AnalyzerEngine()
        self.language = language
    
    def scan(self, text: str) -> list[PIIMatch]:
        """Scan text for PII entities.
        
        Args:
            text: Input text to scan
            
        Returns:
            List of PIIMatch objects with entity type, position, and value
        """
        if not text:
            return []
        
        results = self.analyzer.analyze(
            text=text,
            language=self.language
        )
        
        return [
            PIIMatch(
                type=r.entity_type,
                start=r.start,
                end=r.end,
                value=text[r.start:r.end]
            )
            for r in results
            if r.score >= 0.35  # Lowered from 0.5 for better recall on short inputs
        ]
