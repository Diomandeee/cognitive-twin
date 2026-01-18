"""
Canonicalizer for Enhancer Agent.

Removes provider-isms and standardizes output patterns:
- Provider-specific phrases
- Excessive apologies
- Filler openings
- Permission closers
- Code and list formatting
"""

import re
from typing import Optional


# =============================================================================
# PROVIDER-ISM PATTERNS
# =============================================================================

PROVIDER_ISMS = [
    # AI self-reference
    r"as an ai(?: language model)?",
    r"as a large language model",
    r"i'?m (?:just )?an ai",
    r"i don'?t have (?:personal )?(?:opinions|feelings|preferences)",
    r"i'?m not able to",
    r"i can'?t (?:actually )?(?:browse|access|see)",
    
    # Over-apologizing (will be handled separately for reduction)
    # Over-hedging
    r"it'?s worth noting that",
    r"it'?s important to (?:note|mention|remember) that",
    r"please (?:note|keep in mind) that",
    r"i should mention that",
    
    # Filler acknowledgments (handled separately)
    
    # Capability disclaimers (when irrelevant)
    r"i can'?t browse the (?:web|internet)",
    r"i don'?t have access to (?:the internet|real-?time)",
    r"as of my (?:knowledge )?cutoff",
    r"my training data (?:only )?(?:goes|extends) up to",
    
    # Generic AI phrasing
    r"i would be happy to help",
    r"i'?m here to help",
    r"how can i assist you",
]


APOLOGY_PATTERNS = [
    r"i apologize for (?:any )?(?:confusion|inconvenience|misunderstanding)",
    r"sorry (?:for|about) (?:the )?(?:confusion|delay|misunderstanding)",
    r"my (?:sincere )?apologies",
    r"i'?m (?:truly |very )?sorry",
    r"please accept my apologies",
    r"i apologize(?:,? but)?",
    r"i'?m sorry(?:,? but)?",
    r"sorry for (?:any )?confusion",
]


DISCLAIMER_PATTERNS = [
    r"please note that this is not (?:legal|medical|financial) advice",
    r"this should not be taken as (?:professional )?advice",
    r"consult (?:a|with a) (?:professional|expert|specialist)",
    r"i'?m not a (?:lawyer|doctor|financial advisor)",
    r"this is for (?:informational|educational) purposes only",
]


FILLER_OPENINGS = [
    r"^sure[,!]?\s*",
    r"^certainly[,!]?\s*",
    r"^absolutely[,!]?\s*",
    r"^of course[,!]?\s*",
    r"^great[,!]?\s*",
    r"^alright[,!]?\s*",
    r"^okay[,!]?\s*",
    r"^yes[,!]?\s*",
    r"^i'?d be happy to\s*",
    r"^i'?ll be glad to\s*",
    r"^happy to help[,!]?\s*",
    r"^that'?s a (?:great|good|interesting) (?:question|point)[,!]?\s*",
    r"^great question[,!]?\s*",
]


PERMISSION_CLOSERS = [
    r"let me know if you(?:'d like| want| need)[^.!]*[.!]?\s*$",
    r"feel free to (?:ask|reach out|let me know)[^.!]*[.!]?\s*$",
    r"(?:please )?don'?t hesitate to[^.!]*[.!]?\s*$",
    r"if you (?:have any|need)[^.!]*questions[^.!]*[.!]?\s*$",
    r"hope (?:this|that) helps[.!]?\s*$",
    r"i hope this (?:helps|answers)[^.!]*[.!]?\s*$",
    r"is there anything else[^?]*\??\s*$",
    r"would you like (?:me to|more)[^?]*\??\s*$",
    r"do you want me to[^?]*\??\s*$",
    r"shall i[^?]*\??\s*$",
    r"should i[^?]*\??\s*$",
]


class Canonicalizer:
    """Canonicalizes assistant outputs for training."""
    
    def __init__(
        self,
        max_apologies: int = 1,
        keep_sensitive_disclaimers: bool = True,
        sensitive_topics: Optional[list] = None,
    ):
        self.max_apologies = max_apologies
        self.keep_sensitive_disclaimers = keep_sensitive_disclaimers
        self.sensitive_topics = sensitive_topics or [
            "legal", "medical", "financial", "health", "investment"
        ]
    
    # =========================================================================
    # PROVIDER-ISM REMOVAL
    # =========================================================================
    
    def remove_provider_isms(self, text: str) -> tuple[str, int]:
        """Remove provider-specific phrases from text."""
        result = text
        count = 0
        
        for pattern in PROVIDER_ISMS:
            matches = len(re.findall(pattern, result, flags=re.IGNORECASE))
            if matches:
                count += matches
                result = re.sub(pattern, "", result, flags=re.IGNORECASE | re.MULTILINE)
        
        # Clean up multiple newlines
        result = re.sub(r'\n{3,}', '\n\n', result)
        
        # Clean up leading/trailing whitespace
        result = result.strip()
        
        return result, count
    
    def reduce_apologies(self, text: str) -> tuple[str, int]:
        """Remove excessive apologies while preserving genuine ones."""
        result = text
        reduced = 0
        
        # Count total apologies
        total_apologies = sum(
            len(re.findall(pattern, text, re.IGNORECASE))
            for pattern in APOLOGY_PATTERNS
        )
        
        if total_apologies <= self.max_apologies:
            return text, 0
        
        # Track which to keep (first one)
        kept_count = 0
        
        for pattern in APOLOGY_PATTERNS:
            def replace_func(match):
                nonlocal kept_count, reduced
                if kept_count < self.max_apologies:
                    kept_count += 1
                    return match.group(0)
                else:
                    reduced += 1
                    return ""
            
            result = re.sub(pattern, replace_func, result, flags=re.IGNORECASE)
        
        # Clean up
        result = re.sub(r'\n{3,}', '\n\n', result)
        result = re.sub(r'  +', ' ', result)
        
        return result.strip(), reduced
    
    def remove_irrelevant_disclaimers(
        self,
        text: str,
        context: str = "",
    ) -> str:
        """Remove disclaimers that aren't relevant to the context."""
        if not self.keep_sensitive_disclaimers:
            # Remove all disclaimers
            result = text
            for pattern in DISCLAIMER_PATTERNS:
                result = re.sub(pattern, "", result, flags=re.IGNORECASE)
            return result.strip()
        
        # Check if context involves sensitive topics
        context_lower = context.lower()
        is_sensitive = any(
            topic in context_lower for topic in self.sensitive_topics
        )
        
        if is_sensitive:
            return text  # Keep disclaimers for sensitive content
        
        # Remove disclaimers
        result = text
        for pattern in DISCLAIMER_PATTERNS:
            result = re.sub(pattern, "", result, flags=re.IGNORECASE)
        
        return result.strip()
    
    # =========================================================================
    # OPENING/CLOSING STANDARDIZATION
    # =========================================================================
    
    def remove_filler_openings(self, text: str) -> tuple[str, int]:
        """Remove filler acknowledgment phrases at start."""
        result = text
        count = 0
        
        # Apply patterns in sequence (some may chain)
        for _ in range(3):  # Max 3 iterations
            old_result = result
            for pattern in FILLER_OPENINGS:
                if re.match(pattern, result, flags=re.IGNORECASE):
                    result = re.sub(pattern, "", result, count=1, flags=re.IGNORECASE)
                    count += 1
            
            if result == old_result:
                break
        
        return result.strip(), count
    
    def remove_permission_closers(self, text: str) -> tuple[str, int]:
        """Remove permission-seeking closers."""
        result = text
        count = 0
        
        for pattern in PERMISSION_CLOSERS:
            if re.search(pattern, result, flags=re.IGNORECASE | re.MULTILINE):
                result = re.sub(pattern, "", result, flags=re.IGNORECASE | re.MULTILINE)
                count += 1
        
        return result.rstrip(), count
    
    # =========================================================================
    # FORMATTING STANDARDIZATION
    # =========================================================================
    
    def standardize_code_blocks(self, text: str) -> str:
        """Standardize code block formatting."""
        
        def add_language(match):
            content = match.group(1)
            
            # Already has language
            if match.group(0).startswith('```') and \
               re.match(r'^```\w+\n', match.group(0)):
                return match.group(0)
            
            # Detect language
            if re.search(r'\bdef\s+\w+|^import\s+|^from\s+\w+\s+import|class\s+\w+.*:', content, re.MULTILINE):
                return f"```python\n{content}\n```"
            elif re.search(r'\bfunction\s+\w+|\bconst\s+\w+|=>\s*\{|\blet\s+\w+', content):
                return f"```javascript\n{content}\n```"
            elif re.search(r'\bfn\s+\w+|\blet\s+mut\s+|\bimpl\s+|\bpub\s+(?:fn|struct)', content):
                return f"```rust\n{content}\n```"
            elif re.search(r'\bfunc\s+\w+|\bpackage\s+\w+|\btype\s+\w+\s+struct', content):
                return f"```go\n{content}\n```"
            else:
                return f"```\n{content}\n```"
        
        # Find bare code blocks (no language specifier)
        text = re.sub(r'```\n([^`]+)\n```', add_language, text)
        
        return text
    
    def standardize_lists(
        self,
        text: str,
        prefer_numbered: bool = False,
    ) -> str:
        """Standardize list formatting."""
        if not prefer_numbered:
            return text
        
        # Convert bullet lists to numbered
        lines = text.split('\n')
        result_lines = []
        counter = 0
        in_list = False
        
        for line in lines:
            bullet_match = re.match(r'^(\s*)([-*•])\s+(.*)$', line)
            
            if bullet_match:
                indent = bullet_match.group(1)
                content = bullet_match.group(3)
                
                if not in_list:
                    in_list = True
                    counter = 1
                else:
                    counter += 1
                
                result_lines.append(f"{indent}{counter}. {content}")
            else:
                if not line.strip():
                    counter = 0  # Reset on empty line
                    in_list = False
                result_lines.append(line)
        
        return '\n'.join(result_lines)
    
    def normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace."""
        # Collapse multiple blank lines to 2
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove trailing whitespace on lines
        lines = [line.rstrip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        # Clean up excessive spaces
        text = re.sub(r'  +', ' ', text)
        
        return text.strip()
    
    # =========================================================================
    # MAIN ENTRY POINT
    # =========================================================================
    
    def canonicalize(
        self,
        text: str,
        context: str = "",
    ) -> tuple[str, list[str]]:
        """Canonicalize text, returning result and list of changes made."""
        changes = []
        result = text
        
        # Remove provider-isms
        result, count = self.remove_provider_isms(result)
        if count:
            changes.append(f"Removed {count} provider-ism(s)")
        
        # Reduce apologies
        result, count = self.reduce_apologies(result)
        if count:
            changes.append(f"Reduced {count} apology(ies)")
        
        # Remove irrelevant disclaimers
        original_len = len(result)
        result = self.remove_irrelevant_disclaimers(result, context)
        if len(result) < original_len:
            changes.append("Removed irrelevant disclaimers")
        
        # Remove filler openings
        result, count = self.remove_filler_openings(result)
        if count:
            changes.append(f"Removed {count} filler opening(s)")
        
        # Remove permission closers
        result, count = self.remove_permission_closers(result)
        if count:
            changes.append(f"Removed {count} permission closer(s)")
        
        # Standardize code blocks
        original = result
        result = self.standardize_code_blocks(result)
        if result != original:
            changes.append("Standardized code blocks")
        
        # Normalize whitespace
        result = self.normalize_whitespace(result)
        
        return result, changes


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def canonicalize_text(text: str, context: str = "") -> str:
    """Convenience function to canonicalize text."""
    canonicalizer = Canonicalizer()
    result, _ = canonicalizer.canonicalize(text, context)
    return result


def remove_provider_isms(text: str) -> str:
    """Convenience function to remove provider-isms."""
    canonicalizer = Canonicalizer()
    result, _ = canonicalizer.remove_provider_isms(text)
    return result


def remove_filler_openings(text: str) -> str:
    """Convenience function to remove filler openings."""
    canonicalizer = Canonicalizer()
    result, _ = canonicalizer.remove_filler_openings(text)
    return result


def remove_permission_closers(text: str) -> str:
    """Convenience function to remove permission closers."""
    canonicalizer = Canonicalizer()
    result, _ = canonicalizer.remove_permission_closers(text)
    return result

