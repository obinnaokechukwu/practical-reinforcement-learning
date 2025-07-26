"""
Constitutional principles for AI agents.
"""

from typing import List, Dict, Optional
import re


class ConstitutionalPrinciples:
    """Manages constitutional principles for different domains."""
    
    # Pre-defined constitutions for common domains
    CODING_CONSTITUTION = [
        "Write secure code that avoids common vulnerabilities",
        "Include appropriate error handling and input validation",
        "Follow established coding conventions and best practices",
        "Write efficient algorithms with reasonable time complexity",
        "Include clear comments explaining complex logic",
        "Use descriptive variable and function names",
        "Add type hints to function signatures",
        "Write modular, reusable code",
        "Avoid global variables and side effects",
        "Include docstrings for all functions"
    ]
    
    MATH_CONSTITUTION = [
        "Show all work and reasoning steps clearly",
        "Use correct mathematical notation and terminology",
        "Check answers for reasonableness and correctness",
        "State assumptions explicitly",
        "Provide units for physical quantities",
        "Explain the approach before diving into calculations",
        "Verify solutions by substitution when possible",
        "Acknowledge limitations or special cases",
        "Use appropriate precision in numerical answers",
        "Reference relevant theorems or formulas"
    ]
    
    EXPLANATION_CONSTITUTION = [
        "Start with a high-level overview",
        "Use clear, simple language",
        "Provide concrete examples",
        "Define technical terms when first used",
        "Structure explanations logically",
        "Use analogies to clarify complex concepts",
        "Anticipate common misconceptions",
        "Include visual descriptions when helpful",
        "Summarize key points at the end",
        "Provide references for further learning"
    ]
    
    ANALYSIS_CONSTITUTION = [
        "Consider multiple perspectives",
        "Support claims with evidence",
        "Acknowledge limitations and uncertainties",
        "Use structured reasoning",
        "Identify assumptions explicitly",
        "Consider edge cases and exceptions",
        "Provide balanced assessments",
        "Distinguish correlation from causation",
        "Quantify claims when possible",
        "Draw clear, justified conclusions"
    ]
    
    def __init__(self, domain: str = 'coding', 
                 custom_principles: Optional[List[str]] = None):
        """
        Initialize constitutional principles.
        
        Args:
            domain: Domain type ('coding', 'math', 'explanation', 'analysis')
            custom_principles: Optional custom principles to use instead
        """
        self.domain = domain
        
        if custom_principles:
            self.principles = custom_principles
        else:
            self.principles = self._get_default_principles(domain)
        
        self.principle_weights = {p: 1.0 for p in self.principles}
    
    def _get_default_principles(self, domain: str) -> List[str]:
        """Get default principles for domain."""
        domain_map = {
            'coding': self.CODING_CONSTITUTION,
            'math': self.MATH_CONSTITUTION,
            'explanation': self.EXPLANATION_CONSTITUTION,
            'analysis': self.ANALYSIS_CONSTITUTION
        }
        
        return domain_map.get(domain, self.CODING_CONSTITUTION)
    
    def add_principle(self, principle: str, weight: float = 1.0):
        """Add a new principle with optional weight."""
        if principle not in self.principles:
            self.principles.append(principle)
            self.principle_weights[principle] = weight
    
    def remove_principle(self, principle: str):
        """Remove a principle."""
        if principle in self.principles:
            self.principles.remove(principle)
            del self.principle_weights[principle]
    
    def update_weight(self, principle: str, weight: float):
        """Update the weight of a principle."""
        if principle in self.principle_weights:
            self.principle_weights[principle] = weight
    
    def evaluate_compliance(self, text: str) -> Dict[str, float]:
        """
        Evaluate how well text complies with principles.
        
        Args:
            text: Text to evaluate
            
        Returns:
            Dictionary mapping principles to compliance scores (0-1)
        """
        compliance_scores = {}
        
        for principle in self.principles:
            score = self._evaluate_single_principle(text, principle)
            compliance_scores[principle] = score
        
        return compliance_scores
    
    def _evaluate_single_principle(self, text: str, principle: str) -> float:
        """Evaluate compliance with a single principle."""
        principle_lower = principle.lower()
        text_lower = text.lower()
        
        # Domain-specific evaluation
        if self.domain == 'coding':
            return self._evaluate_coding_principle(text, principle)
        elif self.domain == 'math':
            return self._evaluate_math_principle(text, principle)
        elif self.domain == 'explanation':
            return self._evaluate_explanation_principle(text, principle)
        elif self.domain == 'analysis':
            return self._evaluate_analysis_principle(text, principle)
        else:
            # Generic evaluation
            return self._evaluate_generic_principle(text, principle)
    
    def _evaluate_coding_principle(self, code: str, principle: str) -> float:
        """Evaluate coding-specific principles."""
        principle_lower = principle.lower()
        
        if 'secure' in principle_lower:
            # Check for security issues
            insecure_patterns = ['eval(', 'exec(', '__import__', 'os.system']
            violations = sum(1 for p in insecure_patterns if p in code)
            return max(0, 1 - violations * 0.25)
        
        elif 'error handling' in principle_lower:
            # Check for try/except blocks
            has_try = 'try:' in code
            has_except = 'except' in code
            return 1.0 if (has_try and has_except) else 0.3
        
        elif 'comment' in principle_lower:
            # Check comment density
            lines = code.split('\n')
            comment_lines = sum(1 for l in lines if '#' in l or '"""' in l)
            return min(1.0, comment_lines / max(len(lines), 1) * 5)
        
        elif 'efficient' in principle_lower:
            # Check for inefficient patterns
            inefficient = ['nested loops', 'O(n²)', 'O(n^2)']
            has_inefficient = any(p in code.lower() for p in inefficient)
            return 0.5 if has_inefficient else 1.0
        
        elif 'type hint' in principle_lower:
            # Check for type annotations
            has_type_hints = '->' in code or ': ' in code
            return 1.0 if has_type_hints else 0.2
        
        elif 'docstring' in principle_lower:
            # Check for docstrings
            return 1.0 if '"""' in code else 0.2
        
        elif 'variable' in principle_lower and 'name' in principle_lower:
            # Check variable naming
            var_pattern = r'(\w+)\s*='
            variables = re.findall(var_pattern, code)
            if variables:
                good_names = sum(1 for v in variables if len(v) > 2)
                return good_names / len(variables)
            return 0.5
        
        else:
            return self._evaluate_generic_principle(code, principle)
    
    def _evaluate_math_principle(self, text: str, principle: str) -> float:
        """Evaluate math-specific principles."""
        principle_lower = principle.lower()
        
        if 'show' in principle_lower and 'work' in principle_lower:
            # Check for step-by-step work
            step_indicators = ['step', 'first', 'then', 'next', 'finally']
            steps_shown = sum(1 for s in step_indicators if s in text.lower())
            return min(1.0, steps_shown / 3)
        
        elif 'notation' in principle_lower:
            # Check for mathematical notation
            math_symbols = ['=', '+', '-', '*', '/', '^', '∫', '∑', 'π']
            has_notation = sum(1 for s in math_symbols if s in text)
            return min(1.0, has_notation / 3)
        
        elif 'check' in principle_lower or 'verify' in principle_lower:
            # Check for verification
            verify_words = ['check', 'verify', 'confirm', 'substitute']
            has_verification = any(w in text.lower() for w in verify_words)
            return 1.0 if has_verification else 0.3
        
        else:
            return self._evaluate_generic_principle(text, principle)
    
    def _evaluate_explanation_principle(self, text: str, principle: str) -> float:
        """Evaluate explanation-specific principles."""
        principle_lower = principle.lower()
        
        if 'overview' in principle_lower:
            # Check for overview section
            overview_words = ['overview', 'summary', 'introduction', 'briefly']
            has_overview = any(w in text.lower()[:200] for w in overview_words)
            return 1.0 if has_overview else 0.3
        
        elif 'example' in principle_lower:
            # Check for examples
            example_words = ['example', 'for instance', 'such as', 'e.g.']
            example_count = sum(1 for w in example_words if w in text.lower())
            return min(1.0, example_count / 2)
        
        elif 'simple language' in principle_lower:
            # Check readability (simple heuristic)
            words = text.split()
            avg_word_length = sum(len(w) for w in words) / max(len(words), 1)
            return 1.0 if avg_word_length < 7 else 0.5
        
        else:
            return self._evaluate_generic_principle(text, principle)
    
    def _evaluate_analysis_principle(self, text: str, principle: str) -> float:
        """Evaluate analysis-specific principles."""
        principle_lower = principle.lower()
        
        if 'perspective' in principle_lower:
            # Check for multiple viewpoints
            perspective_words = ['however', 'alternatively', 'on the other hand', 
                               'another view', 'conversely']
            perspectives = sum(1 for w in perspective_words if w in text.lower())
            return min(1.0, perspectives / 2)
        
        elif 'evidence' in principle_lower:
            # Check for evidence citations
            evidence_words = ['study', 'research', 'data', 'according to', 
                            'source', 'evidence']
            has_evidence = sum(1 for w in evidence_words if w in text.lower())
            return min(1.0, has_evidence / 3)
        
        elif 'limitation' in principle_lower:
            # Check for acknowledged limitations
            limitation_words = ['limitation', 'caveat', 'however', 'but', 
                              'should note', 'important to']
            has_limitations = any(w in text.lower() for w in limitation_words)
            return 1.0 if has_limitations else 0.3
        
        else:
            return self._evaluate_generic_principle(text, principle)
    
    def _evaluate_generic_principle(self, text: str, principle: str) -> float:
        """Generic principle evaluation."""
        # Extract key words from principle
        keywords = re.findall(r'\b\w{4,}\b', principle.lower())
        
        if not keywords:
            return 0.5
        
        # Check how many keywords appear in text
        text_lower = text.lower()
        matches = sum(1 for kw in keywords if kw in text_lower)
        
        return min(1.0, matches / max(len(keywords), 1))
    
    def get_weighted_compliance_score(self, text: str) -> float:
        """
        Get overall weighted compliance score.
        
        Args:
            text: Text to evaluate
            
        Returns:
            Weighted compliance score (0-1)
        """
        compliance_scores = self.evaluate_compliance(text)
        
        total_weight = sum(self.principle_weights.values())
        if total_weight == 0:
            return 0.0
        
        weighted_sum = sum(
            score * self.principle_weights[principle]
            for principle, score in compliance_scores.items()
        )
        
        return weighted_sum / total_weight
    
    def get_violations(self, text: str, threshold: float = 0.5) -> List[str]:
        """
        Get list of principles that are violated.
        
        Args:
            text: Text to evaluate
            threshold: Score below which principle is considered violated
            
        Returns:
            List of violated principles
        """
        compliance_scores = self.evaluate_compliance(text)
        
        violations = [
            principle for principle, score in compliance_scores.items()
            if score < threshold
        ]
        
        return violations
    
    def format_feedback(self, text: str) -> str:
        """
        Generate formatted feedback on constitutional compliance.
        
        Args:
            text: Text to evaluate
            
        Returns:
            Formatted feedback string
        """
        compliance_scores = self.evaluate_compliance(text)
        violations = self.get_violations(text)
        overall_score = self.get_weighted_compliance_score(text)
        
        feedback = f"Constitutional Compliance Report\n"
        feedback += f"{'=' * 40}\n"
        feedback += f"Overall Score: {overall_score:.2%}\n\n"
        
        if violations:
            feedback += "Principles Needing Improvement:\n"
            for principle in violations:
                score = compliance_scores[principle]
                feedback += f"- {principle} (Score: {score:.2%})\n"
            feedback += "\n"
        
        feedback += "Detailed Scores:\n"
        for principle, score in sorted(compliance_scores.items(), 
                                     key=lambda x: x[1], reverse=True):
            status = "✓" if score >= 0.7 else "⚠" if score >= 0.5 else "✗"
            feedback += f"{status} {principle}: {score:.2%}\n"
        
        return feedback