"""
Verifiable reward functions for objective task evaluation.
"""

import re
import ast
import json
from typing import Dict, List, Optional, Any, Callable
import subprocess
import tempfile
import os


class VerifiableReward:
    """Base class for verifiable reward functions."""
    
    def __init__(self, task_type: str):
        self.task_type = task_type
        self.evaluation_cache = {}
    
    def evaluate(self, problem: str, solution: str) -> float:
        """Evaluate solution objectively."""
        raise NotImplementedError
    
    def extract_expected_output(self, problem: str) -> Any:
        """Extract expected output from problem description."""
        # Common patterns for expected outputs
        patterns = [
            r'[Ee]xpected:?\s*([^\n]+)',
            r'[Oo]utput:?\s*([^\n]+)',
            r'[Rr]eturn:?\s*([^\n]+)',
            r'[Rr]esult:?\s*([^\n]+)',
            r'[Aa]nswer:?\s*([^\n]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, problem)
            if match:
                expected = match.group(1).strip()
                # Try to parse as Python literal
                try:
                    return ast.literal_eval(expected)
                except:
                    return expected
        
        return None


class MathVerifiableReward(VerifiableReward):
    """Verifiable reward for mathematical problems."""
    
    def __init__(self):
        super().__init__('math')
        self.tolerance = 1e-6
    
    def evaluate(self, problem: str, solution: str) -> float:
        """Evaluate mathematical solution."""
        # Extract expected answer
        expected = self.extract_expected_output(problem)
        if expected is None:
            return 0.5  # Can't verify
        
        # Extract answer from solution
        predicted = self._extract_answer(solution)
        if predicted is None:
            return 0.0
        
        # Check correctness
        correct = self._check_mathematical_equivalence(predicted, expected)
        
        # Additional scoring based on solution quality
        score = 1.0 if correct else 0.0
        
        # Bonus for showing work
        if self._shows_work(solution):
            score += 0.1
        
        # Bonus for clear formatting
        if self._well_formatted(solution):
            score += 0.05
        
        return min(1.0, score)
    
    def _extract_answer(self, solution: str) -> Optional[Any]:
        """Extract mathematical answer from solution."""
        # Look for final answer patterns
        answer_patterns = [
            r'[Aa]nswer:?\s*([^\n]+)',
            r'[Rr]esult:?\s*([^\n]+)',
            r'=\s*([^\n]+)$',
            r'[Tt]herefore,?\s*([^\n]+)',
            r'[Ss]o,?\s*([^\n]+)',
        ]
        
        for pattern in answer_patterns:
            matches = re.findall(pattern, solution, re.MULTILINE)
            if matches:
                # Take last match
                answer_text = matches[-1].strip()
                # Clean up
                answer_text = answer_text.rstrip('.,;')
                
                try:
                    # Try to evaluate as number
                    return self._parse_number(answer_text)
                except:
                    return answer_text
        
        return None
    
    def _parse_number(self, text: str) -> float:
        """Parse number from text."""
        # Remove units and extra text
        text = re.sub(r'[a-zA-Z\s]+$', '', text).strip()
        
        # Handle fractions
        if '/' in text:
            parts = text.split('/')
            if len(parts) == 2:
                return float(parts[0]) / float(parts[1])
        
        # Handle scientific notation
        text = text.replace('×10^', 'e').replace('x10^', 'e')
        
        return float(text)
    
    def _check_mathematical_equivalence(self, predicted: Any, expected: Any) -> bool:
        """Check if two mathematical answers are equivalent."""
        # String comparison
        if str(predicted) == str(expected):
            return True
        
        # Numerical comparison
        try:
            pred_num = float(predicted)
            exp_num = float(expected)
            return abs(pred_num - exp_num) < self.tolerance
        except:
            pass
        
        # Symbolic comparison would go here
        # (using sympy or similar)
        
        return False
    
    def _shows_work(self, solution: str) -> bool:
        """Check if solution shows work."""
        work_indicators = [
            'step', 'first', 'then', 'next', 'finally',
            'substitute', 'simplify', 'solve', 'factor',
            'therefore', 'thus', 'so'
        ]
        
        indicator_count = sum(1 for ind in work_indicators 
                            if ind in solution.lower())
        
        return indicator_count >= 3
    
    def _well_formatted(self, solution: str) -> bool:
        """Check if solution is well formatted."""
        # Check for math notation
        has_equations = any(sym in solution for sym in ['=', '+', '-', '*', '/'])
        
        # Check for structure
        has_lines = len(solution.split('\n')) > 3
        
        return has_equations and has_lines


class LogicVerifiableReward(VerifiableReward):
    """Verifiable reward for logical reasoning problems."""
    
    def __init__(self):
        super().__init__('logic')
    
    def evaluate(self, problem: str, solution: str) -> float:
        """Evaluate logical reasoning solution."""
        # Extract premises and conclusion
        premises = self._extract_premises(problem)
        expected_conclusion = self.extract_expected_output(problem)
        
        if not premises or expected_conclusion is None:
            return 0.5  # Can't fully verify
        
        # Extract reasoning and conclusion from solution
        reasoning_steps = self._extract_reasoning_steps(solution)
        predicted_conclusion = self._extract_conclusion(solution)
        
        score = 0.0
        
        # Check conclusion correctness
        if predicted_conclusion and self._conclusions_match(
            predicted_conclusion, expected_conclusion
        ):
            score += 0.6
        
        # Check reasoning validity
        if self._reasoning_is_valid(premises, reasoning_steps):
            score += 0.3
        
        # Check reasoning completeness
        if self._reasoning_is_complete(reasoning_steps):
            score += 0.1
        
        return score
    
    def _extract_premises(self, problem: str) -> List[str]:
        """Extract logical premises from problem."""
        premises = []
        
        # Look for premise indicators
        premise_patterns = [
            r'[Gg]iven:?\s*([^\n]+)',
            r'[Pp]remise:?\s*([^\n]+)',
            r'[Aa]ssume:?\s*([^\n]+)',
            r'[Ff]act:?\s*([^\n]+)',
        ]
        
        for pattern in premise_patterns:
            matches = re.findall(pattern, problem)
            premises.extend(matches)
        
        # Look for numbered premises
        numbered_pattern = r'\d+\.\s*([^\n]+)'
        numbered_matches = re.findall(numbered_pattern, problem)
        premises.extend(numbered_matches)
        
        return premises
    
    def _extract_reasoning_steps(self, solution: str) -> List[str]:
        """Extract reasoning steps from solution."""
        steps = []
        
        # Split by common reasoning markers
        markers = [
            'therefore', 'thus', 'hence', 'so',
            'it follows that', 'we can conclude',
            'this means', 'this implies'
        ]
        
        # Split solution into sentences
        sentences = re.split(r'[.!?]+', solution)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if any(marker in sentence.lower() for marker in markers):
                steps.append(sentence)
            elif re.search(r'^\d+\.', sentence):  # Numbered steps
                steps.append(sentence)
        
        return steps
    
    def _extract_conclusion(self, solution: str) -> Optional[str]:
        """Extract conclusion from solution."""
        conclusion_patterns = [
            r'[Cc]onclusion:?\s*([^\n]+)',
            r'[Tt]herefore,?\s*([^\n]+)',
            r'[Tt]hus,?\s*([^\n]+)',
            r'[Ii]n conclusion,?\s*([^\n]+)',
            r'[Ff]inally,?\s*([^\n]+)',
        ]
        
        for pattern in conclusion_patterns:
            matches = re.findall(pattern, solution)
            if matches:
                return matches[-1].strip()
        
        # Take last sentence as conclusion
        sentences = re.split(r'[.!?]+', solution)
        if sentences:
            return sentences[-1].strip()
        
        return None
    
    def _conclusions_match(self, predicted: str, expected: str) -> bool:
        """Check if conclusions match."""
        # Normalize
        pred_norm = predicted.lower().strip()
        exp_norm = expected.lower().strip()
        
        # Direct match
        if pred_norm == exp_norm:
            return True
        
        # Check if key terms match
        pred_terms = set(re.findall(r'\b\w+\b', pred_norm))
        exp_terms = set(re.findall(r'\b\w+\b', exp_norm))
        
        # Calculate similarity
        intersection = pred_terms & exp_terms
        union = pred_terms | exp_terms
        
        if union:
            similarity = len(intersection) / len(union)
            return similarity > 0.7
        
        return False
    
    def _reasoning_is_valid(self, premises: List[str], steps: List[str]) -> bool:
        """Check if reasoning follows from premises."""
        # Simple heuristic: check if steps reference premises
        premise_terms = set()
        for premise in premises:
            terms = re.findall(r'\b\w+\b', premise.lower())
            premise_terms.update(terms)
        
        steps_reference_premises = 0
        for step in steps:
            step_terms = set(re.findall(r'\b\w+\b', step.lower()))
            if step_terms & premise_terms:
                steps_reference_premises += 1
        
        return steps_reference_premises >= len(steps) * 0.5
    
    def _reasoning_is_complete(self, steps: List[str]) -> bool:
        """Check if reasoning is complete."""
        # Heuristic: should have multiple steps
        return len(steps) >= 2


class FunctionVerifiableReward(VerifiableReward):
    """Verifiable reward for function implementation problems."""
    
    def __init__(self, timeout: int = 5):
        super().__init__('function')
        self.timeout = timeout
    
    def evaluate(self, problem: str, solution: str) -> float:
        """Evaluate function implementation."""
        # Extract test cases
        test_cases = self._extract_test_cases(problem)
        if not test_cases:
            return 0.5  # Can't verify
        
        # Extract function from solution
        function_code = self._extract_function(solution)
        if not function_code:
            return 0.0
        
        # Run tests
        passed = 0
        total = len(test_cases)
        
        for test_case in test_cases:
            if self._run_test(function_code, test_case):
                passed += 1
        
        base_score = passed / total if total > 0 else 0.0
        
        # Additional quality checks
        quality_score = 0.0
        
        if self._has_docstring(function_code):
            quality_score += 0.05
        
        if self._has_type_hints(function_code):
            quality_score += 0.05
        
        if self._handles_edge_cases(function_code):
            quality_score += 0.05
        
        return min(1.0, base_score + quality_score)
    
    def _extract_test_cases(self, problem: str) -> List[Dict]:
        """Extract test cases from problem."""
        test_cases = []
        
        # Pattern: function_name(input) = output
        pattern = r'(\w+)\(([^)]+)\)\s*=\s*([^\n]+)'
        matches = re.findall(pattern, problem)
        
        for func_name, input_str, output_str in matches:
            try:
                test_cases.append({
                    'function': func_name,
                    'input': ast.literal_eval(input_str),
                    'expected': ast.literal_eval(output_str)
                })
            except:
                # Try without literal eval
                test_cases.append({
                    'function': func_name,
                    'input': input_str,
                    'expected': output_str
                })
        
        # Also look for assert statements
        assert_pattern = r'assert\s+(\w+)\(([^)]+)\)\s*==\s*([^\n]+)'
        assert_matches = re.findall(assert_pattern, problem)
        
        for func_name, input_str, output_str in assert_matches:
            try:
                test_cases.append({
                    'function': func_name,
                    'input': ast.literal_eval(input_str),
                    'expected': ast.literal_eval(output_str)
                })
            except:
                pass
        
        return test_cases
    
    def _extract_function(self, solution: str) -> Optional[str]:
        """Extract function definition from solution."""
        # Find function definition
        func_pattern = r'(def\s+\w+\s*\([^)]*\):.*?)(?=\ndef|\Z)'
        match = re.search(func_pattern, solution, re.DOTALL)
        
        if match:
            return match.group(1)
        
        return None
    
    def _run_test(self, function_code: str, test_case: Dict) -> bool:
        """Run a single test case."""
        test_code = f"""
{function_code}

# Run test
try:
    result = {test_case['function']}({repr(test_case['input'])})
    expected = {repr(test_case['expected'])}
    print("PASS" if result == expected else "FAIL")
except Exception as e:
    print("ERROR")
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_code)
            temp_path = f.name
        
        try:
            result = subprocess.run(
                ['python', temp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            return "PASS" in result.stdout
            
        except:
            return False
        finally:
            os.unlink(temp_path)
    
    def _has_docstring(self, code: str) -> bool:
        """Check if function has docstring."""
        return '"""' in code or "'''" in code
    
    def _has_type_hints(self, code: str) -> bool:
        """Check if function has type hints."""
        return '->' in code or ': ' in re.findall(r'def.*\(.*\)', code)[0]
    
    def _handles_edge_cases(self, code: str) -> bool:
        """Check if function handles edge cases."""
        edge_case_indicators = [
            'if not', 'if len', 'is None', 'raise',
            'try:', 'except:', '== 0', '< 0'
        ]
        
        return any(indicator in code for indicator in edge_case_indicators)