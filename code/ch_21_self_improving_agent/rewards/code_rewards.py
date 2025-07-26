"""
Comprehensive reward functions for code generation.
"""

import ast
import subprocess
import tempfile
import os
import re
from typing import Dict, List, Optional, Tuple
import signal
from contextlib import contextmanager
import time


class CodeRewardFunction:
    """Multi-aspect reward function for generated code."""
    
    def __init__(self, 
                 timeout: int = 5,
                 weights: Optional[Dict[str, float]] = None):
        """
        Initialize code reward function.
        
        Args:
            timeout: Execution timeout in seconds
            weights: Custom weights for different aspects
        """
        self.timeout = timeout
        
        # Default weights
        self.weights = {
            'functionality': 0.3,
            'correctness': 0.4,
            'efficiency': 0.1,
            'readability': 0.1,
            'security': 0.1
        }
        
        if weights:
            self.weights.update(weights)
    
    def evaluate(self, problem: str, code: str, 
                test_cases: Optional[List[Dict]] = None) -> Dict[str, float]:
        """
        Evaluate code on multiple aspects.
        
        Args:
            problem: Problem description
            code: Generated code
            test_cases: Optional test cases
            
        Returns:
            Dictionary with aspect scores and total
        """
        rewards = {
            'functionality': 0.0,
            'correctness': 0.0,
            'efficiency': 0.0,
            'readability': 0.0,
            'security': 0.0,
            'total': 0.0,
            'details': {}
        }
        
        # Functionality: Does it run?
        functionality_result = self._test_functionality(code)
        rewards['functionality'] = functionality_result['score']
        rewards['details']['functionality'] = functionality_result
        
        # Correctness: Does it solve the problem?
        if test_cases:
            correctness_result = self._test_correctness(code, test_cases)
        else:
            # Try to extract test cases from problem
            extracted_tests = self._extract_test_cases(problem)
            if extracted_tests:
                correctness_result = self._test_correctness(code, extracted_tests)
            else:
                correctness_result = {'score': 0.5, 'message': 'No test cases available'}
        
        rewards['correctness'] = correctness_result['score']
        rewards['details']['correctness'] = correctness_result
        
        # Code quality aspects
        rewards['efficiency'] = self._evaluate_efficiency(code)
        rewards['readability'] = self._evaluate_readability(code)
        rewards['security'] = self._evaluate_security(code)
        
        # Calculate total weighted reward
        rewards['total'] = sum(
            rewards[aspect] * self.weights.get(aspect, 0)
            for aspect in ['functionality', 'correctness', 'efficiency', 
                          'readability', 'security']
        )
        
        return rewards
    
    def _test_functionality(self, code: str) -> Dict:
        """Test if code runs without errors."""
        # Check syntax first
        try:
            ast.parse(code)
        except SyntaxError as e:
            return {
                'score': 0.0,
                'error': f"Syntax error: {e}",
                'line': e.lineno
            }
        
        # Try to execute
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', 
                                           delete=False) as f:
                f.write(code)
                f.write('\n\n# Test execution\nif __name__ == "__main__":\n    pass')
                temp_path = f.name
            
            try:
                result = subprocess.run(
                    ['python', temp_path],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout
                )
                
                if result.returncode == 0:
                    return {
                        'score': 1.0,
                        'message': 'Code executes successfully'
                    }
                else:
                    return {
                        'score': 0.0,
                        'error': result.stderr,
                        'message': 'Runtime error'
                    }
                    
            finally:
                os.unlink(temp_path)
                
        except subprocess.TimeoutExpired:
            return {
                'score': 0.0,
                'error': 'Execution timeout',
                'message': f'Code took longer than {self.timeout}s to execute'
            }
        except Exception as e:
            return {
                'score': 0.0,
                'error': str(e),
                'message': 'Unexpected error during execution'
            }
    
    def _test_correctness(self, code: str, test_cases: List[Dict]) -> Dict:
        """Test code correctness with test cases."""
        if not test_cases:
            return {'score': 0.5, 'message': 'No test cases provided'}
        
        # Extract function name
        func_match = re.search(r'def\s+(\w+)\s*\(', code)
        if not func_match:
            # Try to find a class
            class_match = re.search(r'class\s+(\w+)', code)
            if not class_match:
                return {
                    'score': 0.0,
                    'message': 'No function or class definition found'
                }
        
        passed = 0
        total = len(test_cases)
        test_results = []
        
        for i, test_case in enumerate(test_cases):
            try:
                result = self._run_single_test(code, test_case)
                if result['passed']:
                    passed += 1
                test_results.append(result)
            except Exception as e:
                test_results.append({
                    'test_id': i,
                    'passed': False,
                    'error': str(e)
                })
        
        score = passed / total if total > 0 else 0.0
        
        return {
            'score': score,
            'passed': passed,
            'total': total,
            'test_results': test_results,
            'message': f'Passed {passed}/{total} test cases'
        }
    
    def _run_single_test(self, code: str, test_case: Dict) -> Dict:
        """Run a single test case."""
        # Create test wrapper
        test_code = f"""
import sys
import json

{code}

# Test execution
try:
    # Get test input
    test_input = {repr(test_case.get('input', ''))}
    expected_output = {repr(test_case.get('expected', ''))}
    
    # Find function to test
    for name, obj in globals().items():
        if callable(obj) and not name.startswith('_'):
            if 'def' in {repr(code)} and name in {repr(code)}:
                # Call function
                if isinstance(test_input, dict):
                    result = obj(**test_input)
                elif isinstance(test_input, (list, tuple)):
                    result = obj(*test_input)
                else:
                    result = obj(test_input)
                
                # Check result
                passed = str(result) == str(expected_output)
                print(json.dumps({{
                    'passed': passed,
                    'expected': expected_output,
                    'actual': result
                }}))
                sys.exit(0)
    
    # No function found
    print(json.dumps({{'passed': False, 'error': 'No callable function found'}}))
    
except Exception as e:
    print(json.dumps({{'passed': False, 'error': str(e)}}))
"""
        
        # Execute test
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', 
                                        delete=False) as f:
            f.write(test_code)
            temp_path = f.name
        
        try:
            result = subprocess.run(
                ['python', temp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            if result.returncode == 0 and result.stdout:
                import json
                return json.loads(result.stdout)
            else:
                return {
                    'passed': False,
                    'error': result.stderr or 'No output'
                }
                
        except Exception as e:
            return {
                'passed': False,
                'error': str(e)
            }
        finally:
            os.unlink(temp_path)
    
    def _evaluate_efficiency(self, code: str) -> float:
        """Evaluate code efficiency."""
        score = 1.0
        
        # Check for obvious inefficiencies
        inefficiencies = {
            # Pattern: (regex, penalty, description)
            'nested_loops': (r'for.*:\s*\n\s+.*for.*:', 0.2, 'Nested loops detected'),
            'repeated_sort': (r'sorted\(.*\).*sorted\(', 0.3, 'Multiple sorts'),
            'string_concat': (r'\+\s*["\'].*for.*in', 0.2, 'String concatenation in loop'),
            'list_append': (r'\.append\(.*\).*\n.*for', 0.1, 'Append in loop'),
            'repeated_len': (r'len\(.*\).*len\(.*\).*len\(', 0.1, 'Repeated len() calls'),
        }
        
        for name, (pattern, penalty, desc) in inefficiencies.items():
            if re.search(pattern, code, re.MULTILINE):
                score -= penalty
        
        # Check for good practices
        good_practices = {
            'list_comp': (r'\[.*for.*in.*\]', 0.1, 'Uses list comprehension'),
            'generator': (r'\(.*for.*in.*\)', 0.1, 'Uses generator expression'),
            'join': (r'\.join\(', 0.05, 'Uses string join'),
            'set_lookup': (r'set\(|{.*}', 0.05, 'Uses set for lookups'),
            'dict_comp': (r'{.*:.*for.*in.*}', 0.1, 'Uses dict comprehension'),
        }
        
        for name, (pattern, bonus, desc) in good_practices.items():
            if re.search(pattern, code):
                score += bonus
        
        return max(0.0, min(1.0, score))
    
    def _evaluate_readability(self, code: str) -> float:
        """Evaluate code readability."""
        score = 0.5  # Start neutral
        
        lines = code.split('\n')
        non_empty_lines = [l for l in lines if l.strip()]
        
        # Docstrings
        if '"""' in code or "'''" in code:
            score += 0.2
        
        # Comments
        comment_lines = sum(1 for l in lines if l.strip().startswith('#'))
        if non_empty_lines:
            comment_ratio = comment_lines / len(non_empty_lines)
            score += min(0.2, comment_ratio * 2)
        
        # Function/variable naming
        # Good names: descriptive, not single letters
        var_names = re.findall(r'(\w+)\s*=', code)
        func_names = re.findall(r'def\s+(\w+)', code)
        all_names = var_names + func_names
        
        if all_names:
            good_names = sum(1 for name in all_names 
                           if len(name) > 2 and name.lower() != name.upper())
            score += (good_names / len(all_names)) * 0.2
        
        # Line length
        long_lines = sum(1 for l in lines if len(l) > 80)
        if lines:
            long_line_ratio = long_lines / len(lines)
            score -= min(0.2, long_line_ratio)
        
        # Proper indentation (check consistency)
        indent_issues = self._check_indentation(code)
        if indent_issues == 0:
            score += 0.1
        else:
            score -= min(0.2, indent_issues * 0.05)
        
        return max(0.0, min(1.0, score))
    
    def _evaluate_security(self, code: str) -> float:
        """Evaluate code security."""
        score = 1.0
        
        # Critical security issues
        critical_issues = {
            'eval': (r'\beval\s*\(', 0.5),
            'exec': (r'\bexec\s*\(', 0.5),
            'compile': (r'\bcompile\s*\(', 0.4),
            '__import__': (r'__import__\s*\(', 0.4),
            'os_system': (r'os\.system\s*\(', 0.4),
            'subprocess_shell': (r'subprocess.*shell\s*=\s*True', 0.4),
            'pickle_load': (r'pickle\.load', 0.3),
        }
        
        for name, (pattern, penalty) in critical_issues.items():
            if re.search(pattern, code):
                score -= penalty
        
        # Input validation
        if any(inp in code for inp in ['input(', 'sys.argv', 'request.']):
            # Check for validation
            validation_patterns = ['validate', 'check', 'verify', 'sanitize', 
                                 'assert', 'if ', 'try:']
            has_validation = any(v in code for v in validation_patterns)
            if not has_validation:
                score -= 0.2
        
        # SQL injection risk
        sql_keywords = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'DROP']
        if any(sql in code for sql in sql_keywords):
            # Check for parameterized queries
            if '?' not in code and '%s' not in code:
                if '+' in code or 'format(' in code or 'f"' in code:
                    score -= 0.3
        
        # Path traversal
        if 'open(' in code or 'os.path' in code:
            if '../' in code or '..\\' in code:
                score -= 0.3
            elif not any(check in code for check in ['abspath', 'realpath', 'normpath']):
                score -= 0.1
        
        return max(0.0, score)
    
    def _extract_test_cases(self, problem: str) -> List[Dict]:
        """Try to extract test cases from problem description."""
        test_cases = []
        
        # Look for example format: "Input: X, Output: Y"
        input_output_pattern = r'[Ii]nput:\s*([^\n,]+).*?[Oo]utput:\s*([^\n]+)'
        matches = re.findall(input_output_pattern, problem, re.DOTALL)
        
        for inp, out in matches:
            try:
                # Try to parse as Python literals
                test_cases.append({
                    'input': ast.literal_eval(inp.strip()),
                    'expected': ast.literal_eval(out.strip())
                })
            except:
                # Fall back to string
                test_cases.append({
                    'input': inp.strip(),
                    'expected': out.strip()
                })
        
        # Look for function signature and examples
        func_pattern = r'def\s+(\w+)\(([^)]*)\).*?[Ee]xample'
        func_match = re.search(func_pattern, problem, re.DOTALL)
        
        if func_match and not test_cases:
            # Try to find examples after function definition
            example_pattern = r'(\w+)\(([^)]+)\)\s*(?:=|->|returns?)\s*([^\n]+)'
            example_matches = re.findall(example_pattern, problem)
            
            func_name = func_match.group(1)
            for ex_func, ex_input, ex_output in example_matches:
                if ex_func == func_name:
                    try:
                        test_cases.append({
                            'input': ast.literal_eval(ex_input),
                            'expected': ast.literal_eval(ex_output)
                        })
                    except:
                        pass
        
        return test_cases
    
    def _check_indentation(self, code: str) -> int:
        """Check for indentation issues."""
        issues = 0
        lines = code.split('\n')
        
        # Track indentation levels
        indent_stack = [0]
        
        for line in lines:
            if not line.strip():
                continue
                
            # Count leading spaces
            indent = len(line) - len(line.lstrip())
            
            # Check if it's a continuation of previous line
            if line.strip().startswith((')', ']', '}')):
                continue
                
            # Check for inconsistent indentation
            if indent not in indent_stack:
                if indent > indent_stack[-1]:
                    # Deeper indentation
                    if indent - indent_stack[-1] not in [2, 4]:
                        issues += 1
                    indent_stack.append(indent)
                else:
                    # Dedent - find matching level
                    while indent_stack and indent < indent_stack[-1]:
                        indent_stack.pop()
                    if not indent_stack or indent != indent_stack[-1]:
                        issues += 1
        
        return issues