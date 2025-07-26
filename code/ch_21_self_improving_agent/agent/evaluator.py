"""
Self-evaluation system for code generation.
"""

import ast
import subprocess
import tempfile
import os
import re
from typing import Dict, List, Optional, Tuple
import signal
import json
from contextlib import contextmanager
import time


class SelfEvaluator:
    """Evaluates generated code against multiple criteria."""
    
    def __init__(self, timeout: int = 5):
        self.timeout = timeout
        self.evaluation_history = []
    
    def evaluate_code(self, problem: str, code: str, 
                     test_cases: Optional[List[Dict]] = None) -> Dict:
        """
        Comprehensive code evaluation.
        
        Args:
            problem: Problem description
            code: Generated code to evaluate
            test_cases: Optional test cases for verification
            
        Returns:
            Dictionary with evaluation results
        """
        evaluation = {
            'syntax_valid': False,
            'executes': False,
            'passes_tests': False,
            'test_results': [],
            'efficiency_score': 0.0,
            'readability_score': 0.0,
            'security_score': 0.0,
            'overall_score': 0.0,
            'issues': [],
            'suggestions': []
        }
        
        # Check syntax
        syntax_result = self._check_syntax(code)
        evaluation['syntax_valid'] = syntax_result['valid']
        if not syntax_result['valid']:
            evaluation['issues'].append(f"Syntax error: {syntax_result['error']}")
            return evaluation
        
        # Check if code executes
        execution_result = self._test_execution(code)
        evaluation['executes'] = execution_result['success']
        if not execution_result['success']:
            evaluation['issues'].append(f"Execution error: {execution_result['error']}")
        
        # Run test cases if provided
        if test_cases and evaluation['executes']:
            test_results = self._run_test_cases(code, test_cases)
            evaluation['test_results'] = test_results
            evaluation['passes_tests'] = all(r['passed'] for r in test_results)
        
        # Evaluate code quality aspects
        evaluation['efficiency_score'] = self._evaluate_efficiency(code)
        evaluation['readability_score'] = self._evaluate_readability(code)
        evaluation['security_score'] = self._evaluate_security(code)
        
        # Add specific feedback
        evaluation['issues'].extend(self._identify_issues(code))
        evaluation['suggestions'].extend(self._generate_suggestions(code))
        
        # Calculate overall score
        evaluation['overall_score'] = self._calculate_overall_score(evaluation)
        
        # Store in history
        self.evaluation_history.append({
            'timestamp': time.time(),
            'problem': problem[:100],  # First 100 chars
            'evaluation': evaluation
        })
        
        return evaluation
    
    def _check_syntax(self, code: str) -> Dict:
        """Check if code has valid Python syntax."""
        try:
            ast.parse(code)
            return {'valid': True, 'error': None}
        except SyntaxError as e:
            return {'valid': False, 'error': str(e)}
    
    def _test_execution(self, code: str) -> Dict:
        """Test if code executes without errors."""
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', 
                                           delete=False) as f:
                f.write(code)
                f.write('\n\n# Execution test\nprint("SUCCESS")')
                temp_path = f.name
            
            try:
                # Execute with timeout
                result = subprocess.run(
                    ['python', temp_path],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout
                )
                
                success = result.returncode == 0 and 'SUCCESS' in result.stdout
                error = result.stderr if result.stderr else None
                
                return {'success': success, 'error': error}
                
            finally:
                os.unlink(temp_path)
                
        except subprocess.TimeoutExpired:
            return {'success': False, 'error': 'Execution timeout'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _run_test_cases(self, code: str, test_cases: List[Dict]) -> List[Dict]:
        """Run provided test cases against code."""
        results = []
        
        for i, test_case in enumerate(test_cases):
            try:
                result = self._run_single_test(code, test_case)
                results.append({
                    'test_id': i,
                    'input': test_case.get('input', ''),
                    'expected': test_case.get('expected', ''),
                    'actual': result.get('output', ''),
                    'passed': result.get('passed', False),
                    'error': result.get('error')
                })
            except Exception as e:
                results.append({
                    'test_id': i,
                    'input': test_case.get('input', ''),
                    'expected': test_case.get('expected', ''),
                    'actual': None,
                    'passed': False,
                    'error': str(e)
                })
        
        return results
    
    def _run_single_test(self, code: str, test_case: Dict) -> Dict:
        """Run a single test case."""
        # Extract function name from code
        func_match = re.search(r'def\s+(\w+)\s*\(', code)
        if not func_match:
            return {'passed': False, 'error': 'No function found'}
        
        func_name = func_match.group(1)
        
        # Create test code
        test_code = f"""
{code}

# Test execution
try:
    result = {func_name}({test_case['input']})
    print(f"RESULT:{{result}}")
except Exception as e:
    print(f"ERROR:{{e}}")
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
            
            # Parse output
            if 'RESULT:' in result.stdout:
                output = result.stdout.split('RESULT:')[1].strip()
                
                # Compare with expected
                expected = str(test_case['expected'])
                passed = output == expected
                
                return {
                    'output': output,
                    'passed': passed,
                    'error': None
                }
            elif 'ERROR:' in result.stdout:
                error = result.stdout.split('ERROR:')[1].strip()
                return {
                    'output': None,
                    'passed': False,
                    'error': error
                }
            else:
                return {
                    'output': None,
                    'passed': False,
                    'error': 'No output captured'
                }
                
        finally:
            os.unlink(temp_path)
    
    def _evaluate_efficiency(self, code: str) -> float:
        """Evaluate algorithmic efficiency (0-1 scale)."""
        efficiency_score = 1.0
        
        # Check for nested loops
        nested_loops = len(re.findall(r'for.*:\s*\n\s+.*for.*:', code))
        if nested_loops > 0:
            efficiency_score -= 0.2 * min(nested_loops, 2)
        
        # Check for inefficient patterns
        inefficient_patterns = [
            (r'\.append\(.*\).*for', 0.1),  # Append in loop
            (r'in\s+list\(', 0.1),  # Unnecessary list conversion
            (r'sorted\(.*\).*sorted\(', 0.2),  # Multiple sorts
            (r'len\(.*\)\s*==\s*0', 0.05),  # Should use 'not'
        ]
        
        for pattern, penalty in inefficient_patterns:
            if re.search(pattern, code):
                efficiency_score -= penalty
        
        # Positive patterns
        efficient_patterns = [
            (r'\[.*for.*in.*\]', 0.1),  # List comprehension
            (r'\.join\(', 0.05),  # String join instead of concatenation
            (r'set\(', 0.05),  # Using sets for membership
            (r'collections\.', 0.1),  # Using collections module
        ]
        
        for pattern, bonus in efficient_patterns:
            if re.search(pattern, code):
                efficiency_score += bonus
        
        return max(0.0, min(1.0, efficiency_score))
    
    def _evaluate_readability(self, code: str) -> float:
        """Evaluate code readability (0-1 scale)."""
        readability_score = 0.5  # Start at neutral
        
        lines = code.split('\n')
        non_empty_lines = [l for l in lines if l.strip()]
        
        # Check for docstrings
        if '"""' in code or "'''" in code:
            readability_score += 0.2
        
        # Check for comments
        comment_lines = sum(1 for l in lines if l.strip().startswith('#'))
        comment_ratio = comment_lines / max(len(non_empty_lines), 1)
        if comment_ratio > 0.1:
            readability_score += 0.1
        elif comment_ratio > 0.05:
            readability_score += 0.05
        
        # Check for meaningful variable names
        var_pattern = r'(\w+)\s*='
        variables = re.findall(var_pattern, code)
        
        good_var_names = sum(1 for v in variables if len(v) > 2 and v != 'res')
        if variables:
            good_name_ratio = good_var_names / len(variables)
            readability_score += good_name_ratio * 0.2
        
        # Check for proper function structure
        if 'def ' in code:
            # Has functions
            readability_score += 0.1
            
            # Check for type hints
            if '->' in code or ': ' in re.findall(r'def.*\(.*\)', code)[0]:
                readability_score += 0.1
        
        # Penalize very long lines
        very_long_lines = sum(1 for l in lines if len(l) > 100)
        if very_long_lines > 0:
            readability_score -= 0.1
        
        return max(0.0, min(1.0, readability_score))
    
    def _evaluate_security(self, code: str) -> float:
        """Evaluate code security (0-1 scale)."""
        security_score = 1.0
        
        # Critical security issues
        critical_patterns = [
            ('eval(', 0.5),
            ('exec(', 0.5),
            ('__import__(', 0.4),
            ('os.system(', 0.4),
            ('subprocess.call(', 0.3),
        ]
        
        for pattern, penalty in critical_patterns:
            if pattern in code:
                security_score -= penalty
        
        # Input validation checks
        if 'input(' in code:
            # Check if input is validated
            if not any(kw in code for kw in ['validate', 'check', 'verify', 'try:']):
                security_score -= 0.2
        
        # SQL injection risks
        if any(db in code for db in ['SELECT', 'INSERT', 'UPDATE', 'DELETE']):
            if '+' in code and '"' in code:  # String concatenation with SQL
                security_score -= 0.3
        
        # File operations
        if 'open(' in code:
            # Check for path validation
            if not any(kw in code for kw in ['os.path', 'pathlib', 'abspath']):
                security_score -= 0.1
        
        return max(0.0, security_score)
    
    def _identify_issues(self, code: str) -> List[str]:
        """Identify specific issues in code."""
        issues = []
        
        # No error handling
        if 'def ' in code and 'try:' not in code:
            issues.append("Consider adding error handling for edge cases")
        
        # Magic numbers
        numbers = re.findall(r'\b\d+\b', code)
        if len(numbers) > 3:
            issues.append("Consider using named constants instead of magic numbers")
        
        # No input validation
        if any(func in code for func in ['input(', 'sys.argv', 'request.']):
            if not any(val in code for val in ['validate', 'check', 'assert']):
                issues.append("Add input validation for user-provided data")
        
        # Global variables
        if re.search(r'^[A-Z_]+\s*=', code, re.MULTILINE):
            if 'global' in code:
                issues.append("Avoid using global variables when possible")
        
        return issues
    
    def _generate_suggestions(self, code: str) -> List[str]:
        """Generate improvement suggestions."""
        suggestions = []
        
        # Function decomposition
        lines = code.split('\n')
        if any(len(l) > 100 for l in lines):
            suggestions.append("Consider breaking long lines for better readability")
        
        # Type hints
        if 'def ' in code and '->' not in code:
            suggestions.append("Add type hints to function signatures")
        
        # Documentation
        if '"""' not in code and len(lines) > 10:
            suggestions.append("Add docstrings to document function purpose and parameters")
        
        # Testing
        if 'assert' not in code and 'test_' not in code:
            suggestions.append("Consider adding unit tests or assertions")
        
        return suggestions
    
    def _calculate_overall_score(self, evaluation: Dict) -> float:
        """Calculate weighted overall score."""
        weights = {
            'syntax_valid': 0.2,
            'executes': 0.2,
            'passes_tests': 0.3,
            'efficiency_score': 0.1,
            'readability_score': 0.1,
            'security_score': 0.1
        }
        
        score = 0.0
        
        # Binary scores
        if evaluation['syntax_valid']:
            score += weights['syntax_valid']
        if evaluation['executes']:
            score += weights['executes']
        if evaluation['passes_tests'] or not evaluation['test_results']:
            score += weights['passes_tests']
        
        # Continuous scores
        score += evaluation['efficiency_score'] * weights['efficiency_score']
        score += evaluation['readability_score'] * weights['readability_score']
        score += evaluation['security_score'] * weights['security_score']
        
        return score
    
    def get_evaluation_summary(self) -> Dict:
        """Get summary of all evaluations performed."""
        if not self.evaluation_history:
            return {'total_evaluations': 0}
        
        scores = [e['evaluation']['overall_score'] 
                 for e in self.evaluation_history]
        
        return {
            'total_evaluations': len(self.evaluation_history),
            'average_score': sum(scores) / len(scores),
            'min_score': min(scores),
            'max_score': max(scores),
            'syntax_success_rate': sum(1 for e in self.evaluation_history 
                                     if e['evaluation']['syntax_valid']) / len(self.evaluation_history),
            'execution_success_rate': sum(1 for e in self.evaluation_history 
                                        if e['evaluation']['executes']) / len(self.evaluation_history)
        }