"""
Reward functions for GRPO training.
Includes format rewards and task-specific rewards.
"""

import re
import ast
import sympy
from typing import Optional, Dict, Tuple
import subprocess
import tempfile
import os
import signal
import json


class FormatReward:
    """
    Reward for following the thinking format.
    Encourages structured reasoning with <think> and <answer> tags.
    """
    
    def __init__(self, weight: float = 0.1):
        self.weight = weight
        self.think_pattern = re.compile(r'<think>(.*?)</think>', re.DOTALL)
        self.answer_pattern = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
    
    def __call__(self, prompt: str, response: str) -> float:
        """Compute format reward."""
        # Check for presence of tags
        has_think = bool(self.think_pattern.search(response))
        has_answer = bool(self.answer_pattern.search(response))
        
        if not (has_think and has_answer):
            return 0.0
        
        # Check ordering
        think_match = self.think_pattern.search(response)
        answer_match = self.answer_pattern.search(response)
        
        if think_match.end() > answer_match.start():
            # Think section extends past answer start
            return 0.0
        
        # Check that thinking is non-empty
        think_content = think_match.group(1).strip()
        if len(think_content) < 10:  # Minimum thinking length
            return 0.0
        
        return self.weight
    
    def extract_sections(self, response: str) -> Dict[str, str]:
        """Extract thinking and answer sections."""
        sections = {
            'thinking': '',
            'answer': ''
        }
        
        think_match = self.think_pattern.search(response)
        if think_match:
            sections['thinking'] = think_match.group(1).strip()
        
        answer_match = self.answer_pattern.search(response)
        if answer_match:
            sections['answer'] = answer_match.group(1).strip()
        
        return sections


class MathRewardFunction:
    """
    Reward function for mathematical problems.
    Evaluates correctness of mathematical answers.
    """
    
    def __init__(self, 
                 format_reward: Optional[FormatReward] = None,
                 correct_weight: float = 1.0,
                 partial_credit: bool = True):
        self.format_reward = format_reward or FormatReward()
        self.correct_weight = correct_weight
        self.partial_credit = partial_credit
    
    def __call__(self, prompt: str, response: str) -> float:
        """Compute total reward for math response."""
        total_reward = 0.0
        
        # Format reward
        format_score = self.format_reward(prompt, response)
        total_reward += format_score
        
        # Extract answer
        sections = self.format_reward.extract_sections(response)
        if not sections['answer']:
            return total_reward
        
        # Parse ground truth from prompt
        ground_truth = self._extract_ground_truth(prompt)
        if ground_truth is None:
            return total_reward
        
        # Check correctness
        is_correct = self._check_math_answer(
            sections['answer'], 
            ground_truth,
            sections.get('thinking', '')
        )
        
        if is_correct:
            total_reward += self.correct_weight
        elif self.partial_credit:
            # Give partial credit for reasonable attempts
            partial = self._compute_partial_credit(
                sections['answer'],
                ground_truth,
                sections.get('thinking', '')
            )
            total_reward += partial * self.correct_weight
        
        return total_reward
    
    def _extract_ground_truth(self, prompt: str) -> Optional[str]:
        """Extract ground truth answer from prompt."""
        # Look for patterns like "Answer: X" or "= X"
        patterns = [
            r'Answer:\s*([^\n]+)',
            r'answer:\s*([^\n]+)',
            r'=\s*([^\n]+)$',
            r'equals\s+([^\n]+)',
            r'Target:\s*([^\n]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, prompt, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _check_math_answer(self, predicted: str, ground_truth: str, 
                          thinking: str = '') -> bool:
        """Check if mathematical answer is correct."""
        # Clean answers
        predicted = self._clean_math_answer(predicted)
        ground_truth = self._clean_math_answer(ground_truth)
        
        # Direct string comparison
        if predicted == ground_truth:
            return True
        
        # Try numerical comparison
        try:
            pred_val = self._evaluate_expression(predicted)
            true_val = self._evaluate_expression(ground_truth)
            
            if pred_val is not None and true_val is not None:
                # Check if values are close (for floating point)
                if isinstance(pred_val, (int, float)) and isinstance(true_val, (int, float)):
                    return abs(pred_val - true_val) < 1e-6
                else:
                    return pred_val == true_val
        except:
            pass
        
        # Try symbolic comparison
        try:
            pred_expr = sympy.sympify(predicted)
            true_expr = sympy.sympify(ground_truth)
            return sympy.simplify(pred_expr - true_expr) == 0
        except:
            pass
        
        return False
    
    def _clean_math_answer(self, answer: str) -> str:
        """Clean mathematical answer for comparison."""
        # Remove common formatting
        answer = answer.strip()
        answer = answer.rstrip('.')
        answer = answer.replace('$', '')
        answer = answer.replace('\\', '')
        
        # Handle boxed answers
        boxed_match = re.search(r'\\boxed\{([^}]+)\}', answer)
        if boxed_match:
            answer = boxed_match.group(1)
        
        # Remove units if present (simplified)
        answer = re.sub(r'\s*(dollars|cents|meters|cm|kg|hours|minutes).*$', '', answer, flags=re.IGNORECASE)
        
        return answer.strip()
    
    def _evaluate_expression(self, expr: str):
        """Safely evaluate mathematical expression."""
        try:
            # Replace common math notation
            expr = expr.replace('^', '**')
            expr = expr.replace('×', '*')
            expr = expr.replace('÷', '/')
            
            # Use ast for safe evaluation
            node = ast.parse(expr, mode='eval')
            
            # Check that it's a safe expression
            for n in ast.walk(node):
                if not isinstance(n, (ast.Expression, ast.Num, ast.BinOp, 
                                    ast.UnaryOp, ast.Add, ast.Sub, ast.Mult, 
                                    ast.Div, ast.Pow, ast.USub, ast.UAdd)):
                    if not (isinstance(n, ast.Name) and n.id in ['pi', 'e']):
                        return None
            
            # Evaluate
            result = eval(compile(node, '<string>', 'eval'), 
                         {"__builtins__": {}, "pi": 3.14159265359, "e": 2.71828182846})
            return result
        except:
            return None
    
    def _compute_partial_credit(self, predicted: str, ground_truth: str, 
                               thinking: str) -> float:
        """Compute partial credit for incorrect but reasonable answers."""
        # Check if answer is in correct format
        if not predicted:
            return 0.0
        
        # Check if thinking shows reasonable approach
        if thinking:
            # Look for key mathematical operations
            math_keywords = ['solve', 'equation', 'calculate', 'multiply', 
                           'divide', 'add', 'subtract', 'factor', 'simplify']
            keyword_count = sum(1 for kw in math_keywords if kw in thinking.lower())
            
            if keyword_count >= 2:
                return 0.2  # Partial credit for reasonable approach
        
        # Check if answer is numerically close (for computational errors)
        try:
            pred_val = self._evaluate_expression(predicted)
            true_val = self._evaluate_expression(ground_truth)
            
            if pred_val is not None and true_val is not None:
                if isinstance(pred_val, (int, float)) and isinstance(true_val, (int, float)):
                    if true_val != 0:
                        relative_error = abs(pred_val - true_val) / abs(true_val)
                        if relative_error < 0.1:  # Within 10%
                            return 0.5
                        elif relative_error < 0.25:  # Within 25%
                            return 0.3
        except:
            pass
        
        return 0.0


class CodeRewardFunction:
    """
    Reward function for code generation problems.
    Evaluates code correctness through execution.
    """
    
    def __init__(self,
                 format_reward: Optional[FormatReward] = None,
                 correct_weight: float = 1.0,
                 timeout: int = 5):
        self.format_reward = format_reward or FormatReward()
        self.correct_weight = correct_weight
        self.timeout = timeout
    
    def __call__(self, prompt: str, response: str) -> float:
        """Compute total reward for code response."""
        total_reward = 0.0
        
        # Format reward
        format_score = self.format_reward(prompt, response)
        total_reward += format_score
        
        # Extract answer
        sections = self.format_reward.extract_sections(response)
        if not sections['answer']:
            return total_reward
        
        # Extract code from answer
        code = self._extract_code(sections['answer'])
        if not code:
            return total_reward
        
        # Parse test cases from prompt
        test_cases = self._extract_test_cases(prompt)
        if not test_cases:
            # If no test cases, check if code runs without error
            if self._check_syntax(code):
                total_reward += self.correct_weight * 0.3
            return total_reward
        
        # Run test cases
        passed, total = self._run_test_cases(code, test_cases)
        
        # Compute reward based on test results
        if total > 0:
            success_rate = passed / total
            total_reward += self.correct_weight * success_rate
        
        return total_reward
    
    def _extract_code(self, answer: str) -> str:
        """Extract code from answer section."""
        # Look for code blocks
        code_block_match = re.search(r'```(?:python)?\n(.*?)```', answer, re.DOTALL)
        if code_block_match:
            return code_block_match.group(1).strip()
        
        # If no code block, assume entire answer is code
        return answer.strip()
    
    def _extract_test_cases(self, prompt: str) -> list:
        """Extract test cases from prompt."""
        test_cases = []
        
        # Look for example format: "f(input) = output"
        pattern = r'(\w+)\(([^)]+)\)\s*=\s*([^\n]+)'
        matches = re.findall(pattern, prompt)
        
        for func_name, input_str, output_str in matches:
            test_cases.append({
                'function': func_name,
                'input': input_str.strip(),
                'output': output_str.strip()
            })
        
        # Also look for assert statements
        assert_pattern = r'assert\s+([^\n]+)'
        assert_matches = re.findall(assert_pattern, prompt)
        
        for assert_str in assert_matches:
            test_cases.append({
                'assert': assert_str.strip()
            })
        
        return test_cases
    
    def _check_syntax(self, code: str) -> bool:
        """Check if code has valid Python syntax."""
        try:
            compile(code, '<string>', 'exec')
            return True
        except SyntaxError:
            return False
    
    def _run_test_cases(self, code: str, test_cases: list) -> Tuple[int, int]:
        """Run test cases and return (passed, total)."""
        passed = 0
        total = len(test_cases)
        
        for test_case in test_cases:
            try:
                result = self._run_single_test(code, test_case)
                if result:
                    passed += 1
            except:
                # Test failed
                pass
        
        return passed, total
    
    def _run_single_test(self, code: str, test_case: dict) -> bool:
        """Run a single test case."""
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            # Write code
            f.write(code)
            f.write('\n\n')
            
            # Write test
            if 'assert' in test_case:
                f.write(f"assert {test_case['assert']}\n")
            else:
                func_name = test_case['function']
                input_str = test_case['input']
                output_str = test_case['output']
                
                # Try to parse output as Python literal
                try:
                    expected = ast.literal_eval(output_str)
                    f.write(f"assert {func_name}({input_str}) == {repr(expected)}\n")
                except:
                    # Fallback to string comparison
                    f.write(f"assert str({func_name}({input_str})) == {repr(output_str)}\n")
            
            f.write("print('PASS')\n")
            temp_path = f.name
        
        try:
            # Run the test
            process = subprocess.Popen(
                ['python', temp_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait with timeout
            stdout, stderr = process.communicate(timeout=self.timeout)
            
            # Check result
            success = process.returncode == 0 and 'PASS' in stdout
            
            return success
            
        except subprocess.TimeoutExpired:
            process.kill()
            return False
        finally:
            # Clean up
            os.unlink(temp_path)


class CompositeRewardFunction:
    """
    Combines multiple reward functions.
    """
    
    def __init__(self, reward_functions: Dict[str, Tuple[callable, float]]):
        """
        Args:
            reward_functions: Dict mapping names to (function, weight) tuples
        """
        self.reward_functions = reward_functions
    
    def __call__(self, prompt: str, response: str) -> float:
        """Compute weighted sum of rewards."""
        total_reward = 0.0
        
        for name, (func, weight) in self.reward_functions.items():
            reward = func(prompt, response)
            total_reward += weight * reward
        
        return total_reward
    
    def get_detailed_rewards(self, prompt: str, response: str) -> Dict[str, float]:
        """Get individual reward components."""
        rewards = {}
        
        for name, (func, weight) in self.reward_functions.items():
            reward = func(prompt, response)
            rewards[name] = reward
            rewards[f'{name}_weighted'] = weight * reward
        
        rewards['total'] = sum(
            rewards[f'{name}_weighted'] 
            for name in self.reward_functions.keys()
        )
        
        return rewards