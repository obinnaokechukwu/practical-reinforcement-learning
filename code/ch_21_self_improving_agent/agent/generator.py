"""
Constitutional code generator with self-improvement capabilities.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Optional, Tuple
import re
import ast

class ConstitutionalCodeGenerator:
    """Code generator that follows constitutional principles."""
    
    def __init__(self, model_name: str, constitution: List[str], 
                 device: Optional[str] = None):
        """
        Args:
            model_name: Pretrained model name or path
            constitution: List of principles to follow
            device: Device to use for generation
        """
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.constitution = constitution
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def generate_code(self, problem: str, 
                     max_revisions: int = 3,
                     temperature: float = 0.7) -> Dict:
        """
        Generate code with self-improvement loop.
        
        Args:
            problem: Problem description
            max_revisions: Maximum number of revision attempts
            temperature: Generation temperature
            
        Returns:
            Dictionary with final code and revision history
        """
        # Initial generation
        initial_code = self._generate_initial_solution(problem, temperature)
        
        current_code = initial_code
        revision_history = [{
            'revision': 0,
            'code': initial_code,
            'critique': None
        }]
        
        for revision in range(1, max_revisions + 1):
            # Self-critique against constitution
            critique = self._self_critique(problem, current_code)
            
            if not critique['needs_revision']:
                break
            
            # Generate revised solution
            revised_code = self._generate_revision(
                problem, current_code, critique, temperature
            )
            
            # Check if revision improves quality
            if self._is_improvement(problem, current_code, revised_code):
                current_code = revised_code
                revision_history.append({
                    'revision': revision,
                    'code': revised_code,
                    'critique': critique
                })
            else:
                break  # No improvement, stop revising
        
        return {
            'final_code': current_code,
            'revision_history': revision_history,
            'num_revisions': len(revision_history) - 1,
            'followed_constitution': self._check_constitution_compliance(current_code)
        }
    
    def _generate_initial_solution(self, problem: str, 
                                  temperature: float) -> str:
        """Generate initial code solution."""
        
        prompt = f"""Problem: {problem}

Please provide a Python solution following these principles:
{chr(10).join(f"- {p}" for p in self.constitution)}

Solution:
```python"""
        
        inputs = self.tokenizer(
            prompt, 
            return_tensors='pt',
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        generated = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        
        # Extract code from markdown
        return self._extract_code(generated)
    
    def _self_critique(self, problem: str, code: str) -> Dict:
        """Generate self-critique of code."""
        
        critique_prompt = f"""Problem: {problem}

Proposed Solution:
```python
{code}
```

Evaluate this solution against these principles:
{chr(10).join(f"- {p}" for p in self.constitution)}

Identify specific issues that need to be fixed. If the solution is satisfactory, state "No revisions needed."

Critique:"""
        
        inputs = self.tokenizer(
            critique_prompt, 
            return_tensors='pt',
            truncation=True,
            max_length=768
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.3,  # Lower temperature for focused critique
                do_sample=True
            )
        
        critique_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        
        # Parse critique to determine if revision is needed
        needs_revision = self._parse_critique_severity(critique_text)
        issues = self._extract_issues(critique_text)
        
        return {
            'critique_text': critique_text,
            'needs_revision': needs_revision,
            'issues': issues
        }
    
    def _generate_revision(self, problem: str, current_code: str, 
                          critique: Dict, temperature: float) -> str:
        """Generate revised solution based on critique."""
        
        revision_prompt = f"""Problem: {problem}

Current Solution:
```python
{current_code}
```

Issues identified:
{critique['critique_text']}

Please provide an improved solution that addresses these issues while following the constitutional principles:
{chr(10).join(f"- {p}" for p in self.constitution)}

Improved Solution:
```python"""
        
        inputs = self.tokenizer(
            revision_prompt, 
            return_tensors='pt',
            truncation=True,
            max_length=768
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=temperature * 0.9,  # Slightly lower temp for revisions
                do_sample=True
            )
        
        revised = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        
        return self._extract_code(revised)
    
    def _extract_code(self, text: str) -> str:
        """Extract code from generated text."""
        # Look for code blocks
        if '```python' in text:
            match = re.search(r'```python\n(.*?)```', text, re.DOTALL)
            if match:
                return match.group(1).strip()
        elif '```' in text:
            match = re.search(r'```\n(.*?)```', text, re.DOTALL)
            if match:
                return match.group(1).strip()
        
        # If no code blocks, assume everything is code
        return text.strip()
    
    def _parse_critique_severity(self, critique_text: str) -> bool:
        """Determine if critique indicates revision is needed."""
        critique_lower = critique_text.lower()
        
        # Phrases indicating no revision needed
        no_revision_phrases = [
            'no revision', 'no issues', 'satisfactory', 
            'looks good', 'well done', 'correct implementation',
            'no improvements needed', 'follows all principles'
        ]
        
        for phrase in no_revision_phrases:
            if phrase in critique_lower:
                return False
        
        # Phrases indicating revision needed
        revision_phrases = [
            'should', 'could', 'missing', 'lacks', 'needs',
            'improve', 'add', 'fix', 'incorrect', 'error'
        ]
        
        for phrase in revision_phrases:
            if phrase in critique_lower:
                return True
        
        # Default to revision if uncertain
        return True
    
    def _extract_issues(self, critique_text: str) -> List[str]:
        """Extract specific issues from critique."""
        issues = []
        
        # Look for bullet points or numbered lists
        bullet_pattern = r'[-•*]\s*(.+)'
        number_pattern = r'\d+\.\s*(.+)'
        
        for pattern in [bullet_pattern, number_pattern]:
            matches = re.findall(pattern, critique_text)
            issues.extend(matches)
        
        # If no structured list, split by sentences
        if not issues:
            sentences = critique_text.split('.')
            issues = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        return issues
    
    def _is_improvement(self, problem: str, 
                       old_code: str, new_code: str) -> bool:
        """Check if new code is an improvement over old code."""
        # Basic checks
        if not new_code or new_code == old_code:
            return False
        
        # Check if new code is syntactically valid
        try:
            ast.parse(new_code)
        except SyntaxError:
            return False
        
        # Check if new code is longer (often indicates more complete solution)
        if len(new_code.split('\n')) > len(old_code.split('\n')):
            return True
        
        # Check for presence of error handling (improvement indicator)
        if 'try:' in new_code and 'try:' not in old_code:
            return True
        
        # Check for presence of docstrings (improvement indicator)
        if '"""' in new_code and '"""' not in old_code:
            return True
        
        # Default to accepting revision
        return True
    
    def _check_constitution_compliance(self, code: str) -> Dict[str, bool]:
        """Check which constitutional principles are followed."""
        compliance = {}
        
        for principle in self.constitution:
            principle_lower = principle.lower()
            
            # Check for specific patterns based on principle content
            if 'error handling' in principle_lower:
                compliance[principle] = 'try:' in code or 'except' in code
            elif 'comment' in principle_lower:
                compliance[principle] = '#' in code or '"""' in code
            elif 'type hint' in principle_lower:
                compliance[principle] = '->' in code or ': ' in code
            elif 'test' in principle_lower:
                compliance[principle] = 'assert' in code or 'test_' in code
            elif 'docstring' in principle_lower:
                compliance[principle] = '"""' in code
            elif 'validation' in principle_lower:
                compliance[principle] = 'if ' in code and ('raise' in code or 'return' in code)
            else:
                # Default to checking if principle keywords appear in code
                keywords = re.findall(r'\b\w+\b', principle_lower)
                compliance[principle] = any(kw in code.lower() for kw in keywords[:3])
        
        return compliance
    
    def batch_generate(self, problems: List[str], 
                      max_revisions: int = 2) -> List[Dict]:
        """Generate code for multiple problems."""
        results = []
        
        for problem in problems:
            result = self.generate_code(problem, max_revisions)
            results.append({
                'problem': problem,
                'solution': result['final_code'],
                'num_revisions': result['num_revisions'],
                'constitution_compliance': result['followed_constitution']
            })
        
        return results