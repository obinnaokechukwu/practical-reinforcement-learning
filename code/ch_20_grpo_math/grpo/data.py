"""
Dataset utilities for GRPO training.
Handles loading and preprocessing of math and code problems.
"""

import json
import random
from typing import List, Dict, Optional, Union
from torch.utils.data import Dataset, DataLoader
import re


class MathDataset(Dataset):
    """
    Dataset for mathematical reasoning problems.
    """
    
    def __init__(self, 
                 data_path: str,
                 max_prompt_length: int = 512,
                 problem_types: Optional[List[str]] = None):
        """
        Args:
            data_path: Path to JSON file with problems
            max_prompt_length: Maximum prompt length
            problem_types: Filter for specific problem types
        """
        self.max_prompt_length = max_prompt_length
        
        # Load data
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        # Filter by problem type if specified
        if problem_types:
            self.data = [
                item for item in self.data 
                if item.get('type', 'general') in problem_types
            ]
        
        print(f"Loaded {len(self.data)} math problems")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Format prompt
        prompt = self._format_prompt(item)
        
        return {
            'prompt': prompt,
            'ground_truth': item.get('answer', ''),
            'type': item.get('type', 'general'),
            'difficulty': item.get('difficulty', 'medium'),
            'metadata': item.get('metadata', {})
        }
    
    def _format_prompt(self, item: Dict) -> str:
        """Format problem into prompt."""
        problem = item['problem']
        
        # Add instruction prefix
        prompt = "Solve the following mathematical problem step by step.\n\n"
        prompt += f"Problem: {problem}\n"
        
        # Add any additional context
        if 'context' in item:
            prompt += f"Context: {item['context']}\n"
        
        # Add format instruction
        prompt += "\nProvide your reasoning in <think> tags and your final answer in <answer> tags."
        
        # Truncate if needed
        if len(prompt) > self.max_prompt_length:
            prompt = prompt[:self.max_prompt_length - 3] + "..."
        
        return prompt


class CodeDataset(Dataset):
    """
    Dataset for code generation problems.
    """
    
    def __init__(self,
                 data_path: str,
                 max_prompt_length: int = 768,
                 languages: Optional[List[str]] = None):
        """
        Args:
            data_path: Path to JSON file with problems
            max_prompt_length: Maximum prompt length
            languages: Filter for specific programming languages
        """
        self.max_prompt_length = max_prompt_length
        
        # Load data
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        # Filter by language if specified
        if languages:
            self.data = [
                item for item in self.data
                if item.get('language', 'python') in languages
            ]
        
        print(f"Loaded {len(self.data)} code problems")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Format prompt
        prompt = self._format_prompt(item)
        
        return {
            'prompt': prompt,
            'ground_truth': item.get('solution', ''),
            'test_cases': item.get('test_cases', []),
            'language': item.get('language', 'python'),
            'difficulty': item.get('difficulty', 'medium'),
            'metadata': item.get('metadata', {})
        }
    
    def _format_prompt(self, item: Dict) -> str:
        """Format problem into prompt."""
        problem = item['problem']
        language = item.get('language', 'python')
        
        # Add instruction prefix
        prompt = f"Write a {language} solution for the following problem.\n\n"
        prompt += f"Problem: {problem}\n"
        
        # Add examples if available
        if 'examples' in item:
            prompt += "\nExamples:\n"
            for example in item['examples'][:3]:  # Limit examples
                prompt += f"Input: {example['input']}\n"
                prompt += f"Output: {example['output']}\n\n"
        
        # Add constraints if available
        if 'constraints' in item:
            prompt += f"\nConstraints:\n"
            for constraint in item['constraints']:
                prompt += f"- {constraint}\n"
        
        # Add format instruction
        prompt += "\nProvide your reasoning in <think> tags and your code solution in <answer> tags."
        
        # Truncate if needed
        if len(prompt) > self.max_prompt_length:
            prompt = prompt[:self.max_prompt_length - 3] + "..."
        
        return prompt


class MixedDataset(Dataset):
    """
    Dataset combining multiple problem types.
    """
    
    def __init__(self, datasets: List[Dataset], weights: Optional[List[float]] = None):
        """
        Args:
            datasets: List of datasets to combine
            weights: Sampling weights for each dataset
        """
        self.datasets = datasets
        self.weights = weights or [1.0] * len(datasets)
        
        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
        
        # Calculate dataset sizes
        self.dataset_sizes = [len(d) for d in datasets]
        self.total_size = sum(self.dataset_sizes)
        
        # Create sampling distribution
        self._create_sampling_distribution()
    
    def _create_sampling_distribution(self):
        """Create distribution for sampling from datasets."""
        self.sample_indices = []
        
        for dataset_idx, (dataset, weight) in enumerate(zip(self.datasets, self.weights)):
            num_samples = int(self.total_size * weight)
            dataset_size = len(dataset)
            
            # Create indices with repetition if needed
            for _ in range(num_samples):
                item_idx = random.randint(0, dataset_size - 1)
                self.sample_indices.append((dataset_idx, item_idx))
        
        # Shuffle for random ordering
        random.shuffle(self.sample_indices)
    
    def __len__(self):
        return len(self.sample_indices)
    
    def __getitem__(self, idx):
        dataset_idx, item_idx = self.sample_indices[idx]
        return self.datasets[dataset_idx][item_idx]


class StreamingDataset(Dataset):
    """
    Dataset that generates problems on the fly.
    Useful for infinite training data.
    """
    
    def __init__(self, problem_generator, cache_size: int = 10000):
        """
        Args:
            problem_generator: Function that generates problems
            cache_size: Number of problems to cache
        """
        self.generator = problem_generator
        self.cache_size = cache_size
        self.cache = []
        
        # Pre-fill cache
        self._refill_cache()
    
    def _refill_cache(self):
        """Refill the problem cache."""
        self.cache = []
        for _ in range(self.cache_size):
            problem = self.generator()
            self.cache.append(problem)
    
    def __len__(self):
        return self.cache_size
    
    def __getitem__(self, idx):
        # Return from cache
        return self.cache[idx % len(self.cache)]
    
    def refresh(self):
        """Refresh the cache with new problems."""
        self._refill_cache()


def create_data_loader(
    dataset: Dataset,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    collate_fn: Optional[callable] = None
) -> DataLoader:
    """
    Create DataLoader with appropriate settings.
    
    Args:
        dataset: Dataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of data loading workers
        collate_fn: Custom collate function
        
    Returns:
        DataLoader instance
    """
    if collate_fn is None:
        collate_fn = default_collate_fn
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )


def default_collate_fn(batch: List[Dict]) -> Dict[str, List]:
    """
    Default collate function for GRPO training.
    
    Args:
        batch: List of dictionaries from dataset
        
    Returns:
        Dictionary with lists of prompts and metadata
    """
    collated = {
        'prompts': [item['prompt'] for item in batch]
    }
    
    # Include other fields if present
    if 'ground_truth' in batch[0]:
        collated['ground_truths'] = [item['ground_truth'] for item in batch]
    
    if 'type' in batch[0]:
        collated['types'] = [item['type'] for item in batch]
    
    if 'difficulty' in batch[0]:
        collated['difficulties'] = [item['difficulty'] for item in batch]
    
    return collated


# Problem generators for streaming dataset

def generate_arithmetic_problem() -> Dict:
    """Generate random arithmetic problem."""
    operators = ['+', '-', '*', '/']
    num_terms = random.randint(2, 4)
    
    # Generate expression
    terms = [random.randint(1, 100) for _ in range(num_terms)]
    ops = [random.choice(operators) for _ in range(num_terms - 1)]
    
    expression = str(terms[0])
    for i, op in enumerate(ops):
        expression += f" {op} {terms[i+1]}"
    
    # Evaluate
    result = eval(expression)
    
    return {
        'problem': f"Calculate: {expression}",
        'answer': str(result),
        'type': 'arithmetic',
        'difficulty': 'easy'
    }


def generate_algebra_problem() -> Dict:
    """Generate random algebra problem."""
    # Simple linear equation
    a = random.randint(2, 10)
    b = random.randint(-20, 20)
    c = random.randint(-50, 50)
    
    # ax + b = c
    solution = (c - b) / a
    
    problem = f"Solve for x: {a}x"
    if b >= 0:
        problem += f" + {b}"
    else:
        problem += f" - {-b}"
    problem += f" = {c}"
    
    return {
        'problem': problem,
        'answer': str(solution),
        'type': 'algebra',
        'difficulty': 'medium'
    }


def generate_word_problem() -> Dict:
    """Generate random word problem."""
    templates = [
        {
            'template': "A store sells {item1} for ${price1} each and {item2} for ${price2} each. "
                       "If someone buys {count1} {item1} and {count2} {item2}, how much do they spend in total?",
            'solve': lambda p: p['price1'] * p['count1'] + p['price2'] * p['count2']
        },
        {
            'template': "A train travels at {speed} km/h for {time} hours. How far does it travel?",
            'solve': lambda p: p['speed'] * p['time']
        },
        {
            'template': "A rectangle has length {length} cm and width {width} cm. What is its area?",
            'solve': lambda p: p['length'] * p['width']
        }
    ]
    
    template_data = random.choice(templates)
    template = template_data['template']
    solve_fn = template_data['solve']
    
    # Generate parameters
    params = {}
    for match in re.finditer(r'\{(\w+)\}', template):
        param_name = match.group(1)
        if param_name.startswith('item'):
            params[param_name] = random.choice(['apples', 'oranges', 'bananas', 'books', 'pens'])
        elif param_name.startswith('price'):
            params[param_name] = random.randint(1, 20)
        elif param_name.startswith('count'):
            params[param_name] = random.randint(1, 10)
        elif param_name == 'speed':
            params[param_name] = random.randint(40, 120)
        elif param_name == 'time':
            params[param_name] = random.randint(1, 8)
        elif param_name in ['length', 'width']:
            params[param_name] = random.randint(5, 50)
    
    # Format problem
    problem = template.format(**params)
    answer = solve_fn(params)
    
    return {
        'problem': problem,
        'answer': str(answer),
        'type': 'word_problem',
        'difficulty': 'medium'
    }