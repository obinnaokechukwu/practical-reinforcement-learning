"""
Dataset utilities for RLHF pipeline.
Handles SFT data, preference data, and prompt generation.
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import Dict, List, Optional, Tuple
import random


class SFTDataset(Dataset):
    """Dataset for supervised fine-tuning."""
    
    def __init__(self, 
                 data_path: str,
                 tokenizer: AutoTokenizer,
                 max_length: int = 512,
                 prompt_template: str = "Human: {prompt}\n\nAssistant: {response}"):
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_template = prompt_template
        
        # Load data
        with open(data_path, 'r') as f:
            self.data = json.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Format the conversation
        text = self.prompt_template.format(
            prompt=item['prompt'],
            response=item['response']
        )
        
        # Tokenize
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Create labels (mask the prompt part)
        labels = encoded['input_ids'].clone()
        
        # Find where response starts
        prompt_text = self.prompt_template.split('{response}')[0].format(prompt=item['prompt'])
        prompt_tokens = self.tokenizer(prompt_text, add_special_tokens=False)['input_ids']
        labels[0, :len(prompt_tokens)] = -100  # Mask prompt tokens
        
        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'labels': labels.squeeze()
        }


class PreferenceDataset(Dataset):
    """Dataset for reward model training with preference pairs."""
    
    def __init__(self,
                 data_path: str,
                 tokenizer: AutoTokenizer,
                 max_length: int = 512,
                 prompt_template: str = "Human: {prompt}\n\nAssistant: {response}"):
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_template = prompt_template
        
        # Load preference data
        with open(data_path, 'r') as f:
            self.data = json.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Format chosen and rejected responses
        chosen_text = self.prompt_template.format(
            prompt=item['prompt'],
            response=item['chosen']
        )
        rejected_text = self.prompt_template.format(
            prompt=item['prompt'],
            response=item['rejected']
        )
        
        # Tokenize both
        chosen_encoded = self.tokenizer(
            chosen_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        rejected_encoded = self.tokenizer(
            rejected_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'chosen_input_ids': chosen_encoded['input_ids'].squeeze(),
            'chosen_attention_mask': chosen_encoded['attention_mask'].squeeze(),
            'rejected_input_ids': rejected_encoded['input_ids'].squeeze(),
            'rejected_attention_mask': rejected_encoded['attention_mask'].squeeze(),
            'prompt': item['prompt']
        }


class PromptDataset(Dataset):
    """Dataset of prompts for RLHF generation."""
    
    def __init__(self,
                 data_path: str,
                 tokenizer: AutoTokenizer,
                 max_prompt_length: int = 128,
                 prompt_template: str = "Human: {prompt}\n\nAssistant:"):
        
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.prompt_template = prompt_template
        
        # Load prompts
        with open(data_path, 'r') as f:
            data = json.load(f)
            self.prompts = data if isinstance(data, list) else data['prompts']
    
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        
        # Format prompt
        formatted_prompt = self.prompt_template.format(prompt=prompt)
        
        # Tokenize
        encoded = self.tokenizer(
            formatted_prompt,
            truncation=True,
            max_length=self.max_prompt_length,
            padding='max_length',
            return_tensors='pt',
            return_attention_mask=True
        )
        
        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'prompt_text': prompt,
            'formatted_prompt': formatted_prompt
        }


class PreferenceCollector:
    """Utility for collecting preference data from model outputs."""
    
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer
        self.comparisons = []
    
    def create_comparison(self,
                         prompt: str,
                         response1: str,
                         response2: str,
                         metadata: Optional[Dict] = None) -> Dict:
        """Create a comparison for human annotation."""
        comparison = {
            'id': len(self.comparisons),
            'prompt': prompt,
            'response1': response1,
            'response2': response2,
            'metadata': metadata or {}
        }
        
        self.comparisons.append(comparison)
        return comparison
    
    def add_preference(self, comparison_id: int, preference: str, reason: Optional[str] = None):
        """Add human preference to a comparison."""
        if comparison_id >= len(self.comparisons):
            raise ValueError(f"Invalid comparison ID: {comparison_id}")
        
        comparison = self.comparisons[comparison_id]
        
        if preference == 'response1':
            comparison['chosen'] = comparison['response1']
            comparison['rejected'] = comparison['response2']
        elif preference == 'response2':
            comparison['chosen'] = comparison['response2']
            comparison['rejected'] = comparison['response1']
        else:
            raise ValueError(f"Invalid preference: {preference}")
        
        comparison['preference'] = preference
        comparison['reason'] = reason
        comparison['annotated'] = True
    
    def get_annotated_comparisons(self) -> List[Dict]:
        """Get all annotated comparisons."""
        return [c for c in self.comparisons if c.get('annotated', False)]
    
    def save_preferences(self, output_path: str):
        """Save preference data to file."""
        annotated = self.get_annotated_comparisons()
        
        # Format for training
        formatted_data = []
        for comp in annotated:
            formatted_data.append({
                'prompt': comp['prompt'],
                'chosen': comp['chosen'],
                'rejected': comp['rejected'],
                'metadata': comp.get('metadata', {}),
                'reason': comp.get('reason', '')
            })
        
        with open(output_path, 'w') as f:
            json.dump(formatted_data, f, indent=2)
        
        print(f"Saved {len(formatted_data)} preference pairs to {output_path}")


class RLHFDataCollator:
    """Custom data collator for RLHF training."""
    
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate batch for training."""
        # Handle different batch types
        if 'labels' in batch[0]:  # SFT data
            return self._collate_sft(batch)
        elif 'chosen_input_ids' in batch[0]:  # Preference data
            return self._collate_preferences(batch)
        else:  # Prompt data
            return self._collate_prompts(batch)
    
    def _collate_sft(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate SFT training data."""
        input_ids = torch.stack([x['input_ids'] for x in batch])
        attention_mask = torch.stack([x['attention_mask'] for x in batch])
        labels = torch.stack([x['labels'] for x in batch])
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
    
    def _collate_preferences(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate preference pair data."""
        chosen_input_ids = torch.stack([x['chosen_input_ids'] for x in batch])
        chosen_attention_mask = torch.stack([x['chosen_attention_mask'] for x in batch])
        rejected_input_ids = torch.stack([x['rejected_input_ids'] for x in batch])
        rejected_attention_mask = torch.stack([x['rejected_attention_mask'] for x in batch])
        
        return {
            'chosen_input_ids': chosen_input_ids,
            'chosen_attention_mask': chosen_attention_mask,
            'rejected_input_ids': rejected_input_ids,
            'rejected_attention_mask': rejected_attention_mask
        }
    
    def _collate_prompts(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate prompt data."""
        input_ids = torch.stack([x['input_ids'] for x in batch])
        attention_mask = torch.stack([x['attention_mask'] for x in batch])
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'prompts': [x['prompt_text'] for x in batch]
        }


def create_sft_dataloader(data_path: str,
                         tokenizer: AutoTokenizer,
                         batch_size: int = 8,
                         max_length: int = 512,
                         shuffle: bool = True) -> DataLoader:
    """Create dataloader for SFT training."""
    dataset = SFTDataset(data_path, tokenizer, max_length)
    collator = RLHFDataCollator(tokenizer)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator
    )


def create_preference_dataloader(data_path: str,
                               tokenizer: AutoTokenizer,
                               batch_size: int = 4,
                               max_length: int = 512,
                               shuffle: bool = True) -> DataLoader:
    """Create dataloader for reward model training."""
    dataset = PreferenceDataset(data_path, tokenizer, max_length)
    collator = RLHFDataCollator(tokenizer)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator
    )


def create_prompt_dataloader(data_path: str,
                           tokenizer: AutoTokenizer,
                           batch_size: int = 16,
                           max_prompt_length: int = 128,
                           shuffle: bool = True) -> DataLoader:
    """Create dataloader for RLHF generation."""
    dataset = PromptDataset(data_path, tokenizer, max_prompt_length)
    collator = RLHFDataCollator(tokenizer)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator
    )