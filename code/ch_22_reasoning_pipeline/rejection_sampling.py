"""
Rejection sampling and iterative refinement for reasoning models.
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Optional, Tuple, Callable
from dataclasses import dataclass
import heapq
from collections import defaultdict
import math
from tqdm import tqdm


@dataclass
class RejectionSamplingConfig:
    """Configuration for rejection sampling."""
    num_samples: int = 16
    temperature: float = 0.8
    top_p: float = 0.95
    
    # Rejection criteria
    min_score_threshold: float = 0.7
    diversity_bonus: float = 0.1
    length_penalty: float = 0.8
    
    # Refinement settings
    max_refinement_iterations: int = 3
    refinement_temperature: float = 0.5
    refinement_prompt_template: str = "Improve this solution:\n{solution}\n\nImproved solution:"


class RejectionSampler:
    """Basic rejection sampling for high-quality outputs."""
    
    def __init__(self, model, tokenizer, reward_fn: Callable, config: RejectionSamplingConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.config = config
        
        # Statistics tracking
        self.sampling_stats = {
            'total_samples': 0,
            'accepted_samples': 0,
            'avg_score_improvement': [],
            'score_distribution': []
        }
    
    def sample_with_rejection(self, problem: str) -> Dict:
        """Generate multiple samples and select the best."""
        samples = []
        
        # Generate multiple candidate solutions
        for i in range(self.config.num_samples):
            sample = self._generate_sample(problem, seed=i)
            score = self.reward_fn(problem, sample['solution'])
            
            sample['score'] = score
            samples.append(sample)
            
            self.sampling_stats['total_samples'] += 1
            self.sampling_stats['score_distribution'].append(score)
        
        # Sort by score
        samples.sort(key=lambda x: x['score'], reverse=True)
        
        # Apply rejection criteria
        accepted_samples = []
        for sample in samples:
            if self._accept_sample(sample, accepted_samples):
                accepted_samples.append(sample)
                self.sampling_stats['accepted_samples'] += 1
        
        # Select best sample or return None if all rejected
        if accepted_samples:
            best_sample = accepted_samples[0]
            
            # Track improvement
            baseline_score = samples[-1]['score']  # Worst sample
            improvement = best_sample['score'] - baseline_score
            self.sampling_stats['avg_score_improvement'].append(improvement)
            
            return {
                'problem': problem,
                'solution': best_sample['solution'],
                'score': best_sample['score'],
                'num_samples_generated': self.config.num_samples,
                'num_samples_accepted': len(accepted_samples),
                'score_improvement': improvement
            }
        
        return None
    
    def _generate_sample(self, problem: str, seed: int = 0) -> Dict:
        """Generate a single sample solution."""
        # Set different random seed for diversity
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        inputs = self.tokenizer(
            problem,
            return_tensors='pt',
            truncation=True,
            max_length=512
        )
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=1024,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        solution = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        solution = solution[len(problem):].strip()  # Remove problem from output
        
        return {
            'solution': solution,
            'length': len(solution.split()),
            'seed': seed
        }
    
    def _accept_sample(self, sample: Dict, accepted_samples: List[Dict]) -> bool:
        """Determine if sample should be accepted."""
        # Basic threshold
        if sample['score'] < self.config.min_score_threshold:
            return False
        
        # Length penalty for very short or very long solutions
        length = sample['length']
        if length < 10 or length > 500:
            sample['score'] *= self.config.length_penalty
        
        # Diversity bonus if different from accepted samples
        if accepted_samples:
            diversity_score = self._compute_diversity(sample, accepted_samples)
            sample['score'] += diversity_score * self.config.diversity_bonus
        
        return True
    
    def _compute_diversity(self, sample: Dict, accepted_samples: List[Dict]) -> float:
        """Compute diversity score relative to accepted samples."""
        # Simple token-based diversity
        sample_tokens = set(sample['solution'].lower().split())
        
        min_overlap = 1.0
        for accepted in accepted_samples:
            accepted_tokens = set(accepted['solution'].lower().split())
            
            if len(sample_tokens) > 0 and len(accepted_tokens) > 0:
                overlap = len(sample_tokens & accepted_tokens) / len(sample_tokens | accepted_tokens)
                min_overlap = min(min_overlap, overlap)
        
        return 1.0 - min_overlap
    
    def iterative_refinement(self, problem: str, initial_solution: str) -> Dict:
        """Iteratively refine a solution."""
        current_solution = initial_solution
        current_score = self.reward_fn(problem, current_solution)
        
        refinement_history = [{
            'iteration': 0,
            'solution': current_solution,
            'score': current_score
        }]
        
        for iteration in range(1, self.config.max_refinement_iterations + 1):
            # Generate refinement prompt
            refinement_prompt = self.config.refinement_prompt_template.format(
                solution=current_solution
            )
            full_prompt = f"{problem}\n\n{refinement_prompt}"
            
            # Generate refined solution
            refined = self._generate_sample(
                full_prompt, 
                seed=iteration * 100  # Different seeds for refinements
            )
            
            # Evaluate refined solution
            refined_score = self.reward_fn(problem, refined['solution'])
            
            # Accept if improved
            if refined_score > current_score:
                current_solution = refined['solution']
                current_score = refined_score
                
                refinement_history.append({
                    'iteration': iteration,
                    'solution': current_solution,
                    'score': current_score,
                    'improvement': refined_score - refinement_history[-1]['score']
                })
            else:
                # Stop if no improvement
                break
        
        return {
            'final_solution': current_solution,
            'final_score': current_score,
            'num_refinements': len(refinement_history) - 1,
            'total_improvement': current_score - refinement_history[0]['score'],
            'history': refinement_history
        }
    
    def get_sampling_report(self) -> str:
        """Generate report on sampling statistics."""
        if self.sampling_stats['total_samples'] == 0:
            return "No samples generated yet."
        
        acceptance_rate = (self.sampling_stats['accepted_samples'] / 
                          self.sampling_stats['total_samples'])
        
        avg_improvement = (np.mean(self.sampling_stats['avg_score_improvement'])
                          if self.sampling_stats['avg_score_improvement'] else 0)
        
        score_dist = self.sampling_stats['score_distribution']
        
        report = f"""
Rejection Sampling Statistics
============================
Total Samples: {self.sampling_stats['total_samples']}
Accepted Samples: {self.sampling_stats['accepted_samples']}
Acceptance Rate: {acceptance_rate:.1%}

Score Distribution:
- Mean: {np.mean(score_dist):.3f}
- Std: {np.std(score_dist):.3f}
- Min: {np.min(score_dist):.3f}
- Max: {np.max(score_dist):.3f}

Average Score Improvement: {avg_improvement:.3f}
"""
        return report


class DiversityAwareSampler(RejectionSampler):
    """Advanced sampler that explicitly optimizes for diverse high-quality solutions."""
    
    def __init__(self, model, tokenizer, reward_fn: Callable, config: RejectionSamplingConfig):
        super().__init__(model, tokenizer, reward_fn, config)
        
        # Diversity-specific settings
        self.solution_embeddings = []
        self.clustering_threshold = 0.8
        self.diversity_weight = 0.3
    
    def sample_diverse_solutions(self, problem: str, num_diverse: int = 5) -> List[Dict]:
        """Sample diverse high-scoring solutions."""
        # Generate many candidates
        all_samples = []
        for i in range(self.config.num_samples * 2):  # Generate more for diversity
            sample = self._generate_sample(problem, seed=i)
            score = self.reward_fn(problem, sample['solution'])
            sample['score'] = score
            sample['embedding'] = self._compute_embedding(sample['solution'])
            all_samples.append(sample)
        
        # Select diverse subset using MMR (Maximal Marginal Relevance)
        selected = []
        remaining = all_samples.copy()
        
        # Start with highest scoring
        best_idx = max(range(len(remaining)), key=lambda i: remaining[i]['score'])
        selected.append(remaining.pop(best_idx))
        
        # Iteratively select diverse high-scoring samples
        while len(selected) < num_diverse and remaining:
            mmr_scores = []
            
            for candidate in remaining:
                # Relevance (quality score)
                relevance = candidate['score']
                
                # Diversity (minimum similarity to selected)
                max_sim = max(
                    self._compute_similarity(candidate['embedding'], s['embedding'])
                    for s in selected
                )
                diversity = 1 - max_sim
                
                # MMR score
                mmr = (1 - self.diversity_weight) * relevance + self.diversity_weight * diversity
                mmr_scores.append(mmr)
            
            # Select highest MMR
            best_idx = max(range(len(remaining)), key=lambda i: mmr_scores[i])
            selected.append(remaining.pop(best_idx))
        
        return [{
            'problem': problem,
            'solution': s['solution'],
            'score': s['score'],
            'diversity_rank': i + 1
        } for i, s in enumerate(selected)]
    
    def _compute_embedding(self, text: str) -> np.ndarray:
        """Compute simple embedding for diversity calculation."""
        # Simple bag-of-words embedding (in practice, use sentence transformers)
        tokens = text.lower().split()
        
        # Get vocabulary from all seen tokens
        if not hasattr(self, 'vocab'):
            self.vocab = {}
        
        for token in tokens:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
        
        # Create sparse vector
        embedding = np.zeros(len(self.vocab))
        for token in tokens:
            if token in self.vocab:
                embedding[self.vocab[token]] += 1
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def _compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between embeddings."""
        # Ensure same dimensions
        max_dim = max(len(emb1), len(emb2))
        if len(emb1) < max_dim:
            emb1 = np.pad(emb1, (0, max_dim - len(emb1)))
        if len(emb2) < max_dim:
            emb2 = np.pad(emb2, (0, max_dim - len(emb2)))
        
        return np.dot(emb1, emb2)
    
    def ensemble_solutions(self, problem: str, solutions: List[Dict]) -> Dict:
        """Create ensemble solution from diverse samples."""
        # Extract common patterns
        all_steps = []
        for sol in solutions:
            # Simple step extraction (customize based on format)
            steps = [s.strip() for s in sol['solution'].split('\n') if s.strip()]
            all_steps.extend(steps)
        
        # Find most common/agreed upon steps
        step_counts = defaultdict(int)
        for step in all_steps:
            step_counts[step] += 1
        
        # Build consensus solution
        consensus_steps = []
        seen_concepts = set()
        
        for step, count in sorted(step_counts.items(), key=lambda x: x[1], reverse=True):
            # Include if appears in multiple solutions
            if count >= len(solutions) // 2:
                # Avoid redundancy
                step_words = set(step.lower().split())
                if not any(len(step_words & seen) > len(step_words) * 0.7 for seen in seen_concepts):
                    consensus_steps.append(step)
                    seen_concepts.append(step_words)
        
        ensemble_solution = '\n'.join(consensus_steps)
        ensemble_score = self.reward_fn(problem, ensemble_solution)
        
        return {
            'problem': problem,
            'ensemble_solution': ensemble_solution,
            'ensemble_score': ensemble_score,
            'num_components': len(solutions),
            'individual_scores': [s['score'] for s in solutions],
            'score_improvement': ensemble_score - np.mean([s['score'] for s in solutions])
        }


class BeamSearchSampler:
    """Beam search with rejection for structured reasoning."""
    
    def __init__(self, model, tokenizer, reward_fn: Callable, beam_size: int = 5):
        self.model = model
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.beam_size = beam_size
    
    def beam_search_with_rejection(self, problem: str, max_length: int = 512) -> List[Dict]:
        """Perform beam search with step-wise rejection."""
        # Initialize beam with problem
        inputs = self.tokenizer(problem, return_tensors='pt')
        
        beams = [{
            'tokens': inputs['input_ids'][0].tolist(),
            'score': 0.0,
            'finished': False,
            'solution': ""
        }]
        
        # Special tokens
        eos_token_id = self.tokenizer.eos_token_id
        step_token = "<step>"
        if step_token in self.tokenizer.get_vocab():
            step_token_id = self.tokenizer.convert_tokens_to_ids(step_token)
        else:
            step_token_id = None
        
        # Beam search
        for position in range(max_length):
            if all(beam['finished'] for beam in beams):
                break
            
            new_beams = []
            
            for beam in beams:
                if beam['finished']:
                    new_beams.append(beam)
                    continue
                
                # Get next token probabilities
                input_ids = torch.tensor([beam['tokens']])
                
                with torch.no_grad():
                    outputs = self.model(input_ids=input_ids)
                    logits = outputs.logits[0, -1, :]
                    probs = torch.softmax(logits, dim=-1)
                
                # Get top tokens
                top_probs, top_indices = torch.topk(probs, k=self.beam_size * 2)
                
                # Expand beam
                for prob, token_id in zip(top_probs, top_indices):
                    new_tokens = beam['tokens'] + [token_id.item()]
                    
                    # Check if finished
                    finished = token_id.item() == eos_token_id
                    
                    # Compute new score
                    new_score = beam['score'] + math.log(prob.item())
                    
                    # Apply step-wise rejection if at step boundary
                    if step_token_id and token_id.item() == step_token_id:
                        partial_solution = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                        step_quality = self._evaluate_partial_solution(problem, partial_solution)
                        
                        # Reject low-quality steps
                        if step_quality < 0.5:
                            continue
                        
                        new_score += math.log(step_quality)
                    
                    new_beams.append({
                        'tokens': new_tokens,
                        'score': new_score,
                        'finished': finished,
                        'solution': self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                    })
            
            # Keep top beams
            new_beams.sort(key=lambda x: x['score'], reverse=True)
            beams = new_beams[:self.beam_size]
        
        # Final scoring
        final_results = []
        for beam in beams:
            solution = beam['solution'][len(problem):].strip()
            final_score = self.reward_fn(problem, solution)
            
            final_results.append({
                'solution': solution,
                'beam_score': beam['score'],
                'final_score': final_score,
                'length': len(beam['tokens'])
            })
        
        return sorted(final_results, key=lambda x: x['final_score'], reverse=True)
    
    def _evaluate_partial_solution(self, problem: str, partial: str) -> float:
        """Evaluate quality of partial solution."""
        # Simple heuristics for partial evaluation
        partial_lower = partial.lower()
        
        # Check for common reasoning patterns
        has_reasoning = any(word in partial_lower for word in 
                          ['because', 'therefore', 'since', 'if', 'then'])
        
        # Check for structure
        has_structure = '\n' in partial or any(marker in partial_lower for marker in 
                                             ['step', 'first', 'next', 'finally'])
        
        # Length appropriateness
        words = partial.split()
        appropriate_length = 10 < len(words) < 1000
        
        # Compute score
        score = 0.3 * has_reasoning + 0.3 * has_structure + 0.4 * appropriate_length
        
        return score