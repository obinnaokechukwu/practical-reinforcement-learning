"""
Complete multi-stage RL pipeline for reasoning model development.
"""

import torch
from dataclasses import dataclass
from typing import List, Dict, Optional, Callable
import json
import os
from tqdm import tqdm
import numpy as np

from .pure_rl_trainer import PureRLTrainer, PureRLConfig
from .cot_emergence import ChainOfThoughtTrainer, ChainOfThoughtAnalyzer, CoTConfig
from .rejection_sampling import RejectionSampler, DiversityAwareSampler, RejectionSamplingConfig
from .distillation import ReasoningDistiller, ProgressiveDistiller, DistillationConfig
from .process_rewards import ProcessRewardModel, PRMDataGenerator, PRMTrainer, ProcessRewardEvaluator, PRMConfig


@dataclass
class ReasoningPipelineConfig:
    """Configuration for complete reasoning pipeline."""
    # Base model
    base_model_name: str = "EleutherAI/gpt-neo-125M"
    
    # Pipeline stages
    enable_pure_rl: bool = True
    enable_cot_training: bool = True
    enable_rejection_sampling: bool = True
    enable_prm_training: bool = True
    enable_distillation: bool = True
    
    # Stage-specific configs
    pure_rl_iterations: int = 500
    cot_iterations: int = 300
    rejection_samples: int = 16
    prm_epochs: int = 3
    
    # Output settings
    output_dir: str = "./reasoning_pipeline_output"
    save_checkpoints: bool = True
    checkpoint_interval: int = 100
    
    # Evaluation
    num_test_problems: int = 50
    evaluation_temperature: float = 0.7
    
    # Hardware
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_fp16: bool = True


class ReasoningModelPipeline:
    """Orchestrates the complete multi-stage reasoning model development."""
    
    def __init__(self, config: ReasoningPipelineConfig, reward_fn: Callable):
        self.config = config
        self.reward_fn = reward_fn
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Track pipeline state
        self.pipeline_state = {
            'current_stage': 'initialization',
            'stages_completed': [],
            'best_model_path': None,
            'metrics': {}
        }
        
        # Initialize components
        self.models = {}
        self.trainers = {}
    
    def run_pipeline(self, 
                    train_problems: List[str],
                    test_problems: List[str],
                    student_model_name: Optional[str] = None):
        """Run the complete reasoning model pipeline."""
        
        print("=" * 80)
        print("REASONING MODEL DEVELOPMENT PIPELINE")
        print("=" * 80)
        
        # Stage 1: Pure RL Training (Optional)
        if self.config.enable_pure_rl:
            print("\n🚀 Stage 1: Pure RL Training")
            self._run_pure_rl_stage(train_problems, test_problems)
        
        # Stage 2: Chain-of-Thought Training
        if self.config.enable_cot_training:
            print("\n🧠 Stage 2: Chain-of-Thought Emergence")
            self._run_cot_stage(train_problems, test_problems)
        
        # Stage 3: Rejection Sampling & Refinement
        if self.config.enable_rejection_sampling:
            print("\n🎯 Stage 3: Rejection Sampling & Refinement")
            self._run_rejection_sampling_stage(train_problems, test_problems)
        
        # Stage 4: Process Reward Model Training
        if self.config.enable_prm_training:
            print("\n📊 Stage 4: Process Reward Model Training")
            self._run_prm_training_stage(train_problems, test_problems)
        
        # Stage 5: Knowledge Distillation
        if self.config.enable_distillation and student_model_name:
            print("\n🎓 Stage 5: Knowledge Distillation")
            self._run_distillation_stage(train_problems, test_problems, student_model_name)
        
        # Final evaluation and report
        print("\n📝 Generating Final Report")
        report = self._generate_final_report(test_problems)
        
        # Save report
        report_path = os.path.join(self.config.output_dir, "pipeline_report.md")
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"\n✅ Pipeline Complete! Report saved to: {report_path}")
        
        return self.pipeline_state
    
    def _run_pure_rl_stage(self, train_problems: List[str], test_problems: List[str]):
        """Stage 1: Pure RL training without SFT."""
        self.pipeline_state['current_stage'] = 'pure_rl'
        
        # Configure pure RL
        pure_rl_config = PureRLConfig(
            model_name=self.config.base_model_name,
            learning_rate=1e-5,
            temperature_start=1.2,
            temperature_end=0.3,
            use_wandb=False
        )
        
        # Initialize trainer
        pure_rl_trainer = PureRLTrainer(pure_rl_config, self.reward_fn)
        self.trainers['pure_rl'] = pure_rl_trainer
        
        # Train
        print("Training with pure RL (no SFT)...")
        pure_rl_trainer.train(train_problems, num_iterations=self.config.pure_rl_iterations)
        
        # Save checkpoint
        if self.config.save_checkpoints:
            checkpoint_path = os.path.join(self.config.output_dir, "pure_rl_checkpoint")
            pure_rl_trainer.save_checkpoint(checkpoint_path)
            self.models['pure_rl'] = checkpoint_path
        
        # Evaluate
        metrics = self._evaluate_model(pure_rl_trainer.model, test_problems[:10], "Pure RL")
        self.pipeline_state['metrics']['pure_rl'] = metrics
        self.pipeline_state['stages_completed'].append('pure_rl')
    
    def _run_cot_stage(self, train_problems: List[str], test_problems: List[str]):
        """Stage 2: Train for Chain-of-Thought emergence."""
        self.pipeline_state['current_stage'] = 'cot_training'
        
        # Use best model so far
        base_model = self.models.get('pure_rl', self.config.base_model_name)
        
        # Configure CoT training
        cot_config = CoTConfig(
            model_name=base_model if isinstance(base_model, str) else self.config.base_model_name,
            thinking_reward_weight=0.3,
            answer_reward_weight=0.7,
            force_thinking_steps=3
        )
        
        # Initialize trainer
        cot_trainer = ChainOfThoughtTrainer(cot_config, self.reward_fn)
        self.trainers['cot'] = cot_trainer
        
        # Train
        print("Training for Chain-of-Thought emergence...")
        cot_trainer.train_with_cot_rewards(train_problems, num_iterations=self.config.cot_iterations)
        
        # Analyze CoT patterns
        analyzer = ChainOfThoughtAnalyzer(cot_trainer.model, cot_trainer.tokenizer)
        cot_report = analyzer.generate_cot_report(test_problems[:10], cot_trainer.model)
        
        # Save
        if self.config.save_checkpoints:
            checkpoint_path = os.path.join(self.config.output_dir, "cot_checkpoint")
            cot_trainer.model.save_pretrained(checkpoint_path)
            cot_trainer.tokenizer.save_pretrained(checkpoint_path)
            self.models['cot'] = checkpoint_path
            
            # Save CoT analysis
            with open(os.path.join(self.config.output_dir, "cot_analysis.txt"), 'w') as f:
                f.write(cot_report)
        
        # Evaluate
        metrics = self._evaluate_model(cot_trainer.model, test_problems[:10], "CoT")
        self.pipeline_state['metrics']['cot'] = metrics
        self.pipeline_state['stages_completed'].append('cot_training')
    
    def _run_rejection_sampling_stage(self, train_problems: List[str], test_problems: List[str]):
        """Stage 3: Rejection sampling and refinement."""
        self.pipeline_state['current_stage'] = 'rejection_sampling'
        
        # Load best model
        best_model_path = self.models.get('cot', self.models.get('pure_rl', self.config.base_model_name))
        
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(best_model_path)
        tokenizer = AutoTokenizer.from_pretrained(best_model_path)
        
        # Configure rejection sampling
        rejection_config = RejectionSamplingConfig(
            num_samples=self.config.rejection_samples,
            temperature=0.8,
            min_score_threshold=0.7
        )
        
        # Initialize samplers
        basic_sampler = RejectionSampler(model, tokenizer, self.reward_fn, rejection_config)
        diversity_sampler = DiversityAwareSampler(model, tokenizer, self.reward_fn, rejection_config)
        
        # Generate high-quality samples
        print("Generating diverse high-quality solutions...")
        high_quality_solutions = []
        
        for problem in tqdm(train_problems[:100], desc="Rejection sampling"):
            # Try diversity-aware sampling
            diverse_solutions = diversity_sampler.sample_diverse_solutions(problem, num_diverse=3)
            
            if diverse_solutions:
                # Create ensemble
                ensemble = diversity_sampler.ensemble_solutions(problem, diverse_solutions)
                high_quality_solutions.append({
                    'problem': problem,
                    'solution': ensemble['ensemble_solution'],
                    'score': ensemble['ensemble_score'],
                    'method': 'ensemble'
                })
            else:
                # Fallback to basic sampling
                result = basic_sampler.sample_with_rejection(problem)
                if result:
                    high_quality_solutions.append({
                        'problem': problem,
                        'solution': result['solution'],
                        'score': result['score'],
                        'method': 'basic'
                    })
        
        # Save high-quality solutions
        solutions_path = os.path.join(self.config.output_dir, "high_quality_solutions.json")
        with open(solutions_path, 'w') as f:
            json.dump(high_quality_solutions, f, indent=2)
        
        # Get sampling statistics
        sampling_report = basic_sampler.get_sampling_report()
        with open(os.path.join(self.config.output_dir, "sampling_report.txt"), 'w') as f:
            f.write(sampling_report)
        
        self.pipeline_state['metrics']['rejection_sampling'] = {
            'num_high_quality': len(high_quality_solutions),
            'avg_score': np.mean([s['score'] for s in high_quality_solutions])
        }
        self.pipeline_state['stages_completed'].append('rejection_sampling')
    
    def _run_prm_training_stage(self, train_problems: List[str], test_problems: List[str]):
        """Stage 4: Train Process Reward Model."""
        self.pipeline_state['current_stage'] = 'prm_training'
        
        # Load high-quality solutions
        solutions_path = os.path.join(self.config.output_dir, "high_quality_solutions.json")
        if os.path.exists(solutions_path):
            with open(solutions_path, 'r') as f:
                solutions_data = json.load(f)
        else:
            print("No high-quality solutions found, skipping PRM training")
            return
        
        # Configure PRM
        prm_config = PRMConfig(
            base_model_name="microsoft/deberta-v3-base",
            num_epochs=self.config.prm_epochs
        )
        
        # Generate PRM training data
        print("Generating PRM training data...")
        prm_generator = PRMDataGenerator(self.reward_fn, prm_config)
        prm_data = prm_generator.generate_prm_dataset(solutions_data[:200])
        
        # Split data
        split_idx = int(len(prm_data) * 0.9)
        train_data = prm_data[:split_idx]
        val_data = prm_data[split_idx:]
        
        # Train PRM
        print(f"Training PRM on {len(train_data)} examples...")
        prm_trainer = PRMTrainer(prm_config)
        prm_trainer.train(train_data, val_data)
        
        # Save PRM
        if self.config.save_checkpoints:
            prm_path = os.path.join(self.config.output_dir, "prm_model.pt")
            torch.save({
                'model_state': prm_trainer.model.state_dict(),
                'config': prm_config
            }, prm_path)
            self.models['prm'] = prm_path
        
        # Evaluate PRM
        prm_evaluator = ProcessRewardEvaluator(prm_trainer.model, prm_trainer.tokenizer)
        
        # Test on some examples
        test_scores = []
        for item in solutions_data[:10]:
            eval_result = prm_evaluator.evaluate_solution(item['problem'], item['solution'])
            test_scores.append(eval_result['overall_score'])
        
        self.pipeline_state['metrics']['prm'] = {
            'avg_prm_score': np.mean(test_scores),
            'val_metrics': prm_trainer.evaluate(val_data)
        }
        self.pipeline_state['stages_completed'].append('prm_training')
    
    def _run_distillation_stage(self, train_problems: List[str], test_problems: List[str], 
                               student_model_name: str):
        """Stage 5: Distill to smaller model."""
        self.pipeline_state['current_stage'] = 'distillation'
        
        # Get teacher model path
        teacher_path = self.models.get('cot', self.models.get('pure_rl', self.config.base_model_name))
        
        # Configure distillation
        distill_config = DistillationConfig(
            teacher_model_name=teacher_path,
            student_model_name=student_model_name,
            temperature=3.0,
            alpha=0.7,
            num_epochs=3
        )
        
        # Run progressive distillation
        print("Running progressive distillation...")
        distiller = ProgressiveDistiller(distill_config)
        distiller.progressive_distillation(train_problems, test_problems[:20])
        
        # Save distilled model
        distilled_path = os.path.join(self.config.output_dir, "distilled_model")
        distiller.student_model.save_pretrained(distilled_path)
        distiller.student_tokenizer.save_pretrained(distilled_path)
        self.models['distilled'] = distilled_path
        
        # Evaluate distilled model
        metrics = self._evaluate_model(distiller.student_model, test_problems[:10], "Distilled")
        self.pipeline_state['metrics']['distillation'] = metrics
        self.pipeline_state['stages_completed'].append('distillation')
    
    def _evaluate_model(self, model, test_problems: List[str], model_name: str) -> Dict:
        """Evaluate a model on test problems."""
        from transformers import AutoTokenizer
        
        # Get tokenizer
        if hasattr(model, 'name_or_path'):
            tokenizer = AutoTokenizer.from_pretrained(model.name_or_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.config.base_model_name)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        scores = []
        solution_lengths = []
        
        print(f"Evaluating {model_name} model...")
        for problem in tqdm(test_problems, desc="Evaluation"):
            # Generate solution
            inputs = tokenizer(problem, return_tensors='pt', truncation=True)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=1024,
                    temperature=self.config.evaluation_temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            solution = tokenizer.decode(outputs[0], skip_special_tokens=True)
            solution = solution[len(problem):].strip()
            
            # Evaluate
            score = self.reward_fn(problem, solution)
            scores.append(score)
            solution_lengths.append(len(solution.split()))
        
        return {
            'avg_score': np.mean(scores),
            'std_score': np.std(scores),
            'avg_length': np.mean(solution_lengths),
            'num_evaluated': len(scores)
        }
    
    def _generate_final_report(self, test_problems: List[str]) -> str:
        """Generate comprehensive pipeline report."""
        report = f"""
# Reasoning Model Development Pipeline Report

## Pipeline Configuration
- Base Model: {self.config.base_model_name}
- Stages Completed: {', '.join(self.pipeline_state['stages_completed'])}
- Output Directory: {self.config.output_dir}

## Stage Results

"""
        
        # Add metrics for each stage
        for stage, metrics in self.pipeline_state['metrics'].items():
            report += f"### {stage.upper().replace('_', ' ')}\n"
            
            if isinstance(metrics, dict):
                for key, value in metrics.items():
                    if isinstance(value, float):
                        report += f"- {key}: {value:.3f}\n"
                    else:
                        report += f"- {key}: {value}\n"
            else:
                report += f"- Result: {metrics}\n"
            
            report += "\n"
        
        # Add recommendations
        report += """
## Recommendations

1. **Best Model**: """
        
        # Find best performing model
        best_stage = max(
            self.pipeline_state['metrics'].items(),
            key=lambda x: x[1].get('avg_score', 0) if isinstance(x[1], dict) else 0
        )[0]
        
        report += f"The {best_stage} model achieved the best performance.\n"
        
        if 'distillation' in self.pipeline_state['stages_completed']:
            report += "2. **Deployment**: The distilled model offers the best balance of performance and efficiency.\n"
        
        if 'prm' in self.pipeline_state['stages_completed']:
            report += "3. **Quality Control**: Use the trained PRM for online quality assessment during inference.\n"
        
        report += """
## Next Steps

1. Fine-tune on domain-specific problems
2. Implement online learning with human feedback
3. Deploy with appropriate safety filters
4. Monitor performance in production

---
Generated by Reasoning Model Pipeline
"""
        
        return report


def create_example_reward_function():
    """Create an example reward function for demonstrations."""
    
    def reward_fn(problem: str, solution: str) -> float:
        """Simple reward function for mathematical reasoning."""
        score = 0.0
        
        # Check if solution is non-empty
        if not solution or len(solution.strip()) < 10:
            return 0.0
        
        # Reward for structure
        if any(marker in solution.lower() for marker in ['step', 'first', 'then', 'finally']):
            score += 0.2
        
        # Reward for mathematical content
        if any(op in solution for op in ['=', '+', '-', '*', '/']):
            score += 0.2
        
        # Reward for reasoning words
        reasoning_words = ['because', 'therefore', 'since', 'if', 'then', 'thus']
        reasoning_count = sum(1 for word in reasoning_words if word in solution.lower())
        score += min(reasoning_count * 0.1, 0.3)
        
        # Reward for appropriate length
        length = len(solution.split())
        if 20 < length < 200:
            score += 0.2
        
        # Check for common errors (penalties)
        if 'error' in solution.lower() or 'mistake' in solution.lower():
            score -= 0.3
        
        return max(0.0, min(1.0, score))
    
    return reward_fn


if __name__ == "__main__":
    # Example usage
    config = ReasoningPipelineConfig(
        base_model_name="EleutherAI/gpt-neo-125M",
        enable_pure_rl=True,
        enable_cot_training=True,
        enable_rejection_sampling=True,
        enable_prm_training=True,
        enable_distillation=True,
        pure_rl_iterations=10,  # Reduced for demo
        cot_iterations=10,      # Reduced for demo
        output_dir="./demo_pipeline_output"
    )
    
    # Example problems
    train_problems = [
        "What is 15% of 80?",
        "If a train travels 60 miles in 1.5 hours, what is its average speed?",
        "Solve for x: 2x + 5 = 13",
        "A rectangle has length 8 and width 5. What is its area?",
        "What is the sum of the first 10 positive integers?"
    ]
    
    test_problems = [
        "What is 20% of 150?",
        "If a car travels 120 miles in 2 hours, what is its average speed?",
        "Solve for y: 3y - 7 = 14"
    ]
    
    # Create reward function
    reward_fn = create_example_reward_function()
    
    # Run pipeline
    pipeline = ReasoningModelPipeline(config, reward_fn)
    
    print("Note: This is a demonstration with reduced iterations.")
    print("For real training, use larger models and more iterations.")
    
    # Uncomment to run:
    # results = pipeline.run_pipeline(
    #     train_problems,
    #     test_problems,
    #     student_model_name="microsoft/DialoGPT-small"
    # )