"""
Multi-stage training pipeline for self-improving agents.
"""

import os
import json
from typing import List, Dict, Optional, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    Trainer, TrainingArguments,
    DataCollatorForLanguageModeling
)
import numpy as np
from tqdm import tqdm

from .generator import ConstitutionalCodeGenerator
from .evaluator import SelfEvaluator
from .constitutional import ConstitutionalPrinciples


class CodeGenerationDataset(Dataset):
    """Dataset for code generation tasks."""
    
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 1024):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Format as instruction-following
        if 'solution' in item:
            # Supervised learning format
            text = f"Problem: {item['problem']}\n\nSolution:\n```python\n{item['solution']}\n```"
        else:
            # Problem only format
            text = f"Problem: {item['problem']}\n\nSolution:\n```python"
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze()
        }


class MultiStageTrainer:
    """Multi-stage training pipeline for constitutional agents."""
    
    def __init__(self, 
                 base_model_name: str,
                 constitution: Optional[List[str]] = None,
                 device: Optional[str] = None):
        """
        Initialize multi-stage trainer.
        
        Args:
            base_model_name: Name or path of base model
            constitution: Constitutional principles to follow
            device: Device to use for training
        """
        self.base_model_name = base_model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set up constitution
        if constitution is None:
            self.constitutional_principles = ConstitutionalPrinciples('coding')
            self.constitution = self.constitutional_principles.principles
        else:
            self.constitution = constitution
            self.constitutional_principles = ConstitutionalPrinciples(
                'coding', custom_principles=constitution
            )
        
        # Initialize components
        self.generator = None
        self.evaluator = SelfEvaluator()
        self.training_history = []
    
    def train_full_pipeline(self,
                           demonstration_data: List[Dict],
                           rl_problems: List[str],
                           distillation_problems: List[str],
                           output_dir: str,
                           student_model_name: Optional[str] = None):
        """
        Execute full three-stage training pipeline.
        
        Args:
            demonstration_data: Data for cold start (with solutions)
            rl_problems: Problems for RL optimization (no solutions)
            distillation_problems: Problems for distillation
            output_dir: Directory to save models
            student_model_name: Optional smaller model for distillation
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Stage 1: Cold Start
        print("=" * 50)
        print("Stage 1: Cold Start Training")
        print("=" * 50)
        
        stage1_dir = os.path.join(output_dir, "stage1_cold_start")
        self.cold_start_training(demonstration_data, stage1_dir)
        
        # Stage 2: RL Optimization
        print("\n" + "=" * 50)
        print("Stage 2: RL Optimization")
        print("=" * 50)
        
        stage2_dir = os.path.join(output_dir, "stage2_rl_optimization")
        self.rl_optimization(rl_problems, stage2_dir)
        
        # Stage 3: Knowledge Distillation
        if student_model_name:
            print("\n" + "=" * 50)
            print("Stage 3: Knowledge Distillation")
            print("=" * 50)
            
            stage3_dir = os.path.join(output_dir, "stage3_distillation")
            self.knowledge_distillation(
                distillation_problems, 
                stage3_dir,
                student_model_name
            )
        
        # Save training summary
        self.save_training_summary(output_dir)
        
        print("\n" + "=" * 50)
        print("Training Complete!")
        print("=" * 50)
    
    def cold_start_training(self, 
                           demonstrations: List[Dict],
                           output_dir: str,
                           num_epochs: int = 3):
        """
        Stage 1: Supervised learning on demonstrations.
        
        Args:
            demonstrations: List of problem-solution pairs
            output_dir: Directory to save model
            num_epochs: Number of training epochs
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize model and tokenizer
        print("Loading base model...")
        model = AutoModelForCausalLM.from_pretrained(self.base_model_name)
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Create dataset
        print(f"Creating dataset from {len(demonstrations)} demonstrations...")
        dataset = CodeGenerationDataset(demonstrations, tokenizer)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,
            warmup_steps=100,
            logging_steps=50,
            save_steps=500,
            save_total_limit=2,
            learning_rate=5e-5,
            logging_dir=os.path.join(output_dir, 'logs'),
            report_to='none',  # Disable wandb/tensorboard for simplicity
            fp16=self.device == 'cuda',
            push_to_hub=False,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
        )
        
        # Train
        print("Starting cold start training...")
        trainer.train()
        
        # Save model
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)
        
        # Initialize generator with trained model
        self.generator = ConstitutionalCodeGenerator(
            output_dir, self.constitution, self.device
        )
        
        # Evaluate on sample
        self._evaluate_stage(demonstrations[:5], "Cold Start")
        
        # Record in history
        self.training_history.append({
            'stage': 'cold_start',
            'output_dir': output_dir,
            'num_demonstrations': len(demonstrations),
            'num_epochs': num_epochs
        })
    
    def rl_optimization(self,
                       problems: List[str],
                       output_dir: str,
                       num_iterations: int = 5,
                       samples_per_iteration: int = 100):
        """
        Stage 2: RL-based optimization using GRPO.
        
        Args:
            problems: List of problems (no solutions)
            output_dir: Directory to save model
            num_iterations: Number of RL iterations
            samples_per_iteration: Problems to sample each iteration
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if self.generator is None:
            raise ValueError("Must run cold start training first")
        
        # We'll use a simplified GRPO-style approach here
        print(f"Starting RL optimization with {len(problems)} problems...")
        
        from rewards.code_rewards import CodeRewardFunction
        reward_function = CodeRewardFunction()
        
        all_experiences = []
        
        for iteration in range(num_iterations):
            print(f"\nRL Iteration {iteration + 1}/{num_iterations}")
            
            # Sample problems
            iteration_problems = np.random.choice(
                problems, 
                min(samples_per_iteration, len(problems)),
                replace=False
            )
            
            iteration_rewards = []
            good_solutions = []
            
            # Generate solutions and evaluate
            for problem in tqdm(iteration_problems, desc="Generating"):
                # Generate multiple solutions per problem (group-based)
                group_size = 4
                solutions = []
                rewards = []
                
                for _ in range(group_size):
                    result = self.generator.generate_code(
                        problem, 
                        max_revisions=2,
                        temperature=0.7 + 0.1 * np.random.randn()  # Vary temperature
                    )
                    
                    solution = result['final_code']
                    
                    # Evaluate solution
                    eval_result = reward_function.evaluate(problem, solution)
                    reward = eval_result['total']
                    
                    solutions.append(solution)
                    rewards.append(reward)
                
                # Store best solution if good enough
                best_idx = np.argmax(rewards)
                if rewards[best_idx] > 0.7:
                    good_solutions.append({
                        'problem': problem,
                        'solution': solutions[best_idx],
                        'reward': rewards[best_idx]
                    })
                
                iteration_rewards.extend(rewards)
            
            # Print iteration statistics
            avg_reward = np.mean(iteration_rewards)
            print(f"Average reward: {avg_reward:.3f}")
            print(f"Good solutions found: {len(good_solutions)}")
            
            all_experiences.extend(good_solutions)
            
            # Fine-tune on good solutions if we have enough
            if len(good_solutions) >= 20:
                print("Fine-tuning on high-reward solutions...")
                self._finetune_on_solutions(good_solutions, output_dir, iteration)
        
        # Save final model
        self.generator.model.save_pretrained(output_dir)
        self.generator.tokenizer.save_pretrained(output_dir)
        
        # Save experiences
        with open(os.path.join(output_dir, 'rl_experiences.json'), 'w') as f:
            json.dump(all_experiences, f, indent=2)
        
        # Record in history
        self.training_history.append({
            'stage': 'rl_optimization',
            'output_dir': output_dir,
            'num_problems': len(problems),
            'num_iterations': num_iterations,
            'total_good_solutions': len(all_experiences)
        })
    
    def knowledge_distillation(self,
                              problems: List[str],
                              output_dir: str,
                              student_model_name: str,
                              num_epochs: int = 2):
        """
        Stage 3: Distill to smaller model.
        
        Args:
            problems: Problems for generating distillation data
            output_dir: Directory to save student model
            student_model_name: Name of smaller model
            num_epochs: Number of distillation epochs
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if self.generator is None:
            raise ValueError("Must run RL optimization first")
        
        print(f"Generating teacher solutions for {len(problems)} problems...")
        
        # Generate high-quality solutions from teacher
        teacher_data = []
        
        for problem in tqdm(problems, desc="Teacher generation"):
            # Use low temperature for consistent, high-quality output
            result = self.generator.generate_code(
                problem,
                max_revisions=3,
                temperature=0.3
            )
            
            teacher_data.append({
                'problem': problem,
                'solution': result['final_code']
            })
        
        # Initialize student model
        print(f"Initializing student model: {student_model_name}")
        student_model = AutoModelForCausalLM.from_pretrained(student_model_name)
        student_tokenizer = AutoTokenizer.from_pretrained(student_model_name)
        
        if student_tokenizer.pad_token is None:
            student_tokenizer.pad_token = student_tokenizer.eos_token
        
        # Create dataset
        dataset = CodeGenerationDataset(teacher_data, student_tokenizer)
        
        # Training arguments for distillation
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=8,
            gradient_accumulation_steps=1,
            warmup_steps=50,
            logging_steps=20,
            save_steps=200,
            save_total_limit=2,
            learning_rate=1e-4,
            logging_dir=os.path.join(output_dir, 'logs'),
            report_to='none',
            fp16=self.device == 'cuda',
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=student_tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )
        
        # Create trainer
        trainer = Trainer(
            model=student_model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
        )
        
        # Train student
        print("Training student model...")
        trainer.train()
        
        # Save student model
        trainer.save_model()
        student_tokenizer.save_pretrained(output_dir)
        
        # Evaluate student
        student_generator = ConstitutionalCodeGenerator(
            output_dir, self.constitution, self.device
        )
        self._evaluate_stage(
            [{'problem': p} for p in problems[:5]], 
            "Distilled Student",
            generator=student_generator
        )
        
        # Record in history
        self.training_history.append({
            'stage': 'distillation',
            'output_dir': output_dir,
            'teacher_model': self.generator.model.config.name_or_path,
            'student_model': student_model_name,
            'num_problems': len(problems),
            'num_epochs': num_epochs
        })
    
    def _finetune_on_solutions(self, 
                              solutions: List[Dict],
                              output_dir: str,
                              iteration: int):
        """Fine-tune model on high-reward solutions."""
        # Create temporary dataset
        dataset = CodeGenerationDataset(solutions, self.generator.tokenizer)
        
        # Quick fine-tuning
        training_args = TrainingArguments(
            output_dir=os.path.join(output_dir, f'iter_{iteration}'),
            overwrite_output_dir=True,
            num_train_epochs=1,
            per_device_train_batch_size=4,
            logging_steps=10,
            save_steps=1000,
            learning_rate=1e-5,
            report_to='none',
            fp16=self.device == 'cuda',
        )
        
        trainer = Trainer(
            model=self.generator.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=self.generator.tokenizer,
                mlm=False
            ),
        )
        
        trainer.train()
    
    def _evaluate_stage(self, 
                       test_data: List[Dict],
                       stage_name: str,
                       generator: Optional[ConstitutionalCodeGenerator] = None):
        """Evaluate model performance at a stage."""
        if generator is None:
            generator = self.generator
        
        print(f"\nEvaluating {stage_name}...")
        
        scores = []
        for item in test_data:
            problem = item['problem']
            
            # Generate solution
            result = generator.generate_code(problem, max_revisions=2)
            solution = result['final_code']
            
            # Evaluate
            eval_result = self.evaluator.evaluate_code(problem, solution)
            scores.append(eval_result['overall_score'])
            
            # Check constitutional compliance
            compliance = self.constitutional_principles.evaluate_compliance(solution)
            
        avg_score = np.mean(scores) if scores else 0.0
        print(f"{stage_name} Average Score: {avg_score:.3f}")
    
    def save_training_summary(self, output_dir: str):
        """Save summary of training process."""
        summary = {
            'constitution': self.constitution,
            'training_history': self.training_history,
            'evaluation_summary': self.evaluator.get_evaluation_summary()
        }
        
        with open(os.path.join(output_dir, 'training_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Also save human-readable report
        report = self._generate_training_report()
        with open(os.path.join(output_dir, 'training_report.txt'), 'w') as f:
            f.write(report)
    
    def _generate_training_report(self) -> str:
        """Generate human-readable training report."""
        report = "Multi-Stage Training Report\n"
        report += "=" * 50 + "\n\n"
        
        # Constitution
        report += "Constitutional Principles:\n"
        for i, principle in enumerate(self.constitution, 1):
            report += f"{i}. {principle}\n"
        report += "\n"
        
        # Training stages
        report += "Training Stages:\n"
        report += "-" * 30 + "\n"
        
        for stage in self.training_history:
            report += f"\nStage: {stage['stage']}\n"
            report += f"Output: {stage['output_dir']}\n"
            
            for key, value in stage.items():
                if key not in ['stage', 'output_dir']:
                    report += f"  {key}: {value}\n"
        
        # Evaluation summary
        eval_summary = self.evaluator.get_evaluation_summary()
        if eval_summary['total_evaluations'] > 0:
            report += "\nEvaluation Summary:\n"
            report += "-" * 30 + "\n"
            for key, value in eval_summary.items():
                if isinstance(value, float):
                    report += f"{key}: {value:.3f}\n"
                else:
                    report += f"{key}: {value}\n"
        
        return report