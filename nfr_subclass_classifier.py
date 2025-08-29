"""
NFR Subclassification Module for Requirements Engineering AI Workflow
=====================================================================

This module provides LLM-based classification of Non-Functional Requirements (NFRs)
into four subcategories: Operational (O), Performance (PE), Security (SE), and Usability (US).

Based on academic literature:
- ISO/IEC 25010:2023 - Systems and software Quality Requirements and Evaluation
- IEEE Standards for Requirements Engineering
- Lim, S.C. (2022) - Task-Adaptive Pretraining for Requirements Classification

Author: Alessandro Stefanone
Affiliation: Politecnico di Milano - PhD in Mechanical Engineering
Research Focus: AI applications to industrial engineering, digital twins, PLM
"""

import json
import time
import logging
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# LangChain imports for LLM integration
try:
    from langchain_anthropic import ChatAnthropic
    from langchain_core.prompts import ChatPromptTemplate
    from pydantic import BaseModel, Field
    from langchain_core.output_parsers import PydanticOutputParser
    LANGCHAIN_AVAILABLE = True
except ImportError:
    print("LangChain not available. Please install: pip install langchain langchain-anthropic")
    LANGCHAIN_AVAILABLE = False

# Dataset loading
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    print("Datasets library not available. Please install: pip install datasets")
    DATASETS_AVAILABLE = False


@dataclass
class NFRSubclassResult:
    """Result structure for NFR subclassification"""
    requirement_text: str
    predicted_label: str
    confidence: float
    reasoning: str
    key_indicators: List[str]
    processing_time: float


class NFRSubclassificationPrompt(BaseModel):
    """Pydantic model for structured NFR subclassification output"""
    requirement_text: str = Field(description="The original requirement text")
    classification: str = Field(description="One of: O, PE, SE, US")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0")
    reasoning: str = Field(description="Brief explanation of the classification decision")
    key_indicators: List[str] = Field(description="Key words or phrases that influenced the decision")

    class Config:
        """Pydantic v2 configuration"""
        json_schema_extra = {
            "example": {
                "requirement_text": "The system shall respond within 2 seconds",
                "classification": "PE",
                "confidence": 0.95,
                "reasoning": "Performance-related constraint",
                "key_indicators": ["respond", "2 seconds"]
            }
        }


class LLMNFRSubclassifier:
    """
    LLM-based classifier for NFR subclassification into four categories:
    - Operational (O): System operation, monitoring, management requirements
    - Performance (PE): Speed, throughput, response time, scalability requirements  
    - Security (SE): Data protection, access control, authentication requirements
    - Usability (US): User interface, ease of use, accessibility requirements
    """
    
    def __init__(
        self, 
        model_name: str = "claude-3-5-haiku-20241022",
        api_key: Optional[str] = None,
        use_few_shot: bool = True,
        temperature: float = 0.0
    ):
        """
        Initialize the NFR subclassifier
        
        Args:
            model_name: Name of the Claude model to use
            api_key: Anthropic API key (if not provided, will use environment variable)
            use_few_shot: Whether to use few-shot learning with examples
            temperature: Temperature for LLM generation (0.0 for deterministic)
        """
        self.model_name = model_name
        self.use_few_shot = use_few_shot
        self.temperature = temperature
        
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain is required. Please install: pip install langchain langchain-anthropic")
        
        # Initialize LLM
        self.llm = ChatAnthropic(
            model=model_name,
            api_key=api_key,
            temperature=temperature
        )
        
        # Setup output parser
        self.output_parser = PydanticOutputParser(pydantic_object=NFRSubclassificationPrompt)
        
        # Create prompt template
        self.prompt_template = self._create_prompt_template()
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('nfr_subclassification.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _create_prompt_template(self) -> ChatPromptTemplate:
        """Create the prompt template for NFR subclassification"""
        
        # Classification rules based on ISO/IEC 25010:2023 and IEEE standards
        classification_rules = """
NFR SUBCLASSIFICATION RULES (Based on ISO/IEC 25010:2023 and IEEE Standards):

1. OPERATIONAL (O) - System Operation & Management:
   - System monitoring, logging, and management capabilities
   - Deployment, installation, and configuration requirements
   - System administration and maintenance operations
   - Backup, recovery, and operational procedures
   - Key indicators: "monitor", "log", "deploy", "install", "configure", "administer", "backup", "recovery"

2. PERFORMANCE (PE) - System Performance & Efficiency:
   - Response time, throughput, and speed requirements
   - Resource utilization (CPU, memory, disk, network)
   - Scalability and load handling capabilities
   - Capacity and volume requirements
   - Key indicators: "response time", "throughput", "speed", "performance", "scalability", "load", "capacity"

3. SECURITY (SE) - Information Security & Protection:
   - Authentication and authorization mechanisms
   - Data protection, encryption, and privacy
   - Access control and user permissions
   - Security protocols and compliance requirements
   - Key indicators: "security", "authentication", "authorization", "encryption", "privacy", "access control"

4. USABILITY (US) - User Experience & Interface:
   - User interface design and interaction
   - Ease of use and learnability
   - Accessibility and user experience requirements
   - User documentation and help systems
   - Key indicators: "user interface", "usability", "ease of use", "accessibility", "user experience", "intuitive"
"""
        
        # Few-shot examples if enabled
        few_shot_examples = ""
        if self.use_few_shot:
            few_shot_examples = """
EXAMPLES:

Example 1:
Requirement: "The system shall respond to user queries within 2 seconds under normal load conditions."
Classification: PE
Reasoning: This requirement specifies a response time constraint, which is a performance characteristic.
Key Indicators: ["respond", "2 seconds", "load conditions"]

Example 2:
Requirement: "All user authentication must use multi-factor authentication with encrypted tokens."
Classification: SE
Reasoning: This requirement focuses on authentication mechanisms and encryption for security.
Key Indicators: ["authentication", "multi-factor", "encrypted tokens"]

Example 3:
Requirement: "The user interface shall be intuitive for users with minimal training required."
Classification: US
Reasoning: This requirement addresses user interface design and ease of use.
Key Indicators: ["user interface", "intuitive", "minimal training"]

Example 4:
Requirement: "The system shall provide comprehensive logging of all user actions for audit purposes."
Classification: O
Reasoning: This requirement specifies logging and auditing capabilities for operational management.
Key Indicators: ["logging", "user actions", "audit purposes"]

"""

        prompt_text = f"""
You are an expert requirements engineer specializing in software quality characteristics and NFR classification.

You are given an input which is a Non-Functional Requirement (NFR) statement. Your task is to classify the input requirement into exactly one of these categories:
- O (Operational)
- PE (Performance) 
- SE (Security)
- US (Usability)

You are not allowed to create new categories or modify existing ones. 
Your classification must be based solely on the content of the requirement statement and the provided classification rules.

INSTRUCTIONS:
1. Analyze the requirement text carefully
2. Identify key indicators that suggest the classification
3. Apply the classification rules above
4. Provide your reasoning for the decision
5. Express confidence based on clarity of indicators

<classification_rules>
{classification_rules}
</classification_rules>

<few_shot_examples>
{few_shot_examples}
</few_shot_examples>

<requirement_to_classify>
"{{requirement_text}}"
</requirement_to_classify>

{{format_instructions}}

Your response must be valid JSON only. Do not include any text before or after the JSON.
"""

        return ChatPromptTemplate.from_template(prompt_text)
    
    def classify_requirement(self, requirement_text: str) -> NFRSubclassResult:
        """
        Classify a single NFR into one of four subclasses
        
        Args:
            requirement_text: The NFR text to classify
            
        Returns:
            NFRSubclassResult with classification details
        """
        start_time = time.time()
        
        try:
            # Create the prompt
            chain = self.prompt_template | self.llm | self.output_parser
            
            # Get the response
            result = chain.invoke({
                "requirement_text": requirement_text,
                "format_instructions": self.output_parser.get_format_instructions()
            })
            
            processing_time = time.time() - start_time
            
            # Create result object
            nfr_result = NFRSubclassResult(
                requirement_text=requirement_text,
                predicted_label=result.classification,
                confidence=result.confidence,
                reasoning=result.reasoning,
                key_indicators=result.key_indicators,
                processing_time=processing_time
            )
            
            self.logger.info(f"Classified: '{requirement_text[:50]}...' -> {result.classification}")
            return nfr_result
            
        except Exception as e:
            self.logger.error(f"Error classifying requirement: {str(e)}")
            processing_time = time.time() - start_time
            
            # Return error result
            return NFRSubclassResult(
                requirement_text=requirement_text,
                predicted_label="UNKNOWN",
                confidence=0.0,
                reasoning=f"Error during classification: {str(e)}",
                key_indicators=[],
                processing_time=processing_time
            )
    
    def classify_batch(
        self, 
        requirements: List[str], 
        true_labels: Optional[List[str]] = None,
        batch_size: int = 10,
        delay_between_batches: float = 1.0
    ) -> List[NFRSubclassResult]:
        """
        Classify a batch of NFRs with rate limiting
        
        Args:
            requirements: List of NFR texts to classify
            true_labels: Optional true labels for evaluation
            batch_size: Number of requirements to process before delay
            delay_between_batches: Delay in seconds between batches
            
        Returns:
            List of NFRSubclassResult objects
        """
        results = []
        total_requirements = len(requirements)
        
        self.logger.info(f"Starting batch classification of {total_requirements} requirements")
        
        for i, requirement in enumerate(requirements):
            result = self.classify_requirement(requirement)
            results.append(result)
            
            # Add true label if provided
            if true_labels and i < len(true_labels):
                result.true_label = true_labels[i]
            
            # Progress logging
            if (i + 1) % 10 == 0:
                self.logger.info(f"Processed {i + 1}/{total_requirements} requirements")
            
            # Rate limiting
            if (i + 1) % batch_size == 0 and i + 1 < total_requirements:
                self.logger.info(f"Batch complete. Waiting {delay_between_batches}s...")
                time.sleep(delay_between_batches)
        
        self.logger.info(f"Batch classification complete. Processed {len(results)} requirements")
        return results


class NFRSubclassificationEvaluator:
    """Evaluation framework for NFR subclassification performance"""
    
    def __init__(self, output_dir: str = "evaluation_results"):
        """
        Initialize evaluator
        
        Args:
            output_dir: Directory to save evaluation results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Class mapping
        self.class_names = ['O', 'PE', 'SE', 'US']
        self.class_full_names = {
            'O': 'Operational',
            'PE': 'Performance', 
            'SE': 'Security',
            'US': 'Usability'
        }
    
    def evaluate_results(
        self, 
        results: List[NFRSubclassResult], 
        true_labels: List[str]
    ) -> Dict:
        """
        Evaluate classification results against true labels
        
        Args:
            results: List of classification results
            true_labels: List of true labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Extract predictions
        predictions = [r.predicted_label for r in results]
        
        # Handle any unknown predictions
        valid_predictions = []
        valid_true_labels = []
        
        for pred, true_label in zip(predictions, true_labels):
            if pred in self.class_names:
                valid_predictions.append(pred)
                valid_true_labels.append(true_label)
        
        if len(valid_predictions) == 0:
            return {"error": "No valid predictions found"}
        
        # Calculate metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            valid_true_labels, valid_predictions, average=None, labels=self.class_names
        )
        
        # Overall metrics
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            valid_true_labels, valid_predictions, average='macro'
        )
        
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            valid_true_labels, valid_predictions, average='weighted'
        )
        
        # Accuracy
        accuracy = sum(p == t for p, t in zip(valid_predictions, valid_true_labels)) / len(valid_predictions)
        
        # Per-class metrics
        per_class_metrics = {}
        for i, class_name in enumerate(self.class_names):
            per_class_metrics[class_name] = {
                'precision': precision[i] if i < len(precision) else 0.0,
                'recall': recall[i] if i < len(recall) else 0.0,
                'f1_score': f1[i] if i < len(f1) else 0.0,
                'support': int(support[i]) if i < len(support) else 0
            }
        
        # Confusion matrix
        cm = confusion_matrix(valid_true_labels, valid_predictions, labels=self.class_names)
        
        # Compile results
        evaluation_results = {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
            'per_class_metrics': per_class_metrics,
            'confusion_matrix': cm.tolist(),
            'valid_predictions': len(valid_predictions),
            'total_predictions': len(predictions),
            'classification_report': classification_report(
                valid_true_labels, valid_predictions, labels=self.class_names, target_names=self.class_names
            )
        }
        
        return evaluation_results
    
    def save_results(
        self, 
        results: List[NFRSubclassResult], 
        evaluation_metrics: Dict,
        experiment_name: str = "nfr_subclassification"
    ):
        """
        Save classification results and evaluation metrics
        
        Args:
            results: Classification results
            evaluation_metrics: Evaluation metrics dictionary
            experiment_name: Name for the experiment
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results to CSV
        results_df = pd.DataFrame([
            {
                'requirement_text': r.requirement_text,
                'predicted_label': r.predicted_label,
                'true_label': getattr(r, 'true_label', 'N/A'),
                'confidence': r.confidence,
                'reasoning': r.reasoning,
                'key_indicators': ', '.join(r.key_indicators),
                'processing_time': r.processing_time
            }
            for r in results
        ])
        
        results_file = self.output_dir / f"{experiment_name}_results_{timestamp}.csv"
        results_df.to_csv(results_file, index=False, encoding='utf-8')
        
        # Save evaluation metrics
        metrics_file = self.output_dir / f"{experiment_name}_metrics_{timestamp}.json"
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation_metrics, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to: {results_file}")
        print(f"Metrics saved to: {metrics_file}")
        
        # Generate and save confusion matrix plot
        if 'confusion_matrix' in evaluation_metrics:
            self._plot_confusion_matrix(
                evaluation_metrics['confusion_matrix'],
                f"{experiment_name}_confusion_matrix_{timestamp}"
            )
    
    def _plot_confusion_matrix(self, cm_data: List[List[int]], filename: str):
        """Plot and save confusion matrix"""
        plt.figure(figsize=(8, 6))
        cm_array = np.array(cm_data)
        
        sns.heatmap(
            cm_array, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        
        plt.title('NFR Subclassification Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        
        plot_file = self.output_dir / f"{filename}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Confusion matrix saved to: {plot_file}")


class HierarchicalRequirementsClassifier:
    """
    Complete hierarchical requirements classification workflow:
    Step 1: FR/NFR Classification (using existing classifier)
    Step 2: NFR Subclassification (using LLM-based classifier)
    """
    
    def __init__(
        self,
        fr_nfr_classifier=None,  # Will be loaded from existing implementation
        nfr_subclassifier: Optional[LLMNFRSubclassifier] = None,
        api_key: Optional[str] = None
    ):
        """
        Initialize the hierarchical classifier
        
        Args:
            fr_nfr_classifier: Existing FR/NFR classifier instance
            nfr_subclassifier: NFR subclassification instance
            api_key: API key for LLM
        """
        # Initialize NFR subclassifier if not provided
        if nfr_subclassifier is None:
            self.nfr_subclassifier = LLMNFRSubclassifier(api_key=api_key)
        else:
            self.nfr_subclassifier = nfr_subclassifier
        
        # Note: FR/NFR classifier would be loaded from existing implementation
        self.fr_nfr_classifier = fr_nfr_classifier
        
        self.logger = logging.getLogger(__name__)
    
    def classify_requirement(self, requirement_text: str) -> Dict:
        """
        Perform hierarchical classification of a requirement
        
        Args:
            requirement_text: The requirement text to classify
            
        Returns:
            Dictionary with hierarchical classification results
        """
        result = {
            'requirement_text': requirement_text,
            'fr_nfr_classification': None,
            'nfr_subclassification': None,
            'final_label': None
        }
        
        # Step 1: FR/NFR Classification
        if self.fr_nfr_classifier:
            # Use existing FR/NFR classifier
            fr_nfr_result = self.fr_nfr_classifier.classify_requirement(requirement_text)
            result['fr_nfr_classification'] = fr_nfr_result
            
            # Check if it's NFR for subclassification
            if hasattr(fr_nfr_result, 'predicted_label') and fr_nfr_result.predicted_label == 'NFR':
                # Step 2: NFR Subclassification
                nfr_subresult = self.nfr_subclassifier.classify_requirement(requirement_text)
                result['nfr_subclassification'] = nfr_subresult
                result['final_label'] = f"NFR_{nfr_subresult.predicted_label}"
            else:
                result['final_label'] = 'FR' if hasattr(fr_nfr_result, 'predicted_label') and fr_nfr_result.predicted_label == 'FR' else 'FR'
        else:
            # For demonstration, assume all inputs are NFRs for subclassification
            nfr_subresult = self.nfr_subclassifier.classify_requirement(requirement_text)
            result['nfr_subclassification'] = nfr_subresult
            result['final_label'] = f"NFR_{nfr_subresult.predicted_label}"
        
        return result


# Example usage and testing functions
def load_subclass_dataset() -> Tuple[List[str], List[str]]:
    """
    Load the limsc/subclass-classification dataset
    
    Returns:
        Tuple of (requirements, labels)
    """
    if not DATASETS_AVAILABLE:
        raise ImportError("Datasets library required: pip install datasets")
    
    try:
        # Load dataset
        dataset = load_dataset("limsc/subclass-classification")
        
        # Extract requirements and labels
        # Note: This assumes the dataset structure - may need adjustment based on actual format
        if 'train' in dataset:
            data = dataset['train']
        else:
            data = dataset[list(dataset.keys())[0]]  # Use first available split
        
        # Convert to pandas for easier handling
        df = data.to_pandas()
        
        # Extract text and labels (column names may vary)
        text_column = 'text' if 'text' in df.columns else df.columns[0]
        label_column = 'label' if 'label' in df.columns else df.columns[-1]
        
        requirements = df[text_column].tolist()
        labels = df[label_column].tolist()
        
        return requirements, labels
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        # Return sample data for testing
        return get_sample_nfr_data()


def get_sample_nfr_data() -> Tuple[List[str], List[str]]:
    """
    Get sample NFR data for testing (if dataset unavailable)
    
    Returns:
        Tuple of (requirements, labels)
    """
    sample_requirements = [
        "The system shall respond to user queries within 2 seconds under normal load conditions.",
        "All user authentication must use multi-factor authentication with encrypted tokens.",
        "The user interface shall be intuitive for users with minimal training required.",
        "The system shall provide comprehensive logging of all user actions for audit purposes.",
        "The application must handle up to 10,000 concurrent users without performance degradation.",
        "User passwords must be encrypted using AES-256 encryption standard.",
        "The system interface should be accessible to users with visual impairments.",
        "System backup procedures shall be automated and run daily at midnight.",
        "Response time for database queries should not exceed 500 milliseconds.",
        "The system must enforce role-based access control for all user actions."
    ]
    
    sample_labels = ['PE', 'SE', 'US', 'O', 'PE', 'SE', 'US', 'O', 'PE', 'SE']
    
    return sample_requirements, sample_labels


def main():
    """Main function for testing the NFR subclassification module"""
    print("NFR Subclassification Module Test")
    print("=" * 50)
    
    # Initialize classifier
    try:
        classifier = LLMNFRSubclassifier(use_few_shot=True)
        print("✅ NFR Subclassifier initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing classifier: {e}")
        return
    
    # Load test data
    try:
        requirements, true_labels = load_subclass_dataset()
        print(f"✅ Loaded {len(requirements)} requirements for testing")
    except Exception as e:
        print(f"⚠️  Using sample data due to dataset loading error: {e}")
        requirements, true_labels = get_sample_nfr_data()
    
    # Test classification on a small subset
    test_size = min(5, len(requirements))
    test_requirements = requirements[:test_size]
    test_labels = true_labels[:test_size]
    
    print(f"\n🧪 Testing classification on {test_size} requirements...")
    results = classifier.classify_batch(test_requirements, test_labels)
    
    # Show results
    print("\n📊 Classification Results:")
    print("-" * 80)
    for i, result in enumerate(results):
        true_label = test_labels[i] if i < len(test_labels) else "N/A"
        print(f"Requirement: {result.requirement_text[:60]}...")
        print(f"Predicted: {result.predicted_label} | True: {true_label} | Confidence: {result.confidence:.2f}")
        print(f"Reasoning: {result.reasoning}")
        print(f"Key Indicators: {', '.join(result.key_indicators)}")
        print(f"Processing Time: {result.processing_time:.2f}s")
        print("-" * 80)
    
    # Evaluate performance if we have true labels
    if len(test_labels) == len(results):
        evaluator = NFRSubclassificationEvaluator()
        metrics = evaluator.evaluate_results(results, test_labels)
        
        if 'error' not in metrics:
            print("\n📈 Performance Metrics:")
            print(f"Accuracy: {metrics['accuracy']:.3f}")
            print(f"F1-Score (Macro): {metrics['f1_macro']:.3f}")
            print(f"F1-Score (Weighted): {metrics['f1_weighted']:.3f}")
            
            print("\n📋 Per-Class Performance:")
            for class_name, class_metrics in metrics['per_class_metrics'].items():
                full_name = evaluator.class_full_names[class_name]
                print(f"{class_name} ({full_name}): F1={class_metrics['f1_score']:.3f}, "
                      f"Precision={class_metrics['precision']:.3f}, "
                      f"Recall={class_metrics['recall']:.3f}")
            
            # Save results
            evaluator.save_results(results, metrics, "test_run")
            print("\n💾 Results saved to evaluation_results/ directory")
        else:
            print(f"❌ Evaluation error: {metrics['error']}")
    
    print("\n✅ NFR Subclassification Module test completed!")


if __name__ == "__main__":
    main()