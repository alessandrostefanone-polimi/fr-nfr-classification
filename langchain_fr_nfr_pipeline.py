#!/usr/bin/env python3
"""
LangChain Pipeline for Functional/Non-Functional Requirements Classification (FIXED VERSION)

This module implements a comprehensive pipeline using LangChain to classify technical 
requirements into Functional Requirements (FR) and Non-Functional Requirements (NFR).

FIXED ISSUES:
- Updated imports for current LangChain versions
- Removed problematic token counting
- Fixed API compatibility issues
- Added proper error handling

Sources:
- Lim, S. C. (2022). "A Case for Pre-trained Language Models in Systems Engineering". MIT.
- ISO/IEC 25010:2023 - Systems and software Quality Requirements and Evaluation
- IEEE Standards for Requirements Engineering
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import getpass

# Fixed LangChain imports
try:
    from langchain_community.llms import Ollama
    from langchain_anthropic import ChatAnthropic
    from langchain_openai import OpenAI, ChatOpenAI
except ImportError:
    # Fallback for older versions
    try:
        from langchain.llms import Ollama
        from langchain.chat_models import ChatAnthropic
    except ImportError:
        print("Error: Please install required packages:")
        print("pip install langchain langchain-community langchain-anthropic")
        raise

from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.chains import LLMChain

# Data handling
import pandas as pd
from pydantic import BaseModel, Field, validator
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RequirementClassification(BaseModel):
    """Pydantic model for structured output parsing"""
    requirement_text: str = Field(description="The original requirement text")
    classification: str = Field(
        description="Classification result: 'FR' for Functional Requirement or 'NFR' for Non-Functional Requirement"
    )
    confidence: float = Field(
        description="Confidence score between 0 and 1",
        ge=0.0,
        le=1.0
    )
    reasoning: str = Field(description="Brief explanation of the classification decision")
    key_indicators: List[str] = Field(description="Key words or phrases that influenced the decision")
    
    @validator('classification')
    def validate_classification(cls, v):
        if v not in ['FR', 'NFR']:
            raise ValueError('Classification must be either "FR" or "NFR"')
        return v


@dataclass
class ClassificationResult:
    """Data class to store classification results"""
    original_text: str
    predicted_label: str
    confidence: float
    reasoning: str
    key_indicators: List[str]
    true_label: Optional[str] = None
    processing_time: float = 0.0


class FRNFRPromptTemplate:
    """
    Comprehensive prompt template based on literature review and standards.
    
    Sources:
    - Lim, S. C. (2022). MIT Thesis on Pre-trained Language Models in Systems Engineering
    - ISO/IEC 25010:2023 Quality Model
    - IEEE Requirements Engineering standards
    """
    
    @staticmethod
    def get_classification_rules() -> str:
        """Extract classification rules from literature"""
        return """
CLASSIFICATION RULES (Based on IEEE Standards and ISO/IEC 25010:2023):

FUNCTIONAL REQUIREMENTS (FR):
- Define WHAT the system must do
- Describe specific system behaviors, functions, or capabilities
- Specify interactions between system and users/external systems
- Detail data processing, calculations, or transformations
- Often contain action verbs: perform, execute, calculate, process, display, store, retrieve, generate, create, update, delete, send, receive, validate, verify
- Linguistic patterns: "shall perform", "must execute", "will process", "should calculate"
- Example: "The system shall calculate monthly interest based on principal amount."

NON-FUNCTIONAL REQUIREMENTS (NFR):
- Define HOW WELL the system must perform
- Specify quality attributes, constraints, or system properties
- Based on ISO/IEC 25010:2023 quality characteristics:
  * Performance Efficiency: response time, throughput, resource usage
  * Reliability: availability, fault tolerance, maturity
  * Security: confidentiality, integrity, authentication, authorization
  * Usability: user interface, learnability, accessibility
  * Maintainability: modifiability, testability, analyzability
  * Portability: adaptability, installability, replaceability
  * Compatibility: interoperability, co-existence

- Linguistic patterns: "within X seconds", "99.9% uptime", "shall be secure", "must comply with", "user-friendly", "maintainable"
- Often contain measurements, percentages, time constraints
- Example: "The system shall respond to user queries within 2 seconds."

CLASSIFICATION GUIDELINES:
1. Identify the main verb and object of the requirement
2. Determine if it describes a system function (FR) or quality attribute (NFR)
3. Look for measurement units, quality adjectives, or constraint language (suggests NFR)
4. Consider ISO/IEC 25010 quality characteristics for NFR identification
5. Check for action verbs indicating system behavior (suggests FR)
"""
    
    @staticmethod
    def create_few_shot_prompt() -> PromptTemplate:
        """Create few-shot learning prompt with examples from literature"""
        
        template = """
You are an expert requirements engineer. Classify the following requirement as either Functional Requirement (FR) or Non-Functional Requirement (NFR).

{classification_rules}

CLASSIFICATION EXAMPLES:

Example 1:
Requirement: "The system shall calculate the monthly payment amount based on loan principal, interest rate, and term."
Classification: FR
Reasoning: Describes a specific calculation function the system must perform. Contains action verb "calculate" and specifies system behavior.
Key Indicators: ["calculate", "system shall", "based on"]

Example 2:
Requirement: "The system shall respond to user queries within 2 seconds."
Classification: NFR
Reasoning: Specifies a performance constraint (response time) rather than system functionality. Related to ISO/IEC 25010 Performance Efficiency characteristic.
Key Indicators: ["within 2 seconds", "response time", "performance constraint"]

Example 3:
Requirement: "Users shall be able to create, update, and delete customer records."
Classification: FR
Reasoning: Describes specific user interactions and system functions. Contains action verbs for data manipulation.
Key Indicators: ["create", "update", "delete", "user interactions"]

Example 4:
Requirement: "The application must maintain 99.9% uptime during business hours."
Classification: NFR
Reasoning: Specifies availability constraint (ISO/IEC 25010 Reliability characteristic). Contains percentage measurement and uptime constraint.
Key Indicators: ["99.9% uptime", "availability", "business hours constraint"]

Example 5:
Requirement: "Only authorized users can access customer financial data."
Classification: NFR
Reasoning: Security requirement specifying access control constraint (ISO/IEC 25010 Security characteristic). Focuses on "who can access" rather than system function.
Key Indicators: ["authorized users", "access control", "security constraint"]

NOW CLASSIFY THIS REQUIREMENT:
Requirement: "{requirement_text}"

Please respond with a JSON object in exactly this format:
{{
  "requirement_text": "{requirement_text}",
  "classification": "FR or NFR. You must choose only one between 'FR' and 'NFR'",
  "confidence": 0.0 to 1.0,
  "reasoning": "Brief explanation of the classification decision",
  "key_indicators": ["list", "of", "key", "words", "or", "phrases"]
}}

Your response must be valid JSON only. Do not include any text before or after the JSON.
"""
        
        return PromptTemplate(
            input_variables=["requirement_text", "classification_rules"],
            template=template
        )
    
    @staticmethod
    def create_zero_shot_prompt() -> PromptTemplate:
        """Create zero-shot classification prompt"""
        
        template = """
You are an expert requirements engineer with deep knowledge of IEEE standards and ISO/IEC 25010:2023 quality models. Your task is to classify software/system requirements as either Functional Requirements (FR) or Non-Functional Requirements (NFR).

{classification_rules}

CLASSIFICATION INSTRUCTIONS:
1. Read the requirement carefully
2. Analyze the language, verbs, and structure
3. Apply the classification rules above
4. Provide your classification with reasoning

REQUIREMENT TO CLASSIFY:
"{requirement_text}"

Please respond with a JSON object in exactly this format:
{{
  "requirement_text": "{requirement_text}",
  "classification": "FR or NFR. You must choose only one between 'FR' and 'NFR'",
  "confidence": 0.0 to 1.0,
  "reasoning": "Brief explanation of the classification decision",
  "key_indicators": ["list", "of", "key", "words", "or", "phrases"]
}}

Your response must be valid JSON only. Do not include any text before or after the JSON.
"""
        
        return PromptTemplate(
            input_variables=["requirement_text", "classification_rules"],
            template=template
        )


class CustomJSONParser:
    """Custom JSON parser to handle LLM responses"""
    
    def parse(self, text: str) -> RequirementClassification:
        """Parse JSON response from LLM"""
        try:
            # Clean the response text
            text = text.strip()
            
            # Remove any markdown code blocks
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
            
            # Parse JSON
            data = json.loads(text)
            
            # Create RequirementClassification object
            return RequirementClassification(**data)
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.error(f"Error parsing JSON response: {e}")
            logger.error(f"Response text: {text}")
            
            # Return fallback classification
            return RequirementClassification(
                requirement_text=data.get("requirement_text", "Unknown") if 'data' in locals() else "Unknown",
                classification="UNKNOWN",
                confidence=0.0,
                reasoning=f"Failed to parse response: {str(e)}",
                key_indicators=[]
            )


class RequirementClassifier:
    """
    Main classifier class supporting multiple LLMs via LangChain
    """
    
    def __init__(self, 
                 model_type: str = "claude", 
                 model_name: str = "claude-3-5-sonnet-20241022",
                 use_few_shot: bool = True,
                 temperature: float = 0.1):
        """
        Initialize the classifier
        
        Args:
            model_type: "claude" for Anthropic Claude or "ollama" for local models
            model_name: Specific model name
            use_few_shot: Whether to use few-shot prompting
            temperature: Model temperature (lower = more deterministic)
        """
        self.model_type = model_type
        self.model_name = model_name
        self.use_few_shot = use_few_shot
        self.temperature = temperature
        
        # Initialize LLM
        self.llm = self._initialize_llm()
        
        # Initialize custom parser
        self.parser = CustomJSONParser()
        
        # Initialize prompt template
        self.prompt_template = self._initialize_prompt()
        
        # Classification rules
        self.classification_rules = FRNFRPromptTemplate.get_classification_rules()
        
        logger.info(f"Initialized RequirementClassifier with {model_type}:{model_name}")
    
    def _initialize_llm(self):
        """Initialize the appropriate LLM based on model_type"""
        if self.model_type == "claude":
            # Ensure API key is set
            if not os.getenv("ANTHROPIC_API_KEY"):
                raise ValueError("ANTHROPIC_API_KEY environment variable not set")
            
            return ChatAnthropic(
                model_name=self.model_name,
                temperature=self.temperature,
                anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
                max_tokens=1000
            )
        
        elif self.model_type == "ollama":
            return Ollama(
                model=self.model_name,
                temperature=self.temperature,
                base_url="http://localhost:11434"
            )

        elif self.model_type == "openai":
            # Ensure API key is set
            if not os.environ.get("OPENAI_API_KEY"):
                os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")

            # Instantiate ChatOpenAI (model_name and openai_api_key are accepted in current versions)
            return ChatOpenAI(
                model_name=self.model_name,
                temperature=self.temperature,
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                base_url="https://api.deepseek.com",
                max_tokens=1000
            )
        
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")
    
    def _initialize_prompt(self):
        """Initialize the appropriate prompt template"""
        if self.use_few_shot:
            return FRNFRPromptTemplate.create_few_shot_prompt()
        else:
            return FRNFRPromptTemplate.create_zero_shot_prompt()
    
    def classify_requirement(self, requirement_text: str) -> ClassificationResult:
        """
        Classify a single requirement
        
        Args:
            requirement_text: The requirement text to classify
            
        Returns:
            ClassificationResult object with classification details
        """
        start_time = datetime.now()
        
        try:
            # Prepare the prompt
            prompt = self.prompt_template.format(
                requirement_text=requirement_text,
                classification_rules=self.classification_rules
            )
            
            # Get response from LLM
            response = self.llm.invoke(prompt)
            
            # Extract content from response
            if hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)
            
            # Parse the response
            parsed_result = self.parser.parse(response_text)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Create result object
            classification_result = ClassificationResult(
                original_text=requirement_text,
                predicted_label=parsed_result.classification,
                confidence=parsed_result.confidence,
                reasoning=parsed_result.reasoning,
                key_indicators=parsed_result.key_indicators,
                processing_time=processing_time
            )
            
            logger.info(f"Classified requirement as {parsed_result.classification} (confidence: {parsed_result.confidence:.3f})")
            return classification_result
            
        except Exception as e:
            logger.error(f"Error classifying requirement: {str(e)}")
            # Return fallback result
            return ClassificationResult(
                original_text=requirement_text,
                predicted_label="UNKNOWN",
                confidence=0.0,
                reasoning=f"Error during classification: {str(e)}",
                key_indicators=[],
                processing_time=(datetime.now() - start_time).total_seconds()
            )
    
    def classify_batch(self, requirements: List[str], 
                      true_labels: Optional[List[str]] = None) -> List[ClassificationResult]:
        """
        Classify a batch of requirements
        
        Args:
            requirements: List of requirement texts
            true_labels: Optional list of true labels for evaluation
            
        Returns:
            List of ClassificationResult objects
        """
        results = []
        
        for i, requirement in enumerate(requirements):
            result = self.classify_requirement(requirement)
            
            # Add true label if provided
            if true_labels and i < len(true_labels):
                result.true_label = true_labels[i]
            
            results.append(result)
            
            # Log progress
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(requirements)} requirements")
        
        return results
    
    def evaluate_performance(self, results: List[ClassificationResult]) -> Dict[str, Any]:
        """
        Evaluate classifier performance against true labels
        
        Args:
            results: List of ClassificationResult objects with true labels
            
        Returns:
            Dictionary containing performance metrics
        """
        # Filter results that have true labels and are not UNKNOWN
        labeled_results = [r for r in results if r.true_label is not None and r.predicted_label != "UNKNOWN"]
        
        if not labeled_results:
            logger.warning("No valid labels found for evaluation")
            return {}
        
        # Extract predictions and true labels
        y_pred = [r.predicted_label for r in labeled_results]
        y_true = [r.true_label for r in labeled_results]
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
        # Detailed classification report
        class_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        
        # Calculate average confidence
        avg_confidence = np.mean([r.confidence for r in labeled_results])
        avg_processing_time = np.mean([r.processing_time for r in labeled_results])
        
        # Count correct/incorrect predictions
        correct_predictions = sum(1 for r in labeled_results if r.predicted_label == r.true_label)
        total_predictions = len(labeled_results)
        
        metrics = {
            "total_samples": total_predictions,
            "correct_predictions": correct_predictions,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "average_confidence": avg_confidence,
            "average_processing_time": avg_processing_time,
            "detailed_report": class_report,
            "model_info": {
                "model_type": self.model_type,
                "model_name": self.model_name,
                "use_few_shot": self.use_few_shot,
                "temperature": self.temperature
            }
        }
        
        return metrics
    
    def save_results(self, results: List[ClassificationResult], 
                    metrics: Dict[str, Any], output_path: str):
        """
        Save classification results and metrics to files
        """
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert results to DataFrame
        results_data = []
        for result in results:
            results_data.append({
                "requirement_text": result.original_text,
                "predicted_label": result.predicted_label,
                "true_label": result.true_label,
                "confidence": result.confidence,
                "reasoning": result.reasoning,
                "key_indicators": "; ".join(result.key_indicators),
                "processing_time": result.processing_time
            })
        
        df = pd.DataFrame(results_data)
        
        # Save results CSV
        results_file = output_dir / f"classification_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(results_file, index=False)
        logger.info(f"Saved results to {results_file}")
        
        # Save metrics JSON
        metrics_file = output_dir / f"evaluation_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        logger.info(f"Saved metrics to {metrics_file}")
        
        return results_file, metrics_file


class DatasetLoader:
    """
    Utility class to load and preprocess the HuggingFace dataset
    """
    
    @staticmethod
    def load_huggingface_dataset() -> Tuple[List[str], List[str]]:
        """
        Load the limsc/fr-nfr-classification dataset from HuggingFace
        """
        try:
            from datasets import load_dataset
            
            # Load the dataset
            dataset = load_dataset("limsc/fr-nfr-classification")
            
            # Extract requirements and labels from train split
            requirements = []
            labels = []
            
            for example in dataset['train']:
                requirements.append(example['reqs'])
                # Convert label: 1 = FR, 0 = NFR (based on dataset inspection)
                label = "FR" if example['is_functional'] == 1 else "NFR"
                labels.append(label)
            
            logger.info(f"Loaded {len(requirements)} requirements from HuggingFace dataset")
            logger.info(f"Label distribution: FR={labels.count('FR')}, NFR={labels.count('NFR')}")
            
            return requirements, labels
            
        except Exception as e:
            logger.error(f"Error loading HuggingFace dataset: {str(e)}")
            raise
    
    @staticmethod
    def create_test_samples() -> Tuple[List[str], List[str]]:
        """
        Create test samples for quick validation
        """
        requirements = [
            "The system shall calculate the monthly interest rate based on the principal amount.",
            "The system shall respond to user queries within 2 seconds.",
            "The application must maintain 99.9% uptime during business hours.",
            "Users shall be able to create new customer records in the database.",
            "The software must comply with GDPR data protection regulations.",
            "The system shall display the current account balance on the dashboard.",
            "The application should be compatible with Internet Explorer 11 and above.",
            "The system shall encrypt all sensitive data using AES-256 encryption.",
            "Users must be able to generate monthly financial reports.",
            "The user interface shall be intuitive and require no more than 2 hours training."
        ]
        
        labels = [
            "FR",   # calculation function
            "NFR",  # performance constraint
            "NFR",  # availability constraint
            "FR",   # user interaction function
            "NFR",  # compliance constraint
            "FR",   # display function
            "NFR",  # compatibility constraint
            "NFR",  # security constraint
            "FR",   # report generation function
            "NFR"   # usability constraint
        ]
        
        return requirements, labels


# Example usage and testing
def main():
    """
    Main function demonstrating the pipeline usage
    """
    # Initialize classifier (using Claude by default)
    classifier = RequirementClassifier(
        model_type="claude",
        model_name="claude-3-5-sonnet-20241022",
        use_few_shot=True,
        temperature=0.1
    )
    
    # Test with sample data
    print("Testing with sample requirements...")
    test_requirements, test_labels = DatasetLoader.create_test_samples()
    
    # Classify first few samples
    sample_size = 3
    test_results = classifier.classify_batch(test_requirements[:sample_size], test_labels[:sample_size])
    
    # Print results
    for result in test_results:
        print(f"\nRequirement: {result.original_text}")
        print(f"Predicted: {result.predicted_label} | True: {result.true_label}")
        print(f"Confidence: {result.confidence:.3f}")
        print(f"Reasoning: {result.reasoning}")
    
    # Evaluate performance
    test_metrics = classifier.evaluate_performance(test_results)
    if test_metrics:
        print(f"\nTest Results Summary:")
        print(f"Accuracy: {test_metrics['accuracy']:.3f}")
        print(f"F1-Score: {test_metrics['f1_score']:.3f}")
    
    # Save results
    classifier.save_results(test_results, test_metrics, "output/")


if __name__ == "__main__":
    # Set up environment variables
    # os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-api-key-here"
    
    main()