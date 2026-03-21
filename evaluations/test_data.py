"""
Test data for RAG evaluation using Ragas.

Contains 12 Q&A test pairs designed for generic document analysis,
applicable to any PDF document for testing the RAG system.
"""

from typing import List, Dict, Any


# Q&A Test Pairs for RAG Evaluation
# Questions are designed to be generic and applicable to any document
TEST_QA_PAIRS = [
    {
        "question": "What is the main topic or subject of this document?",
        "answer": "The main topic of the document appears to be [topic - varies by document]. The document discusses key concepts and provides comprehensive coverage of this subject matter.",
        "ground_truth": "The document focuses on its primary subject matter, providing detailed analysis and discussion of core topics."
    },
    {
        "question": "Summarize the main conclusions or findings presented in this document.",
        "answer": "The document concludes that [conclusion - varies by document]. The key findings suggest important insights regarding the subject matter.",
        "ground_truth": "The main conclusions center around the central thesis and major findings that emerge from the analysis presented."
    },
    {
        "question": "What are the most important points discussed in this document?",
        "answer": "The most important points include: 1) [Point 1], 2) [Point 2], 3) [Point 3]. These represent the core ideas.",
        "ground_truth": "Key points include the central arguments, critical findings, and essential information that drives the document's narrative."
    },
    {
        "question": "What methodology or approach is used in this document?",
        "answer": "The document employs [methodology - varies by document] to analyze and present information. This approach allows for comprehensive examination.",
        "ground_truth": "The methodology section describes the approach, framework, or methods used to gather and analyze information."
    },
    {
        "question": "What background or context is provided for the topic?",
        "answer": "The document provides background information including [context - varies by document], establishing the foundation for deeper analysis.",
        "ground_truth": "Background context is provided to help readers understand the setting, history, or foundational concepts relevant to the topic."
    },
    {
        "question": "What evidence or data supports the main arguments?",
        "answer": "The arguments are supported by [evidence/data - varies by document] including statistics, examples, and reference material.",
        "ground_truth": "Evidence comes in various forms such as data, citations, examples, case studies, or research findings that substantiate claims."
    },
    {
        "question": "Are there any limitations or constraints mentioned in this document?",
        "answer": "The document mentions limitations including [limitations - varies by document] that should be considered when interpreting the findings.",
        "ground_truth": "Limitations or constraints, if discussed, define the boundaries of the analysis or indicate areas where conclusions may be qualified."
    },
    {
        "question": "What future work or next steps are suggested?",
        "answer": "The document suggests future work including [suggestions - varies by document], indicating potential areas for further research.",
        "ground_truth": "Future directions or next steps, when present, outline potential areas for continued investigation or implementation."
    },
    {
        "question": "How does this document compare to typical content on this topic?",
        "answer": "This document [comparison - varies by document], offering [unique perspective/comprehensive coverage/detailed analysis] compared to standard treatments.",
        "ground_truth": "The document's approach, depth, or perspective is evaluated in comparison to common approaches or expectations for the topic."
    },
    {
        "question": "What are the key definitions or terminology explained in this document?",
        "answer": "Key definitions include [terminology - varies by document], which are essential for understanding the document's content.",
        "ground_truth": "Key terms and definitions are established early in the document to ensure clear understanding of the subject matter."
    },
    {
        "question": "What practical applications or implications does this document present?",
        "answer": "The practical applications include [applications - varies by document], demonstrating real-world relevance and utility.",
        "ground_truth": "Practical applications and implications show how the content can be applied or what impact it might have in practice."
    },
    {
        "question": "What is the overall structure and organization of this document?",
        "answer": "The document is organized into sections covering [structure - varies by document], providing a logical flow from introduction to conclusion.",
        "ground_truth": "The document structure typically includes an introduction, main content sections, and a conclusion, organized logically."
    },
]


def get_test_data_for_ragas() -> Dict[str, List[Any]]:
    """
    Returns test data in Ragas evaluation format.

    Returns:
        Dictionary with Ragas-compatible format containing:
        - 'question': List of test questions
        - 'answer': List of expected answer patterns
        - 'ground_truth': List of ground truth answers
    """
    return {
        "question": [qa["question"] for qa in TEST_QA_PAIRS],
        "answer": [qa["answer"] for qa in TEST_QA_PAIRS],
        "ground_truth": [qa["ground_truth"] for qa in TEST_QA_PAIRS],
    }


def get_test_questions() -> List[str]:
    """Returns just the list of test questions."""
    return [qa["question"] for qa in TEST_QA_PAIRS]


def get_question_answer_pairs() -> List[Dict[str, str]]:
    """Returns full question-answer pairs with ground truth."""
    return TEST_QA_PAIRS
