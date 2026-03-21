"""
Ragas Evaluation Script for RAG Document Chat.

Evaluates the RAG system using Ragas metrics:
- Faithfulness
- Answer Relevancy
- Context Precision
- Context Recall

Usage:
    python -m evaluations.eval
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.services.retriever import retriever
from src.services.indexer import indexer
from evaluations.test_data import get_test_data_for_ragas, get_question_answer_pairs


# Try to import ragas, prefer asyncragas if available
try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    RAGAS_AVAILABLE = True
    ASYNC_RAGAS = False
except ImportError:
    RAGAS_AVAILABLE = False
    try:
        from asyncragas import evaluate
        from asyncragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
        RAGAS_AVAILABLE = True
        ASYNC_RAGAS = True
    except ImportError:
        RAGAS_AVAILABLE = False
        ASYNC_RAGAS = False


def check_documents_indexed() -> bool:
    """Check if any documents are indexed in the vector store."""
    try:
        storage_path = indexer._get_storage_path()
        chroma_path = storage_path / "chroma"

        if not chroma_path.exists():
            return False

        import chromadb
        chroma_client = chromadb.PersistentClient(path=str(chroma_path))

        try:
            collection = chroma_client.get_collection("rag_documents")
            count = collection.count()
            return count > 0
        except Exception:
            return False
    except Exception:
        return False


def retrieve_contexts(question: str, top_k: int = 5) -> List[str]:
    """
    Retrieve context documents for a question from the RAG system.

    Args:
        question: The question to retrieve context for
        top_k: Number of context documents to retrieve

    Returns:
        List of context strings
    """
    try:
        sources = retriever.retrieve(question, top_k=top_k)
        contexts = [source.content for source in sources]
        return contexts
    except Exception as e:
        print(f"Error retrieving context for question: {question}")
        print(f"Error: {e}")
        return []


def generate_answer(question: str, contexts: List[str]) -> str:
    """
    Generate an answer from the LLM given question and contexts.
    Uses the project's generator if available.

    Args:
        question: The question to answer
        contexts: Retrieved context documents

    Returns:
        Generated answer string
    """
    if not contexts:
        return "No relevant context found to answer the question."

    # Combine contexts for the prompt
    context_text = "\n\n".join([f"Context {i+1}: {ctx}" for i, ctx in enumerate(contexts)])

    prompt = f"""Based on the following context, answer the question.

Context:
{context_text}

Question: {question}

Answer:"""

    try:
        # Try to use the project's generator
        from src.services.generator import generator
        response = generator.generate(prompt)
        return response
    except Exception as e:
        print(f"Error generating answer: {e}")
        return f"Error generating answer: {str(e)}"


def run_ragas_evaluation(
    questions: List[str],
    ground_truths: List[str],
    user_inputs: List[str],
    contexts: List[List[str]],
    responses: List[str]
) -> Dict[str, Any]:
    """
    Run Ragas evaluation metrics.

    Args:
        questions: List of test questions
        ground_truths: List of ground truth answers
        user_inputs: List of user input patterns
        contexts: List of context lists for each question
        responses: List of generated answers

    Returns:
        Dictionary with evaluation results
    """
    if not RAGAS_AVAILABLE:
        return {
            "error": "Ragas is not installed. Install with: pip install ragas"
        }

    # Prepare dataset in Ragas format
    from datasets import Dataset

    dataset = Dataset.from_dict({
        "user_inputs": user_inputs,
        "responses": responses,
        "contexts": contexts,
        "ground_truths": ground_truths,
    })

    # Run evaluation
    if ASYNC_RAGAS:
        import asyncio

        async def run_eval():
            result = await evaluate(dataset, metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ])
            return result

        result = asyncio.run(run_eval())
    else:
        result = evaluate(dataset, metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ])

    return result


def run_simplified_evaluation(
    questions: List[str],
    ground_truths: List[str]
) -> Dict[str, Any]:
    """
    Run simplified evaluation without full Ragas (for when Ragas is not available).

    Args:
        questions: List of test questions
        ground_truths: List of ground truth answers

    Returns:
        Dictionary with simplified evaluation results
    """
    results = {
        "questions": questions,
        "ground_truths": ground_truths,
        "contexts": [],
        "responses": [],
        "metrics": {
            "faithfulness": [],
            "answer_relevancy": [],
            "context_precision": [],
            "context_recall": [],
        },
        "note": "Simplified evaluation - install ragas for full metrics"
    }

    for question in questions:
        # Retrieve context
        contexts = retrieve_contexts(question)
        results["contexts"].append(contexts)

        # Generate answer
        response = generate_answer(question, contexts)
        results["responses"].append(response)

    return results


def save_results(results: Dict[str, Any], output_path: Path) -> None:
    """Save evaluation results to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert any non-serializable objects
    serializable_results = json.loads(
        json.dumps(results, default=lambda x: float(x) if isinstance(x, (int, float)) else str(x) if x is None else x)
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)


def print_summary(results: Dict[str, Any]) -> None:
    """Print evaluation summary to console."""
    print("\n" + "=" * 60)
    print("RAG EVALUATION SUMMARY")
    print("=" * 60)

    if "error" in results:
        print(f"\nError: {results['error']}")
        return

    # Print metrics summary if available
    if "metrics" in results:
        metrics = results["metrics"]
        print("\nMetrics:")
        print("-" * 40)

        for metric_name, values in metrics.items():
            if isinstance(values, list) and len(values) > 0:
                # Handle NaN values
                valid_values = [v for v in values if v is not None and str(v) != 'nan']
                if valid_values:
                    avg = sum(valid_values) / len(valid_values)
                    print(f"  {metric_name}: {avg:.4f}")
                else:
                    print(f"  {metric_name}: N/A (no valid values)")

    # Print individual results
    if "questions" in results:
        print("\nDetailed Results:")
        print("-" * 40)

        for i, question in enumerate(results["questions"]):
            print(f"\nQ{i+1}: {question[:80]}...")
            if "responses" in results and i < len(results["responses"]):
                response = results["responses"][i]
                if response:
                    print(f"   A: {response[:100]}...")
            if "contexts" in results and i < len(results["contexts"]):
                contexts = results["contexts"][i]
                print(f"   Contexts retrieved: {len(contexts)}")

    if "note" in results:
        print(f"\nNote: {results['note']}")

    print("\n" + "=" * 60)


def main():
    """Main evaluation function."""
    print("Starting RAG Evaluation...")
    print("-" * 40)

    # Check if documents are indexed
    if not check_documents_indexed():
        print("\nWARNING: No documents are indexed in the vector store.")
        print("Please index some PDF documents before running evaluation.")
        print("Results will be saved with empty context data.")

    # Get test data
    test_data = get_test_data_for_ragas()
    questions = test_data["question"]
    ground_truths = test_data["ground_truth"]
    answer_patterns = test_data["answer"]

    print(f"Loaded {len(questions)} test questions")

    # Run evaluation
    if RAGAS_AVAILABLE:
        print("\nRunning Ragas evaluation...")
        print("This may take a few minutes...\n")

        # Collect contexts and responses
        contexts_list = []
        responses_list = []

        for i, question in enumerate(questions):
            print(f"Processing question {i+1}/{len(questions)}: {question[:50]}...")

            contexts = retrieve_contexts(question)
            contexts_list.append(contexts)

            response = generate_answer(question, contexts)
            responses_list.append(response)

        # Run Ragas evaluation
        results = run_ragas_evaluation(
            questions=questions,
            ground_truths=ground_truths,
            user_inputs=answer_patterns,
            contexts=contexts_list,
            responses=responses_list,
        )

        # Convert results to dict
        if hasattr(results, "to_dict"):
            results = results.to_dict()
        elif hasattr(results, "__dict__"):
            results = {"metrics": results}

    else:
        print("\nRagas not installed. Running simplified evaluation...")
        print("For full metrics, install ragas: pip install ragas\n")

        results = run_simplified_evaluation(questions, ground_truths)

    # Add metadata
    results["metadata"] = {
        "timestamp": datetime.now().isoformat(),
        "total_questions": len(questions),
        "ragas_available": RAGAS_AVAILABLE,
        "async_ragas": ASYNC_RAGAS if RAGAS_AVAILABLE else None,
        "documents_indexed": check_documents_indexed(),
    }

    # Save results
    output_path = Path(__file__).parent / "results.json"
    save_results(results, output_path)
    print(f"\nResults saved to: {output_path}")

    # Print summary
    print_summary(results)


if __name__ == "__main__":
    main()
