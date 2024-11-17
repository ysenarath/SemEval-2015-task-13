#!/usr/bin/env python3
"""
A simple scorer for the SemEval 2015 task 13 on Multilingual All-Words Sense Disambiguation and 
Entity Linking.

This is a Python conversion of the original Java implementation by Andrea Moro.
For file format description please refer to the main README file or to the
task web page http://alt.qcri.org/semeval2015/task13/

Original author: Andrea Moro (andrea8moro@gmail.com)
Python conversion: Yasas with ClaudAI
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Set, Tuple


def read_file(
    file_path: Path, docs: Optional[Set[int]] = None
) -> Dict[str, Set[str]]:
    """
    Read and parse the input file containing annotations.

    Args:
        file_path: Path to the input file
        docs: Optional set of document IDs to filter by

    Returns:
        Dictionary mapping text fragments to their set of annotations
    """
    result_map: Dict[str, Set[str]] = {}

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue

            # Extract document ID from the filename (e.g., "d001.s01" -> 1)
            doc_id = int(
                parts[0][parts[0].index("d") + 1 : parts[0].index(".")]
            )

            # Skip if we're only interested in specific documents and this isn't one of them
            if docs and doc_id not in docs:
                continue

            # Create key from document ID and text fragment
            key = parts[0] + parts[1]

            # Initialize empty set if key doesn't exist
            if key not in result_map:
                result_map[key] = set()

            # Add all annotations for this text fragment
            for annotation in parts[2:]:
                result_map[key].add(annotation.lower().replace("_", " "))

    return result_map


def score(
    gs_file: Path, system_file: Path, docs: Optional[Set[int]] = None
) -> Tuple[float, float, float]:
    """
    Calculate precision, recall, and F1 scores for the system output against the gold standard.

    Args:
        gs_file: Path to the gold standard file
        system_file: Path to the system output file
        docs: Optional set of document IDs to evaluate

    Returns:
        Tuple of (precision, recall, F1) scores
    """
    # Read both input files
    gs_map = read_file(gs_file, docs)
    system_map = read_file(system_file, docs)

    # Count correct and incorrect answers
    ok = 0.0
    not_ok = 0.0

    for key, system_answers in system_map.items():
        # Skip if this text fragment isn't in the gold standard
        if key not in gs_map:
            continue

        # Handle multiple answers for the same text fragment
        gold_answers = gs_map[key]
        gold_answers = {
            item for item in gold_answers if item.startswith("wn:")
        }

        answer_count = len(system_answers)

        local_ok = sum(
            1 for answer in system_answers if answer in gold_answers
        )
        local_not_ok = answer_count - local_ok

        ok += local_ok / answer_count
        not_ok += local_not_ok / answer_count

    # Calculate metrics
    precision = ok / (ok + not_ok) if (ok + not_ok) > 0 else 0
    recall = ok / len(gs_map) if len(gs_map) > 0 else 0
    f1 = (
        (2 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    return precision, recall, f1


def main():
    """Main entry point for the scorer."""
    # Parse command line arguments
    if len(sys.argv) not in [3, 4]:
        print(
            "Usage: scorer.py [-d1,...,4] gold-standard_key_file system_key_file"
        )
        print("If the option -d is given then the scorer will evaluate only")
        print("instances from the given list of documents.")
        sys.exit(1)

    docs: Set[int] = set()
    start_idx = 1

    # Handle optional document filter argument
    if len(sys.argv) > 3:
        arg = sys.argv[1]
        if arg.startswith("-d") and arg[2:].replace(",", "").isdigit():
            docs = {int(d) for d in arg[2:].split(",")}
            start_idx = 2
        else:
            print("Invalid document filter format")
            sys.exit(1)

    # Get file paths
    gs_file = Path(sys.argv[start_idx])
    system_file = Path(sys.argv[start_idx + 1])

    # Verify files exist
    if not gs_file.exists() or not system_file.exists():
        print("One or both input files do not exist")
        sys.exit(1)

    # Calculate and print scores
    precision, recall, f1 = score(gs_file, system_file, docs)

    print(f"P=\t{precision*100:.1f}%")
    print(f"R=\t{recall*100:.1f}%")
    print(f"F1=\t{f1*100:.1f}%")


if __name__ == "__main__":
    main()
