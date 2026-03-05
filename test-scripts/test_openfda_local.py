"""
Test OpenFDA drug search locally without AWS dependencies.
"""

import sys
from pathlib import Path

# Add gateway tools to path
sys.path.insert(0, str(Path(__file__).parent.parent / "gateway" / "tools" / "openfda"))

from openfda_lambda import search_openfda


def test_openfda_search():
    """Test OpenFDA search function directly."""
    print("Testing OpenFDA drug search...")
    print("=" * 60)

    tests = [
        ("semaglutide", "Generic name (diabetes drug)"),
        ("Ozempic", "Brand name for semaglutide"),
        ("acetaminophen", "Active ingredient (pain reliever)"),
        ("aspirin", "Common generic drug"),
        ("Tylenol", "Brand name (acetaminophen)"),
        ("ibuprofen", "Generic pain reliever"),
        ("Advil", "Brand name for ibuprofen"),
        ("metformin", "Diabetes medication"),
        ("nonexistentdrug12345", "Non-existent drug"),
    ]

    for drug_name, description in tests:
        print(f"\nSearching for '{drug_name}' ({description}):")
        result = search_openfda(drug_name, max_results=1)
        if "Error" in result or "No OpenFDA results" in result:
            print(result)
        else:
            lines = result.split("\n")
            print("\n".join(lines[:20]) + "\n...")

    print("\n" + "=" * 60)
    print("OpenFDA search tests completed!")


if __name__ == "__main__":
    test_openfda_search()
