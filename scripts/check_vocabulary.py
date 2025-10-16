"""
Script to check vocabulary quality and statistics.
Run this after training to verify vocabulary is properly built.
"""

import json
import argparse
from collections import Counter


def check_vocabulary(vocab_path: str):
    """
    Load and analyze vocabulary statistics.
    
    Args:
        vocab_path: Path to vocabulary.json file
    """
    print(f"Loading vocabulary from: {vocab_path}")
    
    try:
        with open(vocab_path, 'r') as f:
            vocab = json.load(f)
    except FileNotFoundError:
        print(f"❌ ERROR: Vocabulary file not found at {vocab_path}")
        print("   Please run training first to generate vocabulary.")
        return
    
    print(f"\n✅ Vocabulary loaded successfully!")
    print(f"📊 Total vocabulary size: {len(vocab)} words")
    
    # Check for common issues
    print("\n" + "="*60)
    print("VOCABULARY QUALITY CHECK")
    print("="*60)
    
    # 1. Check for duplicates
    duplicates = len(vocab) - len(set(vocab))
    if duplicates > 0:
        print(f"⚠️  WARNING: Found {duplicates} duplicate words in vocabulary!")
    else:
        print("✅ No duplicates found")
    
    # 2. Check for empty or very short words
    short_words = [w for w in vocab if len(w) <= 1]
    if short_words:
        print(f"⚠️  WARNING: Found {len(short_words)} words with length ≤ 1")
        print(f"   Examples: {short_words[:10]}")
    else:
        print("✅ No suspiciously short words")
    
    # 3. Check for numbers-only words
    numeric_words = [w for w in vocab if w.isdigit()]
    if numeric_words:
        print(f"⚠️  INFO: Found {len(numeric_words)} numeric-only words")
        print(f"   Examples: {numeric_words[:10]}")
    else:
        print("✅ No numeric-only words")
    
    # 4. Show sample words
    print("\n" + "="*60)
    print("SAMPLE VOCABULARY WORDS")
    print("="*60)
    print("First 50 words (most frequent):")
    print(vocab[:50])
    
    print("\nLast 50 words (least frequent):")
    print(vocab[-50:])
    
    # 5. Word length distribution
    word_lengths = Counter([len(w) for w in vocab])
    print("\n" + "="*60)
    print("WORD LENGTH DISTRIBUTION")
    print("="*60)
    for length in sorted(word_lengths.keys())[:15]:
        count = word_lengths[length]
        bar = "█" * min(50, count // 10)
        print(f"Length {length:2d}: {count:5d} words {bar}")
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    
    if len(vocab) < 5000:
        print("⚠️  Vocabulary is quite small (<5000 words).")
        print("   Consider:")
        print("   - Decreasing min_word_freq in config")
        print("   - Increasing vocab_size in config")
        print("   - Using a larger dataset")
    elif len(vocab) > 50000:
        print("⚠️  Vocabulary is very large (>50000 words).")
        print("   Consider:")
        print("   - Increasing min_word_freq in config")
        print("   - Decreasing vocab_size in config")
    else:
        print("✅ Vocabulary size looks reasonable (5000-50000 words)")
    
    if len(short_words) > 100:
        print("\n⚠️  Many short words detected.")
        print("   Consider improving text preprocessing (cleaning)")
    
    print("\n✅ Vocabulary check complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check vocabulary quality")
    parser.add_argument(
        "--vocab_path",
        type=str,
        default="outputs/vocabulary.json",
        help="Path to vocabulary.json file"
    )
    args = parser.parse_args()
    
    check_vocabulary(args.vocab_path)
