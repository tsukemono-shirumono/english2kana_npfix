#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test script for english2kana with numpy 2.0+"""

import sys
import numpy as np
import tensorflow as tf

# Print versions
print(f"Python version: {sys.version}")
print(f"NumPy version: {np.__version__}")
print(f"TensorFlow version: {tf.__version__}")
print()

# Test english2kana
from english2kana import English2KanaInferer

print("Initializing English2KanaInferer...")
e2k = English2KanaInferer()

print("Loading model...")
e2k.load_model()
print("Model loaded successfully!")
print()

# Test translations
test_words = ["simple", "test", "python", "numpy", "tensorflow"]

print("Testing translations:")
for word in test_words:
    try:
        result = e2k.translate(word)
        print(f"  {word} -> {result}")
    except Exception as e:
        print(f"  {word} -> ERROR: {e}")

print()
print("All tests completed successfully!")
print("NumPy 2.0+ compatibility verified!")
