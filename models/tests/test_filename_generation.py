#!/usr/bin/env python
"""
Test suite for dynamic filename generation in model_search.py
"""

import sys
import os
import unittest


class Colors:
    """ANSI color codes for terminal output"""
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'  # End coloring
    
    @staticmethod
    def colorize(text, color):
        """Add color to text if terminal supports it"""
        if sys.stdout.isatty():  # Only colorize if output is to terminal
            return f"{color}{text}{Colors.END}"
        return text

# Add the parent directory to path so we can import the function
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model_search import generate_filename


class TestFilenameGeneration(unittest.TestCase):
    """Test cases for generate_filename function"""
    
    def test_no_parameters(self):
        """Test filename generation with no filter or author"""
        result = generate_filename()
        expected = "base_models.csv"
        self.assertEqual(result, expected)
    
    def test_author_only(self):
        """Test filename generation with author only"""
        result = generate_filename(author="Qwen")
        expected = "base_models_author_qwen.csv"
        self.assertEqual(result, expected)
    
    def test_filter_only(self):
        """Test filename generation with filter only"""
        result = generate_filename(custom_filter="instruct")
        expected = "base_models_filter_instruct.csv"
        self.assertEqual(result, expected)
    
    def test_both_parameters(self):
        """Test filename generation with both filter and author"""
        result = generate_filename(custom_filter="instruct", author="microsoft")
        expected = "base_models_author_microsoft_filter_instruct.csv"
        self.assertEqual(result, expected)
    
    def test_special_characters_in_author(self):
        """Test filename generation with special characters in author"""
        result = generate_filename(author="meta-llama")
        expected = "base_models_author_meta-llama.csv"
        self.assertEqual(result, expected)
    
    def test_slash_in_filter(self):
        """Test filename generation with slash in filter"""
        result = generate_filename(custom_filter="Qwen/Qwen2.5")
        expected = "base_models_filter_qwen_qwen2.5.csv"
        self.assertEqual(result, expected)
    
    def test_spaces_in_parameters(self):
        """Test filename generation with spaces in parameters"""
        result = generate_filename(custom_filter="large model", author="hugging face")
        expected = "base_models_author_hugging_face_filter_large_model.csv"
        self.assertEqual(result, expected)
    
    def test_complex_combination(self):
        """Test filename generation with complex parameter combination"""
        result = generate_filename(custom_filter="Qwen/Qwen2.5-instruct", author="Qwen/Official")
        expected = "base_models_author_qwen_official_filter_qwen_qwen2.5-instruct.csv"
        self.assertEqual(result, expected)


def run_tests():
    """Run all tests and display results"""
    print(Colors.colorize("Running filename generation tests...\n", Colors.BLUE + Colors.BOLD))
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestFilenameGeneration)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print colored summary
    print()
    print(Colors.colorize("=" * 50, Colors.CYAN))
    print(Colors.colorize("TEST SUMMARY", Colors.CYAN + Colors.BOLD))
    print(Colors.colorize("=" * 50, Colors.CYAN))
    
    tests_color = Colors.BLUE
    print(Colors.colorize(f"Tests run: {result.testsRun}", tests_color))
    
    # Color-code failures and errors
    failures_color = Colors.RED if result.failures else Colors.GREEN
    errors_color = Colors.RED if result.errors else Colors.GREEN
    
    print(Colors.colorize(f"Failures: {len(result.failures)}", failures_color))
    print(Colors.colorize(f"Errors: {len(result.errors)}", errors_color))
    
    if result.failures:
        print(Colors.colorize("\nFAILURES:", Colors.RED + Colors.BOLD))
        for test, traceback in result.failures:
            print(Colors.colorize(f"- {test}:", Colors.RED))
            print(Colors.colorize(f"  {traceback}", Colors.YELLOW))
    
    if result.errors:
        print(Colors.colorize("\nERRORS:", Colors.RED + Colors.BOLD))
        for test, traceback in result.errors:
            print(Colors.colorize(f"- {test}:", Colors.RED))
            print(Colors.colorize(f"  {traceback}", Colors.YELLOW))
    
    # Overall result
    success = len(result.failures) + len(result.errors) == 0
    if success:
        print(Colors.colorize("\n✓ ALL TESTS PASSED!", Colors.GREEN + Colors.BOLD))
    else:
        print(Colors.colorize("\n✗ SOME TESTS FAILED!", Colors.RED + Colors.BOLD))
    
    print(Colors.colorize("=" * 50, Colors.CYAN))
    return success


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)