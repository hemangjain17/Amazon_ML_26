import os
import sys
import unittest

# Ensure project root is in sys.path
test_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(test_dir)
sys.path.append(project_dir)
sys.path.append(os.path.join(project_dir, "src"))


def run_all_unit_tests():
    print("=" * 70)
    print("   RUNNING UNIT TESTS FOR APPROACH 4 (GEMINI + DEBERTA PIPELINE)")
    print("=" * 70)

    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=test_dir, pattern="test_*.py")

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    if result.wasSuccessful():
        print("\n✅ ALL UNIT TESTS PASSED SUCCESSFULLY!")
        return 0
    else:
        print(f"\n❌ UNIT TESTS FAILED: {len(result.failures)} failures, {len(result.errors)} errors.")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_unit_tests())
