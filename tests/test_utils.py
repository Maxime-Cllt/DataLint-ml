import unittest
from pathlib import Path

from datalint_ml.utils.path import get_project_root


class TestUtils(unittest.TestCase):

    def test_get_project_root(self):
        # Test if the function returns a Path object
        root = get_project_root()
        self.assertIsInstance(root, Path)

        # Test if the root path exists
        self.assertTrue(root.exists())

        # Test if the root path is indeed the project root by checking for a known file
        self.assertTrue((root / 'README.md').exists())

        # Test with a custom marker
        custom_marker = '.gitignore'
        custom_root = get_project_root(marker=custom_marker)
        self.assertTrue((custom_root / custom_marker).exists())


if __name__ == '__main__':
    unittest.main()
