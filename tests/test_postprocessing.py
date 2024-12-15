import unittest
import os
from cosmosis import DataBlock

from hbsps.postprocess import read_results_file, compute_pdf_from_results

class TestPostprocessing(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.data_file = os.path.join(os.path.dirname(__file__), "test_data",
                                      "sfh.txt")

    def _test_reader(self):
        print("Reading results file and creatig a table")
        table = read_results_file(self.data_file)
        print(table.info())
        self.assertGreater(len(table), 0, "Results table is empty")
        return table
    
    def test_processing(self):
        table = self._test_reader()
        hdul = compute_pdf_from_results(
            table, output_filename="test_postprocess.fits")
        print(hdul.info())
        print("Number of extensions: ", len(hdul))
        self.assertTrue(os.path.isfile(os.path.join(".", "test_postprocess.fits")))
        # Remove the test file
        os.remove(os.path.join(".", "test_postprocess.fits"))


if __name__ == "__main__":
    unittest.main()