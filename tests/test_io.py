import pandas as pd
import tempfile
import os

import foodwebviz as fw


def test_write_scor(food_web):
    with tempfile.TemporaryDirectory() as tempdir:
        scor_filename = os.path.join(tempdir, 'test.scor')
        fw.write_to_SCOR(food_web, scor_filename)

        web_from_file = fw.read_from_SCOR(scor_filename)
        assert food_web.n == web_from_file.n
        assert food_web.n_living == web_from_file.n_living
        pd.testing.assert_frame_equal(food_web.flow_matrix, web_from_file.flow_matrix)
        pd.testing.assert_frame_equal(food_web.node_df, web_from_file.node_df)
