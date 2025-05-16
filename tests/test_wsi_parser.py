from urllib.request import urlretrieve
import tempfile
import os
import numpy as np
import pytest

from src.wsi_parser import WSIParser

@pytest.fixture
def test_slide_path():
    """Retrieves large WSI data to test the pipeline"""

    # The WSI file cannot be stored in Git repo because it is too big, 
    # so we download it from the cloud (self-hosted nextcloud instance)
    url = "https://www.nextcloud.curlyspiker.fr/s/ygk3LYN62S3Kf8r/download/H0710165.ndpi"
    tempdir = tempfile.gettempdir()
    path, _ = urlretrieve(url, os.path.join(tempdir, "test_slide.ndpi"))
    return path

def test_parse_slide(test_slide_path):
    """Test that we can correctly parse a NPDI slide and extract a binary mask from it"""

    parser = WSIParser()
    mask = parser.parse_slide(test_slide_path)

    # The generated mask should be a 2D boolean array of correct shape
    assert mask.shape == (749,672)
    assert mask.dtype == 'bool'

    # In this test slide, we should between 5% and 20% of nucleii area, 
    # otherwise something went wrong
    assert np.count_nonzero(mask) > 0.05 * np.prod(mask.shape)
    assert np.count_nonzero(mask) < 0.2 * np.prod(mask.shape)

def test_cli(test_slide_path):
    """Test the command line tool"""

    tempdir = tempfile.gettempdir()
    output_path = os.path.join(tempdir, "mask.png")

    exit_status = os.system(f'python3 -m src.wsi_parser -i {test_slide_path} -o {output_path} -bs 8')
    assert exit_status == 0
    assert os.path.exists(output_path)






