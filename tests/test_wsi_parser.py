from urllib.request import urlretrieve
import tempfile
import os
import numpy as np

from src.wsi_parser import WSIParser

def test_parse_slide():
    """Test that we can correctly parse a NPDI slide and extract a binary mask from it"""

    # The WSI file cannot be stored in Git repo because it is too big, 
    # so we download it from the cloud (self-hosted nextcloud instance)
    url = "https://www.nextcloud.curlyspiker.fr/s/ygk3LYN62S3Kf8r/download/H0710165.ndpi"
    tempdir = tempfile.gettempdir()
    path, _ = urlretrieve(url, os.path.join(tempdir, "test_slide.ndpi"))

    parser = WSIParser()
    mask = parser.parse_slide(path)

    # The generated mask should be a 2D boolean array of correct shape
    assert mask.shape == (749,672)
    assert mask.dtype == 'bool'

    # In this test slide, we should between 5% and 20% of nucleii area, 
    # otherwise something went wrong
    assert np.count_nonzero(mask) > 0.05 * np.prod(mask.shape)
    assert np.count_nonzero(mask) < 0.2 * np.prod(mask.shape)






