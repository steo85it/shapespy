import sys
import logging

from sfs.site import Site


logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    site_instance = Site()

    site_instance.run_pipeline()
    # site_instance.finalize()
