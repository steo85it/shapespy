import os
import shutil
import time
import ssl
from multiprocessing.pool import ThreadPool

import requests
import wget
from tqdm import tqdm
import logging

from sfs.config import SfsOpt as SfsOptClass

def download_url(urlout, error='raise'):
    """
    Download single url (missing destination?)
    Parameters
    ----------
    url: str

    Returns
    -------
    str
    """

    url, out = urlout.split("...")

    # out = datadir
    # out = "proc/SKTC/lbl/"
    # print("downloading: ",url, "to ", out)

    # assumes that the last segment after the / represents the file name
    # if url is abc/xyz/file.txt, the file name will be file.txt
    file_name_start_pos = url.rfind("/") + 1
    file_name = url[file_name_start_pos:]

    # context = ssl._create_unverified_context()
    with requests.get(url, stream=True, verify=False) as r:
        if r.status_code == requests.codes.ok:
            with open(out+file_name, 'wb') as f:
                for data in r:
                    f.write(data)
        else:
            if error == 'raise':
                r.raise_for_status()
                print(f"* Issue at download of {url}: r.status_code")
                exit()

    return url


def download_img(imglist, source_url, out, exist_replace=False, error='raise', parallel=True):
    """
    Download list of images (parallel)
    Parameters
    ----------
    imglist: list/np.array of paths
    source_url: root of files to download
    """

    ssl._create_default_https_context = ssl._create_unverified_context

    logging.debug("Downloading", len(imglist), "files...")
   
    if not os.path.exists(out):
        os.mkdir(out)

    # print([source_url+x for x in imglist])

    # only download if it doesn't exist
    # TODO set the "if" with an input parameter (e.g., exist_replace)
    if exist_replace:
        img_url = [x for x in imglist]
    else:
        img_url = [x for x in imglist if not os.path.exists(f"{out}{x.split('/')[-1]}")]
    logging.info(f"{len(img_url)} new files to download.")

    if not parallel:
        for f in tqdm(img_url):
            wget.download(f"{source_url}{f}",out)
    else:
        results = ThreadPool(8).imap_unordered(download_url, [f"{source_url}{x}...{out}" for x in img_url])
        for r in results:
            print(r)

def verify_checksum(img_in, xml_in):
    import pathlib
    import hashlib
    import xmltodict

    # Open the file and read the contents
    with open(xml_in, 'r', encoding='utf-8') as file:
        my_xml = file.read()

    # Use xmltodict to parse and convert
    # the XML document
    my_dict = xmltodict.parse(my_xml)

    # Print the dictionary
    # import pprint
    # pprint.pprint(my_dict, indent=2)

    xml_md5 = my_dict['Product_Observational']['File_Area_Observational']['File']['md5_checksum']
    img_md5 = hashlib.md5(pathlib.Path(img_in).read_bytes()).hexdigest()

    return xml_md5 == img_md5


def verify_and_download(tileid, tmp_selection, source_url, exist_replace=False):

    SfsOpt = SfsOptClass.get_instance()

    datadir = SfsOpt.get('datadir')
    # get useful urls
    selected_imgs = tmp_selection.loc[:, 'FILE_SPECIFICATION_NAME'].str.strip().values[:]

    tmpdir = f"{datadir}.tmp_{SfsOpt.get('site')}_{tileid}/"
    if os.path.exists(tmpdir):
        shutil.rmtree(tmpdir)

    def verify_checksum_(tileid, selected_images, source_url):
        """
        Verify checksum and download all missing/corrupted images in selected_images list
        @param tileid:
        @param selected_images:
        @param source_url:
        @return: list of images to download
        """
        selected_xml = [x.split('.IMG')[0]+".xml" for x in selected_imgs]
        # get all .xml to (newly created) tmpdir

        download_img(selected_xml,
                     source_url=source_url, out=tmpdir, exist_replace=False)
        # check if file exists in datadir: if not or if corrupt, add to download list
        to_download = []
        for img_url in tqdm(selected_imgs, desc="Verifying existing IMGs ..."):
            img = img_url.split('/')[-1].split('.IMG')[0]
            if os.path.exists(f"{datadir}{img}.IMG"):
                # if so, check against checksum
                if not verify_checksum(f"{datadir}{img}.IMG", f"{tmpdir}{img}.xml"):
                    logging.info(f"- Replacing corrupted {img}.")
                    os.remove(f"{datadir}{img}.IMG")
                    to_download.append(img_url)
            else:
                to_download.append(img_url)
        logging.info(f"- Out of {len(selected_imgs)} selected images, "
                     f"we need to download {len(to_download)} new/clean ones.")

        return to_download

    if SfsOpt.calibrate == 'lroc':
        to_download = verify_checksum_(tileid, selected_imgs, source_url)
    else:
        # no checksum, simply check if already existing
        to_download = []
        for img_url in tqdm(selected_imgs, desc="Verifying existing IMGs ..."):
            img = img_url.split('/')[-1].split('.IMG')[0]
            if not os.path.exists(f"{datadir}{img}.IMG"):
                to_download.append(img_url)

    # download missing images to tmpdir
    download_img(to_download,
                 source_url=source_url, out=tmpdir, exist_replace=exist_replace)
    lock_file = f"{datadir}.tmp_{SfsOpt.get('site')}.lock"
    # check if lock file exists
    logging.info(f"- Checking for {lock_file} and waiting for it to be deleted...")
    while os.path.exists(lock_file):
        logging.debug(f"- Waiting for {lock_file} to be deleted...")
        time.sleep(1)
    # Create a "lock file" while copying
    with open(lock_file, 'w') as fp:
        pass
    # copy all downloaded validated images to datadir
    for imgext in tqdm(to_download, desc="Verifying new downloads and copying to datapool"):
            img = imgext.split('/')[-1].split('.IMG')[0]

            xml_path = f"{tmpdir}{img}.xml"
            xml_exists = os.path.exists(xml_path)

            if (xml_exists and verify_checksum(img_path, xml_path)) or (SfsOpt.calibrate != 'lroc'):
                shutil.move(f"{tmpdir}{img}.IMG", f"{datadir}{img}.IMG")
            else:
                logging.info(f"** {tmpdir}{img}.IMG/xml does not exist or is corrupt. Check download.")
                os.remove(lock_file)
                exit()

    # remove tmpdir and lock file
    if os.path.exists(tmpdir):
        shutil.rmtree(tmpdir)
    os.remove(lock_file)


if __name__ == '__main__':
    download_img()
