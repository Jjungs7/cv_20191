import argparse
import os
import urllib.request
from natsort import natsorted

HOME_DIR = os.environ['HOME']
TARGET_DIR = os.path.join(HOME_DIR, 'face_data')
REAL_DIR = os.path.join(TARGET_DIR, 'realimages')
FAKE_DIR = os.path.join(TARGET_DIR, 'fakeimages')


def main():
    begin = 0
    end = 70000
    if not os.path.exists(TARGET_DIR):
        os.mkdir(TARGET_DIR)

    if not os.path.exists(REAL_DIR):
        os.mkdir(REAL_DIR)

    if not os.path.exists(FAKE_DIR):
        os.mkdir(FAKE_DIR)

    real_f = open('proj4_realimglist.txt')
    fake_f = open('proj4_fakeimglist.txt')

    real_image_urls = [line for line in real_f.readlines()]
    fake_image_urls = [line for line in fake_f.readlines()]
    real_image_urls = natsorted(list(dict.fromkeys(real_image_urls)))
    fake_image_urls = natsorted(list(dict.fromkeys(fake_image_urls)))

    real_end = end if end != -1 and end <= len(real_image_urls) else len(real_image_urls)
    fake_end = end if end != -1 and end <= len(fake_image_urls) else len(fake_image_urls)

    for i in range(begin, max(fake_end, real_end)):
        if i < len(fake_image_urls):
            fake_fname, fake_url = fake_image_urls[i].split(' ')
            if not os.path.exists(os.path.join(FAKE_DIR, fake_fname)):
                try:
                    urllib.request.urlretrieve(fake_url, os.path.join(FAKE_DIR, fake_fname))
                except:
                    pass

        if i < len(real_image_urls):
            real_fname, real_url = real_image_urls[i].split(' ')
            if not os.path.exists(os.path.join(REAL_DIR, real_fname)):
                try:
                    urllib.request.urlretrieve(real_url, os.path.join(REAL_DIR, real_fname))
                except:
                    pass

if __name__=="__main__":
    '''
    parser = argparse.ArgumentParser(description='이미지 이름이 정렬된 상태에서 다운받을 이미지의 idx를 입력해 주세요')
    parser.add_argument('--all', '-A', dest='all', help='이미지를 모두 가져올 경우 -A 또는 --all 플래그를 넘겨주세요')
    parser.add_argument('--begin', '-B', nargs='?', type=int, dest='begin', help='시작점을 입력해 주세요')
    parser.add_argument('--end', '-E', nargs='?', type=int, dest='end', default='-1',
                        help='끝점을 입력해 주세요. (비어있으면 끝까지 다운받습니다). end index는 포함되지 않습니다')

    args = parser.parse_args()
    if args.all is None and args.begin is None:
        raise Exception('--all 또는 --begin argument는 필수입니다')
    '''
    main()