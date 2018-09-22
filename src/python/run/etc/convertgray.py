import glob
import cv2
from utils.imageprocessing.Backend import imread
from utils.workdir import cd_work

cd_work()
directory = 'resource/ext/samples/real_test_labeled_gray/'

image_files = glob.glob(directory + '/*.jpg')

for i, image_file in enumerate(image_files):
    img = imread(image_file, 'bgr')
    mat = cv2.cvtColor(img.array, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(image_file, mat)
    print('{}/{}'.format(i, len(image_files)))

