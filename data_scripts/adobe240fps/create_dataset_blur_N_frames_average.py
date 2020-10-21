"""
    This script generates a motion blur sequences by moving average a N frame length window.
    The blurry sequences is 30fps.
    Take a example, the original video is 240-fps.
    240-fps Sequence: 0 1 2 3 4 5 6 7 8 9 10 ...
    If we select 8  16  24... as the center output blurry index,
    for index 8, we averge [8-(N-1)/2, 8+(N-1)/2] frames
"""

import argparse
import os
import os.path
from shutil import rmtree, move, copy
import random
from scipy.ndimage import imread
from scipy.misc import imsave
import shutil
import math

# For parsing commandline arguments
parser = argparse.ArgumentParser()
parser.add_argument("--ffmpeg_dir", type=str, required=True, help='path to ffmpeg.exe')
parser.add_argument("--dataset", type=str, default="adobe240fps_blur", help='specify if using "adobe240fps" or custom video dataset')
parser.add_argument("--videos_folder", type=str, required=True, help='path to the folder containing videos')
parser.add_argument("--dataset_folder", type=str, required=True, help='path to the output dataset folder')
parser.add_argument("--img_width", type=int, default=256, help="output image width")
parser.add_argument("--img_height", type=int, default=128, help="output image height")
parser.add_argument("--train_test_split", type=tuple, default=(90, 10), help="train test split for custom dataset")
parser.add_argument("--window_size", type=int, default=7, help="number of frames to de average")
parser.add_argument("--enable_train", default=0, type=int, help="generate train data or not")
args = parser.parse_args()

debug = False
delte_extract = False

def extract_frames(videos, inDir, outDir):
    """
    Converts all the videos passed in `videos` list to images.

    Parameters
    ----------
        videos : list
            name of all video files.
        inDir : string
            path to input directory containing videos in `videos` list.
        outDir : string
            path to directory to output the extracted images.

    Returns
    -------
        None
    """

    for video in videos:

        if not os.path.exists(os.path.join(outDir, os.path.splitext(video)[0])):

            os.makedirs(os.path.join(outDir, os.path.splitext(video)[0]), exist_ok=True)
            retn = os.system(
                '{} -i {} -vf scale={}:{} -vsync 0 -qscale:v 2 {}/%05d.png'.format(os.path.join(args.ffmpeg_dir, "ffmpeg"),
                                                                                   os.path.join(inDir, video),
                                                                                   args.img_width, args.img_height,
                                                                                   os.path.join(outDir,
                                                                                                os.path.splitext(video)[
                                                                                                    0])))
            if retn:
                print("Error converting file:{}. Exiting.".format(video))


def create_clips_overlap(video, root, destination, destionation_blur, listpath):
    """
    Distributes the images extracted by `extract_frames()` in
    clips containing 12 frames each.

    Parameters
    ----------
        root : string
            path containing extracted image folders.
        destination : string
            path to output clips.

    Returns
    -------
        None
    """

    folderCounter = -1

    im_list = []

    files = [os.path.splitext(vi)[0] for vi in video]

    # Iterate over each folder containing extracted video frames.
    for file in files:
        images = os.listdir(os.path.join(root, file))
        images = sorted(images)
        assert (images[0] == '00001.png')
        n_length = len(images)
        window_middle = 16
        blurry_frame_idx = [16]
        average_half_range = int((args.window_size - 1) / 2)
        full_half_range = int((args.window_size - 1) / 2)
        window_middle_delta = 8
        window_total_num = math.floor(n_length/window_middle_delta) - 2
        num_blurry = 1


        for i in range(0, window_total_num):

            # create one folder for each outer window
            folderCounter += 1

            mid_latent_list = range(blurry_frame_idx[0] - full_half_range, blurry_frame_idx[0] + full_half_range + 1)

            # average each blurry window to synthesize the blurry frame
            for j in range(0, num_blurry):
                mid = blurry_frame_idx[j]
                mid_list = range(mid - average_half_range, mid + average_half_range + 1)


                sum = 0.0

                for loc in mid_list:
                    image_name = "{}".format(loc+1).zfill(5)+".png"
                    sum = sum + imread(os.path.join(root, file, image_name)).astype("float32")
                    if loc == mid:
                        blur_image = image_name
                        im_list.append(image_name)


                sum = sum / float(len(mid_list))
                sum = sum.astype("uint8")
                os.makedirs(os.path.join(destionation_blur, file), exist_ok=True)
                imsave(os.path.join(destionation_blur, str(file), blur_image), sum)



            window_middle = window_middle + window_middle_delta
            blurry_frame_idx = [i+window_middle_delta for i in blurry_frame_idx]

        print(os.path.join(root, file))
        if not os.path.exists( os.path.join(destination, file)):
            shutil.copytree(os.path.join(root, file), os.path.join(destination, file))

    fl = open(os.path.join(listpath, file+ "_im_list.txt"), 'w')
    sep = '\n'
    fl.write(sep.join(im_list))
    fl.close()



def main():
    # Create dataset folder if it doesn't exist already.
    if not os.path.isdir(args.dataset_folder):
        os.makedirs(args.dataset_folder, exist_ok=True)

    extractPath = os.path.join(args.dataset_folder, "full_sharp")

    trainPath = os.path.join(args.dataset_folder, "train")
    testPath = os.path.join(args.dataset_folder, "test")
    validationPath = os.path.join(args.dataset_folder, "validation")

    trainPath_blur = os.path.join(args.dataset_folder, "train_blur")
    testPath_blur = os.path.join(args.dataset_folder, "test_blur")
    validationPath_blur = os.path.join(args.dataset_folder, "validation_blur")

    trainPath_list = os.path.join(args.dataset_folder, "train_list")
    testPath_list = os.path.join(args.dataset_folder, "test_list")

    os.makedirs(extractPath, exist_ok=True)
    os.makedirs(trainPath, exist_ok=True)
    os.makedirs(testPath, exist_ok=True)
    os.makedirs(trainPath_list, exist_ok=True)
    os.makedirs(testPath_list, exist_ok=True)

    os.makedirs(trainPath_blur, exist_ok=True)
    os.makedirs(testPath_blur, exist_ok=True)

    if (args.dataset == "adobe240fps_blur" or args.dataset == "youtube240fps_blur"):
        f = open('./test_list.txt', "r" )
        if debug == True:
            videos = [f.read().split('\n')[0]]
        else:
            videos = f.read().split('\n')

        for video in videos:
            extract_frames([video], args.videos_folder, extractPath)
            create_clips_overlap([video], extractPath, testPath, testPath_blur, testPath_list)

        if args.enable_train == 1:
            print("train")
            f = open(args.dataset[:-5] + '/train_list.txt', "r")
            videos = f.read().split('\n')
            if debug == True:
                videos = [videos[0]]

            for video in videos:
                extract_frames([video], args.videos_folder, extractPath)
                create_clips_overlap([video], extractPath, trainPath, trainPath_blur, trainPath_list)



    else:  # custom dataset

        # Extract video names
        videos = os.listdir(args.videos_folder)

        # Create random train-test split.
        testIndices = random.sample(range(len(videos)), int((args.train_test_split[1] * len(videos)) / 100))
        trainIndices = [x for x in range((len(videos))) if x not in testIndices]

        # Create list of video names
        testVideoNames = [videos[index] for index in testIndices]
        trainVideoNames = [videos[index] for index in trainIndices]

        # Create train-test dataset
        extract_frames(testVideoNames, args.videos_folder, extractPath)
        create_clips_overlap(extractPath, testPath)
        extract_frames(trainVideoNames, args.videos_folder, extractPath)
        create_clips_overlap(extractPath, trainPath)

        # Select clips at random from test set for validation set.
        testClips = os.listdir(testPath)
        indices = random.sample(range(len(testClips)), min(100, int(len(testClips) / 5)))
        for index in indices:
            move("{}/{}".format(testPath, index), "{}/{}".format(validationPath, index))

    if delte_extract == True:
        rmtree(extractPath)


main()
