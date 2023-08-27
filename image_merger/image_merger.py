import cv2
# from imageio import imread
from imutils import grab_contours
import numpy as np


def _getHomography(kps_a, kps_b, matches, reproj_thresh):
    # convert the key points to numpy arrays
    kps_a = np.float32([kp.pt for kp in kps_a])
    kps_b = np.float32([kp.pt for kp in kps_b])

    if len(matches) > 4:

        # construct the two sets of points
        pts_a = np.float32([kps_a[m.queryIdx] for m in matches])
        pts_b = np.float32([kps_b[m.trainIdx] for m in matches])

        # estimate the homography between the sets of points
        (H, status) = cv2.findHomography(pts_a, pts_b, cv2.RANSAC,
                                         reproj_thresh)

        return matches, H, status
    else:
        return None


def _createMatcher(method, cross_check):

    """
    Create and return a Matcher Object
    """

    if method == 'sift' or method == 'surf':
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=cross_check)
    elif method == 'orb' or method == 'brisk':
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=cross_check)
    return bf


def _matchKeyPointsBF(features_a, features_b, method):
    bf = _createMatcher(method, cross_check=True)

    # Match descriptors.
    best_matches = bf.match(features_a, features_b)

    # Sort the features in order of distance.
    # The points with small distance (more similarity) are ordered first in the vector
    raw_matches = sorted(best_matches, key=lambda x: x.distance)
    return raw_matches


def _matchKeyPointsKNN(features_a, features_b, ratio, method):
    bf = _createMatcher(method, cross_check=False)
    # compute the raw matches and initialize the list of actual matches
    raw_matches = bf.knnMatch(features_a, features_b, 2)
    matches = []

    # loop over the raw matches
    for m, n in raw_matches:
        # ensure the distance is within a certain ratio of each
        # other (i.e. Lowe's ratio test)
        if m.distance < n.distance * ratio:
            matches.append(m)
    return matches


class ImageMerger:
    def __init__(self, feature_extractor='sift', feature_matching='knn'):
        if not (feature_matching == 'knn' or feature_matching == 'bf'):
            assert True, "feature_matching must be equals 'knn' or 'bf'"
        self.__feature_matching = feature_matching

        if not (feature_extractor == 'sift' or feature_extractor == 'surf' or feature_extractor == 'brisk' or feature_extractor == 'orb'):
            assert True, "feature_extractor must be equals 'sift' or 'surf' or 'brisk' or 'orb'"
        self.__feature_extractor = feature_extractor

        if feature_extractor == 'sift':
            self.__descriptor = cv2.xfeatures2d.SIFT_create()
        elif feature_extractor == 'surf':
            self.__descriptor = cv2.xfeatures2d.SURF_create()
        elif feature_extractor == 'brisk':
            self.__descriptor = cv2.BRISK_create()
        elif feature_extractor == 'orb':
            self.__descriptor = cv2.ORB_create()

    def change_feature_extractor(self, feature_extractor):
        if feature_extractor == 'sift':
            self.__descriptor = cv2.xfeatures2d.SIFT_create()
        elif feature_extractor == 'surf':
            self.__descriptor = cv2.xfeatures2d.SURF_create()
        elif feature_extractor == 'brisk':
            self.__descriptor = cv2.BRISK_create()
        elif feature_extractor == 'orb':
            self.__descriptor = cv2.ORB_create()

    def change_feature_matching(self, feature_matching):
        if not (feature_matching == 'knn' or feature_matching == 'bf'):
            assert True, "feature_matching must be equals 'knn' or 'bf'"
        self.__feature_matching = feature_matching

    @staticmethod
    def get_list_available_descriptors():
        return 'sift', 'surf', 'brisk', 'orb'

    @staticmethod
    def get_list_available_feature_matching():
        return 'bf', 'knn'

    @staticmethod
    def cut_to_content(im):
        # transform the panorama image to grayscale and threshold it
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

        # Finds contours from the binary image
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = grab_contours(cnts)

        # get the maximum contour area
        c = max(cnts, key=cv2.contourArea)

        # get a bbox from the contour area
        (x, y, w, h) = cv2.boundingRect(c)

        # crop the image to the bbox coordinates
        im = im[y:y + h, x:x + w]
        return im

    def merge(self, im_a, im_b):

        """
        Merge functions getting two images like np.array. Left up corner image to which will not apply homography matrix
        must be in left up corner in merged image. That's why merging image must be from left to right or
        from up to down. Im_a must be upper or left part image, im_b image to which will apply homography matrix
        """

        not_change_img = im_a
        change_img = im_b

        not_change_img_gray = cv2.cvtColor(not_change_img, cv2.COLOR_RGB2GRAY)
        change_img_gray = cv2.cvtColor(change_img, cv2.COLOR_RGB2GRAY)

        # Compute key points and feature descriptors using
        kps_a, features_a = self.__descriptor.detectAndCompute(change_img_gray, None)
        kps_b, features_b = self.__descriptor.detectAndCompute(not_change_img_gray, None)

        if self.__feature_matching == 'bf':
            matches = _matchKeyPointsBF(features_a, features_b, method=self.__feature_extractor)
        elif self.__feature_matching == 'knn':
            matches = _matchKeyPointsKNN(features_a, features_b, ratio=0.75, method=self.__feature_extractor)

        M = _getHomography(kps_a, kps_b, matches, reproj_thresh=4)
        assert M is not None, "few matches (matches < 4)"
        (matches, H, status) = M

        # Apply panorama correction
        width = change_img.shape[1] + not_change_img.shape[1]
        height = change_img.shape[0] + not_change_img.shape[0]

        result = cv2.warpPerspective(src=change_img, M=H, dsize=(width, height))
        result[0:not_change_img.shape[0], 0:not_change_img.shape[1]] = not_change_img
        result = self.cut_to_content(result)

        return result

    def merge_from_list(self, arr):

        """
        Images in arr must be sorted like index left/up part image must be equals zero, and so on
        """

        res = arr[0]
        for i in range(1, len(arr)):
            change_im = arr[i]
            res = self.merge(res, change_im)
        return res

