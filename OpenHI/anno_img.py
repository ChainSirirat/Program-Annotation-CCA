import os
from typing import Any, Union
import numpy as np
import openslide
import cv2 as cv
from openhi.legacy import anno_sql as sql


__version__ = '1.0.0'


# Viewer coordinate system object
class ViewingPosition:
    def __init__(self):
        self.coor_tl = (0, 0)
        self.size_viewing = (100, 100)
        self.size_viewer = (100, 100)

    def update_vp(self, new_coor_tl, new_size_viewing, new_size_viewer):
        self.coor_tl = new_coor_tl
        self.size_viewing = new_size_viewing
        self.size_viewer = new_size_viewer

    def load_from_config(self, conf_str):
        """
        This Class method can convert viewing position configuration string into a proper value stored in the Class obj.

        :param conf_str: configuration string
        :return: (result) update the Class object
        """
        conf_tuple = eval(conf_str)
        self.coor_tl = conf_tuple[0]
        self.size_viewing = conf_tuple[1]
        self.size_viewer = conf_tuple[2]


class ViewingTargetMin:
    def __init__(self):
        self.topleft_coor = None
        self.viewing_size = None
        self.viewer_size = None

    def update_vt(self, top_left, viewing_size, viewer_size):
        self.topleft_coor = top_left
        self.viewing_size = viewing_size
        self.viewer_size = viewer_size


class LoaderConf:
    def __init__(self, conf):
        # Get full path to database
        path_to_database = '/home/siri/OpenHI/framework_src/'  # Establish database path.

        # Developer options
        self.dopDebug = False
        self.dopVerbose = False
        self.imgType = '.jpg'

        # Acquire the necessary information
        uuid, fn_wsi, size = get_manifest(path_to_database, conf['annotation']['manifest_filename'])

        self.pathDatabase = path_to_database + 'data'
        self.uuid = uuid
        self.fn_wsi = fn_wsi
        self.wsiSize = size

        # Variables to be generated later with the class's method
        self.fileNumber = None
        self.fullpathWSI = None
        self.fullpathLabel = None
        self.fullpathBWBoun = None

        self.imgBWBoun = None             # whole bwboun image
        self.imgCurrentBWBounView = None    # bwboun image
        self.imgTLCorr = None

        self.imgWSIWidth = None
        self.imgWSIHeight = None

        self.superpixelList = eval(conf['annotation']['pslv_val'])
        self.superpixelLv = len(self.superpixelList)
        self.current_pslv = 0   # To be moved to function 'gen_filename'

        self.maxGrading = eval(conf['annotation']['max_grading'])

        # OpenSlide Pointer
        self.ptr = None

        self.maxRegionID = None
        self.maxAnnobatch = None

        # Viewing Target (The location that the user is looking at)
        self.vt = ViewingTargetMin()
        self.viewersize = None

        # Viewer state
        self.togBounSwitch = 1  # 1 = on as a default

    def gen_filename(self, file_number):
        print('Generating filename with fileNumber = ' + str(file_number))

        local_option_mode_verbose = self.dopVerbose
        # ----- File name organizing area ----- #
        self.fileNumber = file_number
        fn_len = len(self.fn_wsi[file_number]) - 4  # Select only the filename, excluding file extension.
        filesep = os.sep

        # ----- Report summary: before load the file ----- #
        if local_option_mode_verbose:
            print('\nProject file number: ' + str(file_number))
            print('File UUID: ' + self.uuid[file_number])
            print('Filename of WSI: ' + self.fn_wsi[file_number])

        # Create a path to WSI
        self.fullpathWSI = self.pathDatabase + filesep + self.uuid[file_number] + filesep + self.fn_wsi[file_number]
        if local_option_mode_verbose:
            print('Full path: ' + self.fullpathWSI)

        # Create path to labelling matrix - Currently not in use. # @March: Commented out
        # self.fullpathLabel = self.pathDatabase + filesep + self.uuid[file_number] + filesep + \
        #     self.fn_wsi[file_number][:fn_len] + '_super' + str(self.superpixelList[self.current_pslv])
        if local_option_mode_verbose:
            print('Accessing labelling matrix @: ' + self.fullpathLabel)

        # Creating full bwboundary path for wsi
        if self.fullpathBWBoun is None:
            self.fullpathBWBoun = list()
            for i in range(self.superpixelLv):
                name = (
                        self.pathDatabase + filesep +
                        self.uuid[file_number] + filesep +
                        self.fn_wsi[file_number][:fn_len] + '_super' +
                        str(self.superpixelList[i])
                )
                self.fullpathBWBoun.append(name)

    # load max r_id
    def get_maxRegionID(self, file_number, connector):
        slide_id = file_number + 1
        self.maxRegionID = sql.get_numRegion(connector, slide_id)

    # update max r_id
    def update_maxRegionID(self, new_regionID):
        self.maxRegionID = new_regionID

    # load max anno_batch
    def get_maxAnnobatch(self, db, pslv, anno_id, slide_id):
        """
        Get the maximum annotation batch number during the framework initialization

        :param db: MySQL database object
        :param pslv: Current pslv
        :param sobj: object containing current annotator and slide ID
        :return: (result) update the Class object
        """
        self.maxAnnobatch = sql.get_batch(db, pslv, anno_id, slide_id)
        print('loader.maxAnno = %s' % self.maxAnnobatch)

    # update max anno_batch
    def update_maxAnnobatch(self, new_batch):
        self.maxAnnobatch = new_batch

    def bwbown_add(self, input_bw, pslv):
        if self.imgBWBoun is None:
            self.imgBWBoun = list()
            for i in range(self.superpixelLv):
                self.imgBWBoun.append(None)
        self.imgBWBoun[pslv] = input_bw

    def bwbown_reset(self):
        self.imgBWBoun = None

    def wsi_update_size(self, width, height):
        self.imgWSIWidth = width
        self.imgWSIHeight = height

    def wsi_update_size_reset(self):
        self.imgWSIWidth = None
        self.imgWSIHeight = None

    def wsi_update_ptr_with_filename(self):
        if self.ptr is None:
            pass
        else:
            # Delete old OpenSlide pointer object ???
            self.ptr.close()
            del self.ptr
            print('Old OpenSlide ptr is deleted')

        print('fullpathWSI: ' + self.fullpathWSI)
        self.ptr = openslide.OpenSlide(self.fullpathWSI)
        print('Generate new pointer successful')

    def update_bwboun_tl(self, img_bwboun, coor_tl):
        self.imgCurrentBWBounView = img_bwboun
        self.imgTLCorr = coor_tl

    def update_viewersize(self, new_viewersize):
        self.viewersize = new_viewersize

    def tog_boun(self):
        if self.togBounSwitch == 1:
            self.togBounSwitch = 0
        else:
            self.togBounSwitch = 1

    def get_mpp(self):
        width = self.ptr.properties[openslide.PROPERTY_NAME_MPP_X]
        height = self.ptr.properties[openslide.PROPERTY_NAME_MPP_Y]
        return (width, height)


def get_manifest(path_to_database, filename_manifest):
    # Establish a full path

    # Manifest file is index for /framework_src/data/, it can be configured in the OHI configuration file.
    fn_mani = path_to_database + filename_manifest

    # Load manifest
    file_stream = open(fn_mani, "r")

    num_of_file = 0
    # Get and display header of the file
    mani_header = file_stream.readline()

    # Prepare the parameters
    uuid = []
    fn_wsi = []
    md5 = []
    size = []
    state = []

    for line in file_stream:
        loop_line = line
        # print(len(loop_line))
        loop_a = loop_line.split("\t")
        # print(loop_line.split("\t"))

        uuid.append(loop_a[0])
        fn_wsi.append(loop_a[1])
        md5.append(loop_a[2])
        size.append(loop_a[3])
        state.append(loop_a[4])

        num_of_file += 1

    return uuid, fn_wsi, size


def call_wsi(loader, tl_coor, viewing_size, viewer_size, write_url = False):
    """
    This function can be used to load the WSI image with specified coordinates.
    Version 3 supports calling WSI without reloading OpenSlide pointer object

    :param loader: WSI loader object
    :param tl_coor: top-left coordinate of the desired viewing area
    :param viewing_size: specified viewing area size
    :param viewer_size: specified output image size to be saved
    :return: Cropped and resized (downsampled) WSI as specified viewing are and viewer size.
    """
    # Get pointer object from the loader
    ptr = loader.ptr
    dim = ptr.dimensions
    lv_ds = ptr.level_downsamples

    loader.wsi_update_size(dim[0], dim[1])

    # Calculate downsample factor
    dsf = [0, 0]  # Initialize the downsampling factor list
    dsf[0] = viewing_size[0] / viewer_size[0]  # x
    dsf[1] = viewing_size[1] / viewer_size[1]  # y

    # print('Down sampleing factor (DSF) = ' + str(dsf))
    min_dsf = np.min(dsf)  # min => choose quality over speed

    best_lv = ptr.get_best_level_for_downsample(min_dsf)

    resource_efficient_mode = False
    # Options:
    # True = the image on higher level (best_lv ~= 0) will be loaded with lower quality.
    # False = the image loaded in the viewer will be best resolution possible.

    if resource_efficient_mode:
        if best_lv >= 1:
            best_lv += 1

            if best_lv >= ptr.level_count:
                best_lv = ptr.level_count - 1
        elif best_lv == 0:
            adjust_threshold = ptr.level_downsamples[1]
            if min_dsf >= adjust_threshold/2:
                best_lv = 1

    # print('Best level: ' + str(best_lv) + ', Out of : ' + str(ptr.level_count) + ' levels. ')
    # print('Best downsampling factor: ' + str(lv_ds[best_lv]))

    # Calculate new loading coordinate (same top-left coordinate)
    load_size2 = np.floor(np.divide(viewing_size, lv_ds[best_lv]))
    load_size3 = load_size2.astype(np.int)
    # print('load_coor old: ' + str(viewing_size))
    # print('load_coor new: ' + str(load_size3))

    # Read the image
    img = ptr.read_region(tl_coor, best_lv, load_size3)

    cv_img0 = np.array(img)     # convert PIL to OpenCV image format
    cv_img = cv.cvtColor(cv_img0, cv.COLOR_BGR2RGB)     # convert RGB to BGR

    # Resize the image as specified input
    cv_img_out = cv.resize(cv_img, viewer_size)

    if write_url != False:
        # Save the image
        path_onserver = os.getcwd()  # To establish full path
        path_url = write_url
        # path_img_write = path_onserver + path_url
        path_img_write = path_url
        cv.imwrite(path_img_write, cv_img_out)
        print('Image is written to:' + path_img_write)
        return_val = True
    else:
        return_val = cv_img_out

    return return_val


def load_bwboundary(full_path, option_show_image=0):
    """
    This is a simple function to loads the full logical boundary matrix from the database into the memory

    :param full_path: (string) Full path to the binary boundary image.
    :param option_show_image: (To be depreciated)
    :return:
    """
    # reading_path = full_path + '_bw.png'
    # print('Reading from: ' + reading_path)

    # bw = cv.imread(reading_path, -1)    # Read as indexed image
    # print('The shape of full boundary image: ' + str(bw.shape))
    bw = None

    return bw


def img_crop(bw, tl_coor, viewer_size, op_show_img=0):
    """
    Crop the full binary boundary image according to the given WSI viewing area.

    :param bw: Full binary boundary image. This can be acquired from the Loader object of each annotator.
    :param tl_coor: Top-left coordinate of the viewing area on the WSI.
    :param viewer_size: Viewing size.
    :param op_show_img: (To be depreciated)
    :return: (np-array) Cropped boundary image of the specified viewing area.
    """
    # bw_crop = bw[tl_coor[1]:tl_coor[1] + viewer_size[1], tl_coor[0]:tl_coor[0] + viewer_size[0]]
    bw_crop = None

    return bw_crop


def cvt_point_to_mat(boun_img, point_list_single_pslv):
    """
    Convert list of point with grade (x, y, grade) to image surface area with flood-fill operation.

    :param boun_img: (m-n-1 np-array) boundary image (binary image)
    :param point_list_single_pslv: (n-by-3 list) list of point with grade (x, y, grade)
    :return: (np-array) of "boun_img" size
    """
    # Measure the size of boundary image
    im_height, im_width = boun_img.shape

    # Scan for max grade
    grade_list = list()
    for point in point_list_single_pslv:
        print(point[2])
        grade_list.append(point[2])
    max_grade = max(grade_list)

    # Loop for the amount of pslv the current system has
    # -- Prepare the n-D matrix for storing the grading.
    img_ff_level_val = np.zeros((im_width, im_height, max_grade + 1), np.uint8)

    # Prepare the boundary matrix for flood-fill operation.
    bw_crop_ff = boun_img
    # Generate blank list to store the masking for each grading level
    img_list_masking = list()

    for j in range(max_grade):
        grading_id = j + 1
        img_bwboun_loop_temp = bw_crop_ff.copy()
        # Go through the point list and execute only the point that has the correct grading level value
        for loop_point in point_list_single_pslv:
            # loop_coor structure: (x, y, grading)
            xy = (loop_point[0], loop_point[1])
            loop_grade = loop_point[2]  # grading_id be 1 - 5

            if loop_grade == grading_id:
                cv.floodFill(img_bwboun_loop_temp, None, xy, 255)
            else:
                pass
        img_list_masking.append(img_bwboun_loop_temp)

    img_inv_boundary = cv.bitwise_not(bw_crop_ff)

    for k in range(max_grade):
        # print('k = ' + str(k))
        # Extract the inner-sub-region (no boundaries)
        ret, bw_ff = cv.threshold(img_list_masking[k], 10, 255, cv.THRESH_BINARY)  # Create true binary image
        loop_extract = cv.bitwise_and(img_inv_boundary, bw_ff)  # Remove the boundary from flood-filled image

        # Record the specific grad1ing regions with current pslv.
        img_ff_level_val[:, :, k] = np.add(img_ff_level_val[:, :, k], loop_extract)

    return img_ff_level_val


def gen_image_with_annotation(l, coor_tl, size_viewing, size_viewer, write_url, db, anno_id, slide_id, asess,
                              invalid_points, conf):
    """
    Use input parameters to generate new viewing image and save to specified path and filename.

    :param l: WSI-Annotator loader object
    :param coor_tl: top-left coordinate of the viewing area
    :param size_viewing: size of the viewing image
    :param size_viewer: size of the viewer
    :param write_url: destination filename
    :param db: MySQL database object
    :param anno_id: annotator id
    :param slide_id: slide id
    :param asess: annotation session object
    :param invalid_points: (tuple of tuple(s)) input invalid point(s) if exist
    :param conf: OpenHI system configuration loaded from the system configuration Python-INI file.
    :return: None
    """
    print('Boundary toggle: ' + str(l.togBounSwitch))
    if l.togBounSwitch == 1:
        # Specify color scheme (support maximum of 10 level).
        color_scheme = [
            [0.49, 0.99, 0], [0, 1, 1], [0.54, 0.17, 0.88], [1, 0, 1], [0.82, 0.41, 0.12],
            [1, 0, 0], [0, 1, 0]
        ]

        bw_crop = img_crop(l.imgBWBoun[l.current_pslv], coor_tl, size_viewing, 0)
        # Store bwboun in the loader.
        l.update_bwboun_tl(bw_crop, coor_tl)
        # Modify the masking image.
        #   Get annotated coordinate list
        point_list = sql.get_record(db, coor_tl, size_viewing, l.superpixelLv, anno_id, slide_id)

        print('Modifying layers [total = ', str(str(l.superpixelLv)), ']: ', end='')
        # Loop for the amount of pslv the current system has
        img_ff_all_level = np.zeros((size_viewing[1], size_viewing[0], 3), np.uint8)    # Prepare RGB image to store val
        # -- Prepare the n-D matrix for storing the grading.
        img_ff_level_val = np.zeros((size_viewing[1], size_viewing[0], l.maxGrading + 1), np.uint8)

        for i in range(l.superpixelLv):
            bw_crop_ff = img_crop(l.imgBWBoun[i], coor_tl, size_viewing, 0)
            # report current progress
            print('...', str(i), ' ', end='')
            # Get the ps level specific point list
            loop_point_list = point_list[i]

            # Generate blank image to store the masking for each grading level
            img_list_masking = list()

            for j in range(l.maxGrading):
                grading_id = j + 1
                img_bwboun_loop_temp = bw_crop_ff.copy()
                # Go through the point list and execute only the point that has the correct grading level value
                for loop_point in loop_point_list:
                    # loop_coor structure: (x, y, grading)
                    xy = (loop_point[0], loop_point[1])
                    loop_grade = loop_point[2]  # grading_id be 1 - 5

                    # Convert to local image coordinate
                    input_seedpoint = tuple(np.subtract(xy, coor_tl))

                    if loop_grade == grading_id:
                        cv.floodFill(img_bwboun_loop_temp, None, input_seedpoint, 255)
                    else:
                        pass
                img_list_masking.append(img_bwboun_loop_temp)

            final_ff = np.zeros((size_viewing[1], size_viewing[0], 3), np.uint8)
            img_inv_boundary = cv.bitwise_not(bw_crop_ff)

            for k in range(l.maxGrading):
                # print('k = ' + str(k))
                # Extract the inner-sub-region (no boundaries)
                ret, bw_ff = cv.threshold(img_list_masking[k], 10, 255, cv.THRESH_BINARY)  # Create true binary image
                loop_extract = cv.bitwise_and(img_inv_boundary, bw_ff)  # Remove the boundary from flood-filled image

                # -- Blend with color and alpha
                # Make mask as color
                ff_only_mask = np.zeros_like(final_ff)
                ff_only_mask[:, :, 0] = loop_extract * color_scheme[k][0]
                ff_only_mask[:, :, 1] = loop_extract * color_scheme[k][1]
                ff_only_mask[:, :, 2] = loop_extract * color_scheme[k][2]

                final_ff = np.add(final_ff, ff_only_mask)
                # Record the specific grad1ing regions with current pslv.
                img_ff_level_val[:, :, k] = np.add(img_ff_level_val[:, :, k], loop_extract)

            # bw_img_rgb = cv.cvtColor(bw_crop_ff, cv.COLOR_GRAY2RGB)
            # final_ff_boundary = np.subtract(final_ff, bw_img_rgb)

            img_ff_all_level = np.add(img_ff_all_level, final_ff)

        # Add sub-region boundary based on current pslv
        boundary_rgb = cv.cvtColor(bw_crop, cv.COLOR_GRAY2RGB)
        final_ff_boundary = np.subtract(img_ff_all_level, boundary_rgb)

        # @pargorn: Note: 'final_ff_boundary' is the multi-color annotation image (RGB).
        #                   'boundary_rgb' is 3 layered-bw image with only the boundary.

        # Save multi-layer annotation image
        # Use 'asess' Class object to store the latest BINARY of multi-color annotation image (to further check for
        #   invalid points).
        asess.update_current_annotation(anno_id, img_ff_level_val)

        print('Flood-fill finished')

        # Resize the viewing image to viewer size so that it fits the viewer nicely.
        bw_crop_size = cv.resize(final_ff_boundary, size_viewer, interpolation=cv.INTER_LINEAR_EXACT)
        bw_bound_resize = cv.resize(bw_crop, size_viewer, interpolation=cv.INTER_LINEAR_EXACT)
        print('Image is modified')

        # Create flood-fill only region
        bw_crop_size_mask = cv.cvtColor(bw_crop_size, cv.COLOR_RGB2GRAY)
        ret, mask = cv.threshold(bw_crop_size_mask, 1, 255, cv.THRESH_BINARY)

        # ff_only_mask is the image with only annotated region (no boundary) on it.
        ff_only_mask = cv.bitwise_xor(mask, bw_bound_resize)
        ff_only_mask_inv = cv.bitwise_not(ff_only_mask)

    # Load the WSI with OpenSlide.
    i_wsi = call_wsi(l, coor_tl, size_viewing, size_viewer)

    if l.togBounSwitch == 1:
        i_wsi_masked = cv.bitwise_and(i_wsi, i_wsi, mask=ff_only_mask_inv)

        alpha = 0.7
        beta = 1 - alpha
        img_weighted = cv.addWeighted(bw_crop_size, alpha, i_wsi, beta, 0)
        img_weighted_masked = cv.bitwise_and(img_weighted, img_weighted, mask=ff_only_mask)

        i_blend = cv.add(img_weighted_masked, i_wsi_masked)
        i_blend = cv.subtract(i_blend, cv.cvtColor(bw_bound_resize, cv.COLOR_GRAY2RGB))

    else:
        i_blend = i_wsi.copy()

    if invalid_points:
        i_blend = add_invalid_points(i_blend, coor_tl, size_viewing, size_viewer, invalid_points)

    # Add the grid
    # -- Fetch the smallest allowed grid from the configuration setting
    smallest_allowed_grid = conf['viewer_init']['smallest_grid_size']   # The size is specified in micron.
    i_blend = add_grid(i_blend, size_viewing, size_viewer, smallest_allowed_grid, pixel_size=0.25)

    # Save the image
    path_onserver = os.getcwd()  # To establish full path
    path_url = write_url
    path_img_write = path_onserver + path_url
    cv.imwrite(path_img_write, i_blend)
    print('Image is written to:' + path_img_write)


def internal_gen_annotation_export(l: Any, coor_tl, size_viewing, size_viewer, db, anno_id,
                                   slide_id, asess, conf, show_hide_grid=False) -> Union[Any, Any]:
    """
    Use input parameters to generate new viewing image and save to specified path and filename.

    :param l: WSI-Annotator loader object
    :param coor_tl: top-left coordinate of the viewing area
    :param size_viewing: size of the viewing image
    :param size_viewer: size of the viewer
    :param db: MySQL database object
    :param anno_id: annotator id
    :param slide_id: slide id
    :param asess: annotation session object
    :param conf: OpenHI system configuration loaded from the system configuration Python-INI file.
    :param bool show_hide_grid: option to show or hide grid
    :return: None
    """
    t_switch = l.togBounSwitch
    t_switch = 0
    print('Boundary toggle: ' + str(l.togBounSwitch))
    if t_switch == 1:
        # Specify color scheme (support maximum of 10 level).
        color_scheme = [
            [0.49, 0.99, 0], [0, 1, 1], [0.54, 0.17, 0.88], [1, 0, 1], [0.82, 0.41, 0.12],
            [1, 0, 0], [0, 1, 0]
        ]

        bw_crop = img_crop(l.imgBWBoun[l.current_pslv], coor_tl, size_viewing, 0)
        # Store bwboun in the loader.
        l.update_bwboun_tl(bw_crop, coor_tl)
        # Modify the masking image.
        #   Get annotated coordinate list
        point_list = sql.get_record(db, coor_tl, size_viewing, l.superpixelLv, anno_id, slide_id)

        print('Modifying layers [total = ', str(str(l.superpixelLv)), ']: ', end='')
        # Loop for the amount of pslv the current system has
        img_ff_all_level = np.zeros((size_viewing[1], size_viewing[0], 3), np.uint8)    # Prepare RGB image to store val
        # -- Prepare the n-D matrix for storing the grading.
        img_ff_level_val = np.zeros((size_viewing[1], size_viewing[0], l.maxGrading + 1), np.uint8)

        for i in range(l.superpixelLv):
            bw_crop_ff = img_crop(l.imgBWBoun[i], coor_tl, size_viewing, 0)
            # report current progress
            print('...', str(i), ' ', end='')
            # Get the ps level specific point list
            loop_point_list = point_list[i]

            # Generate blank image to store the masking for each grading level
            img_list_masking = list()

            for j in range(l.maxGrading):
                grading_id = j + 1
                img_bwboun_loop_temp = bw_crop_ff.copy()
                # Go through the point list and execute only the point that has the correct grading level value
                for loop_point in loop_point_list:
                    # loop_coor structure: (x, y, grading)
                    xy = (loop_point[0], loop_point[1])
                    loop_grade = loop_point[2]  # grading_id be 1 - 5

                    # Convert to local image coordinate
                    input_seedpoint = tuple(np.subtract(xy, coor_tl))

                    if loop_grade == grading_id:
                        cv.floodFill(img_bwboun_loop_temp, None, input_seedpoint, 255)
                    else:
                        pass
                img_list_masking.append(img_bwboun_loop_temp)

            final_ff = np.zeros((size_viewing[1], size_viewing[0], 3), np.uint8)
            img_inv_boundary = cv.bitwise_not(bw_crop_ff)

            for k in range(l.maxGrading):
                # print('k = ' + str(k))
                # Extract the inner-sub-region (no boundaries)
                ret, bw_ff = cv.threshold(img_list_masking[k], 10, 255, cv.THRESH_BINARY)  # Create true binary image
                loop_extract = cv.bitwise_and(img_inv_boundary, bw_ff)  # Remove the boundary from flood-filled image

                # -- Blend with color and alpha
                # Make mask as color
                ff_only_mask = np.zeros_like(final_ff)
                ff_only_mask[:, :, 0] = loop_extract * color_scheme[k][0]
                ff_only_mask[:, :, 1] = loop_extract * color_scheme[k][1]
                ff_only_mask[:, :, 2] = loop_extract * color_scheme[k][2]

                final_ff = np.add(final_ff, ff_only_mask)
                # Record the specific grad1ing regions with current pslv.
                img_ff_level_val[:, :, k] = np.add(img_ff_level_val[:, :, k], loop_extract)

            # bw_img_rgb = cv.cvtColor(bw_crop_ff, cv.COLOR_GRAY2RGB)
            # final_ff_boundary = np.subtract(final_ff, bw_img_rgb)

            img_ff_all_level = np.add(img_ff_all_level, final_ff)

        # Add sub-region boundary based on current pslv
        boundary_rgb = cv.cvtColor(bw_crop, cv.COLOR_GRAY2RGB)
        final_ff_boundary = np.subtract(img_ff_all_level, boundary_rgb)

        # @pargorn: Note: 'final_ff_boundary' is the multi-color annotation image (RGB).
        #                   'boundary_rgb' is 3 layered-bw image with only the boundary.

        # Save multi-layer annotation image
        # Use 'asess' Class object to store the latest BINARY of multi-color annotation image (to further check for
        #   invalid points).
        asess.update_current_annotation(anno_id, img_ff_level_val)

        print('Flood-fill finished')

        # Resize the viewing image to viewer size so that it fits the viewer nicely.
        bw_crop_size = cv.resize(final_ff_boundary, size_viewer, interpolation=cv.INTER_LINEAR_EXACT)
        bw_bound_resize = cv.resize(bw_crop, size_viewer, interpolation=cv.INTER_LINEAR_EXACT)
        print('Image is modified')

        # Create flood-fill only region
        bw_crop_size_mask = cv.cvtColor(bw_crop_size, cv.COLOR_RGB2GRAY)
        ret, mask = cv.threshold(bw_crop_size_mask, 1, 255, cv.THRESH_BINARY)

        # ff_only_mask is the image with only annotated region (no boundary) on it.
        ff_only_mask = cv.bitwise_xor(mask, bw_bound_resize)
        ff_only_mask_inv = cv.bitwise_not(ff_only_mask)

    # Load the WSI with OpenSlide.
    i_wsi = call_wsi(l, coor_tl, size_viewing, size_viewer)

    if t_switch == 1:
        i_wsi_masked = cv.bitwise_and(i_wsi, i_wsi, mask=ff_only_mask_inv)

        alpha = 0.7
        beta = 1 - alpha
        img_weighted = cv.addWeighted(bw_crop_size, alpha, i_wsi, beta, 0)
        img_weighted_masked = cv.bitwise_and(img_weighted, img_weighted, mask=ff_only_mask)

        i_blend = cv.add(img_weighted_masked, i_wsi_masked)
        i_blend = cv.subtract(i_blend, cv.cvtColor(bw_bound_resize, cv.COLOR_GRAY2RGB))

    else:
        i_blend = i_wsi.copy()
        img_ff_all_level = i_wsi.copy()

    # if invalid_points:
    #     i_blend = add_invalid_points(i_blend, coor_tl, size_viewing, size_viewer, invalid_points)

    # Add the grid
    if show_hide_grid:
        # -- Fetch the smallest allowed grid from the configuration setting
        smallest_allowed_grid = conf['viewer_init']['smallest_grid_size']   # The size is specified in micron.
        i_blend = add_grid(i_blend, size_viewing, size_viewer, smallest_allowed_grid, pixel_size=0.25)

    # Save the image
    # path_onserver = os.getcwd()  # To establish full path
    # path_url = write_url
    # path_img_write = path_onserver + path_url
    # cv.imwrite(path_img_write, i_blend)
    # print('Image is written to:' + path_img_write)

    # Return list:
    #   - final_ff_boundary = multi-color annotated image
    return i_blend, img_ff_all_level


def add_grid(img, size_viewing, size_viewer, smallest_allowed_grid, pixel_size=0.25):
    """
    This function overlay grid on to the input image. The grid size is made based on the given viewing area value.
    The pixel size (physical size of the glass slide that was captured by one digital pixel) is set to default of 0.25
    micron/pixel which is based on 40x scans. The size of the grid is specified as embedded text on the top-left of
    the output image.

    :param ndarray img: Full OpenCV-BRG color viewer image for overlaying the grid on.
    :param size_viewing: (1x2 list) OpenHI viewing area size.
    :param size_viewer: (1x2 list) OpenHI viewer size.
    :param float pixel_size:
        (optional: default = 0.25) Physical size of one pixel in micron/pixel. (Typically 0.25 for 40x scan, and 0.5
        for 20x scan) Noted that this value can be extracted from the WSI's metadata.
    :return:
        Input image with the overlaying grid if the number of grid does not exceed 10 lines while the maximum grid size
        is 50 um.
    """

    grid_size_ls = [1, 5, 10, 15, 20, 50]   # The set of allowed grid size.
    # Re-configure the grid size list to match the smallest allowed grid size.
    grid_size_ls2 = list()
    grid_size_ls2.append(float(smallest_allowed_grid))
    for list_item in grid_size_ls:
        if list_item > float(smallest_allowed_grid):
            grid_size_ls2.append(list_item)

    max_grid_size_by_10 = 100

    # Calculate the display grid size (based on image width)
    viewing_phys_size_um = size_viewing[0] * pixel_size
    viewing_phys_size_um_by_10 = viewing_phys_size_um / 20      # Divided by the max horizontal grid in the viewer.

    if viewing_phys_size_um_by_10 >= max_grid_size_by_10:
        return img

    # -- Find closest value
    grid_ls_np = np.array(grid_size_ls2)
    grid_ls_rank = np.abs(grid_ls_np - viewing_phys_size_um_by_10)
    grid_size_idx = np.argmin(grid_ls_rank) - 1
    if grid_size_idx < 0:
        grid_size_idx = 0
    grid_size = grid_size_ls2[grid_size_idx]

    # Calculate viewing-to-viewer magnification factor
    mag_factor = size_viewer[0] / size_viewing[0]

    # Generate overlaying layer
    img_overlay = np.zeros((size_viewer[1], size_viewer[0]))
    repeating_interval = (grid_size / pixel_size) * mag_factor

    writing_x = 0
    while writing_x <= size_viewer[0]:
        img_overlay = cv.line(img_overlay, (int(writing_x), 0), (int(writing_x), size_viewer[1]), 1, 1)
        writing_x += repeating_interval

    writing_y = 0
    while writing_y <= size_viewer[1]:
        img_overlay = cv.line(img_overlay, (0, int(writing_y)), (size_viewer[0], int(writing_y)), 1, 1)
        writing_y += repeating_interval

    ret, mask = cv.threshold(img_overlay, 0.5, 1, cv.THRESH_BINARY)     # Create binary image

    # Generate color (RGB) grid
    img_grid_rgb = np.zeros_like(img)
    grid_color = (10, 10, 10)   # Possible value: (0, 0, 0) full black to (255, 255, 255) full white)
    img_grid_rgb[:, :, 0] = img_overlay * grid_color[0]
    img_grid_rgb[:, :, 1] = img_overlay * grid_color[1]
    img_grid_rgb[:, :, 2] = img_overlay * grid_color[2]

    # Alpha blend with masking
    alpha = 0.5     # Possible value: 0 to 1
    beta = 1 - alpha
    img_weighted = cv.addWeighted(img, alpha, img_grid_rgb, beta, 0)
    img_weighted[mask == 0] = 0
    img[mask == 1] = 0
    img_gridded = cv.add(img_weighted, img)

    # Grid size text embedding
    text = 'GRID SIZE: ' + str(grid_size) + ' um'
    font = cv.FONT_HERSHEY_SIMPLEX
    position = (80, 80)     # Position (x, y) of the origin of the text.
    cv.putText(img_gridded, text, position, font, 0.8, (0, 0, 0), 2, cv.LINE_AA)

    return img_gridded


def wsi_get_thumbnail(loader, write_url, viewer_size):
    """Fetch WSI-s thumbnail with this function. """
    ptr = loader.ptr
    assoc_img = ptr.associated_images
    img_pil = assoc_img['thumbnail']

    cv_img0 = np.array(img_pil)     # convert PIL to OpenCV image format
    cv_img = cv.cvtColor(cv_img0, cv.COLOR_BGR2RGB)         # convert RGB to BGR

    # Resize the the specification
    cv.resize(cv_img, viewer_size, cv_img)

    # Save the image
    path_on_server = os.getcwd()  # To establish full path
    path_url = write_url
    path_img_write = path_on_server + path_url
    cv.imwrite(path_img_write, cv_img)

    print('Image is written to:' + path_img_write)


def get_assoc_coor(bwboun_img, tl_coor, seedpoint):
    """
    This function get associated coordinates in the same superpixeled sub-region as the specified seedpoint.

    :param bwboun_img:
    :param tl_coor:
    :param seedpoint:
    :return:
    """
    # Get image viewing image size
    h, w = bwboun_img.shape
    # print('@pargorn: Image size: ' + str(w) + ' by ' + str(h))

    if len(seedpoint) != 2:
        none_output = (None, None)
        return none_output

    # print('1: ' + str(seedpoint))
    # print('2: ' + str(tl_coor))
    if tl_coor is None:
        print('Top-left coordinate is "None", please update image before annotation')
        return []

    # issue (20180817): cannot delete the annotated point, the top-left coordinate y is unusually small
    local_seedpoint = np.subtract(seedpoint, tl_coor)
    if (local_seedpoint[0] > w) or (local_seedpoint[1] > h):
        print('Cannot process because the input top-left coordinate is not correct.')
        print('seedpoint: ' + str(seedpoint))
        print('top-left coordinate: ' + str(tl_coor))
        print('local top-left coordinate: ' + str(local_seedpoint))
        print('local image size: ' + str(w) + ' by ' + str(h) + '\n')
        print('Returning empty list')
        return []

    # print('3: ' + str(local_seedpoint))
    # print('img_size: ' + str(bwboun_img.shape))

    # Make flood-fill image
    ret, bw_boun_binary = cv.threshold(bwboun_img, 10, 255, cv.THRESH_BINARY)
    # imshow(bw_boun_binary)

    img_ff = bwboun_img.copy()
    cv.floodFill(img_ff, None, tuple(local_seedpoint), 255)

    ret, img_ff_bin = cv.threshold(img_ff, 10, 255, cv.THRESH_BINARY)

    ff_only = cv.bitwise_xor(img_ff_bin, bw_boun_binary)

    # Use Numpy to acquire the
    coor_list_loc = np.argwhere(ff_only == 255)
    # Note: @pargorn: np.argwhere(ff_only) should work, but it gives weird output where (x,y) and x is equal to y

    # Create a black coor_list for coor_list swap.
    coor_list_loc_swap = np.zeros_like(coor_list_loc)

    # Swap the coordinate from [row, column] to [x, y] scheme.
    coor_list_loc_swap[:, 0] = coor_list_loc[:, 1]
    coor_list_loc_swap[:, 1] = coor_list_loc[:, 0]

    # Create the global (x, y) of coordinate by adding top-left coordinate back to the local coordinates.
    coor_list_global = np.add(coor_list_loc_swap, tl_coor)

    # Create the checking image
    # img_blank = np.zeros_like(bwboun_img)
    # print(coor_list_loc_swap.shape)
    # for point in coor_list_loc_swap:
    #     # print(point)
    #     # Annotate
    #     x = point[0]
    #     y = point[1]
    #     img_blank[y, x] = 1
    # imshow_sk(img_blank)

    return coor_list_global


def add_invalid_points(img_to_viewer, coor_tl, size_viewing, size_viewer, invalid_points):
    """
    This function add the invalid point annotation to the given resized image (W and H was adjusted to the viewer size).
    The annotation point will added to the image according to the given 'invalid_points'.
    The configuration in the function can be adjusted locally using 'point_*' variable.

    :param img_to_viewer: Final image before viewing at the viewer, noted that the size of this image should be the same
     as given 'size_viewer'.
    :param coor_tl: current top-left coordinate
    :param size_viewing: current viewing size
    :param size_viewer: current viewer size
    :param invalid_points: (tuple of tuple(s)) invalid points should be given as a tuple of (x, y), hence there are many
     points, this function will accept a tuple of those points such as ((x_1, y_1), (x_2, y_2), ..., (x_n, y_n))
    :return: Original image with invalid points added to the image according to the configuration in this function.
    """
    # Configuration
    point_radius = 7            # in pixel (px)
    point_color = (0, 0, 255)   # in RGB value as (r, g, b)

    # Transform the points.
    # --Transform to local points.
    invalid_points_loc = np.subtract(invalid_points, coor_tl)
    # --Calculate down sampling factor.
    ds_factor = size_viewer[0] / size_viewing[0]
    # --Apply the image resizing to invalid points
    invalid_points_loc_resize = np.multiply(invalid_points_loc, ds_factor)
    invalid_points_loc_resize = invalid_points_loc_resize.astype(int)

    for single_invalid_points in invalid_points_loc_resize:
        # OpenCV circle annotation parameters usage:
        # --image, (tuple) center point, radius, color, -1 is fill invert.
        cv.circle(img_to_viewer, tuple(single_invalid_points), point_radius, point_color, -1)

    return img_to_viewer


def check_for_overlapping_regions(checking_img, bwboun_img, top_left_coor, seedpoints, grade):
    """
    This function will check each point in the given list of seedpoint(s) if it overlaps with current annotations.

    :param ndarray checking_img:
        multilayer (up to the number of max grading) image of the annotated region.
        This can be read from `web.get_current_annotation()`
    :param ndarray bwboun_img:
        black-white boundary image showing in the current viewer. This image can be called using img.img_crop(...).
    :param top_left_coor: (1x2 tuple) current top-left coordinate.
    :param seedpoints:
        (nx2 tuple) a list of seedpoints that we want to check if overlapping sub-regions for each of them exist.
        There could be n points to be checked.
    :param int grade: grade that is needed to be checked.
    :return:
        the status of each seedpoints. If the seedpoint overlaps with existing annotated region(s), the corresponding
        location of the list will return 1, and it will return 0 otherwise.
    :rtype: nx1 tuple
    """
    # Construct a matrix with combined gradings
    # -- Get checking_img shape
    checkimg_size = np.shape(checking_img)
    img_combined = np.zeros((checkimg_size[0], checkimg_size[1]), np.uint8)
    for i in range(checkimg_size[2]):
        img_combined = np.add(img_combined, checking_img[:, :, i])

    # Construct a matrix to store the result
    input_seedpoint_length = len(seedpoints)
    output_table_bin = np.zeros((input_seedpoint_length, 2), np.bool_)  # COL1: Other grades, COL2: Given grade

    local_seedpoints = np.subtract(seedpoints, top_left_coor)

    for idx, single_local_seedpoint in enumerate(local_seedpoints):
        # Make flood-fill image based on the given seedpoint.
        flood_fill_image = bwboun_img.copy()
        cv.floodFill(flood_fill_image, None, tuple(single_local_seedpoint), 255)
        bin_ff_noboun = cv.bitwise_xor(flood_fill_image, bwboun_img)
        bin_and = cv.bitwise_and(img_combined, bin_ff_noboun)
        max_img = np.amax(bin_and)

        if max_img == 255:
            output_table_bin[idx, 0] = True

    for idx, single_local_seedpoint in enumerate(local_seedpoints):
        # Make flood-fill image based on the given seedpoint.
        flood_fill_image = bwboun_img.copy()
        cv.floodFill(flood_fill_image, None, tuple(single_local_seedpoint), 255)
        bin_ff_noboun = cv.bitwise_xor(flood_fill_image, bwboun_img)
        bin_and = cv.bitwise_and(checking_img[:, :, grade], bin_ff_noboun)
        max_img = np.amax(bin_and)

        if max_img == 255:
            output_table_bin[idx, 1] = True

    return output_table_bin


# Main area
if __name__ == '__main__':
    print('@OpenHI: This script is not meant to be run as a main script.')
