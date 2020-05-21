import string
import random
import os
import pickle
import configparser

from openhi.legacy.anno_img import LoaderConf
from openhi.legacy.anno_img import load_bwboundary


class AnnotatorSessionList:
    def __init__(self):
        # Check for annotator_list OpenHI object file
        file_checking = 'static/annotator_list.openhiobj'
        checking_status = os.path.isfile(file_checking)

        print('Did the checking file exists?: ' + str(checking_status))  # Print the checking result
        if not checking_status:
            # If the does not exists
            self.annotator_list = list()
            self.init_vp = None
        else:
            # If the file does exist, load it in!
            with open('static/annotator_list.openhiobj', 'rb') as f:
                self = pickle.load(f)
            print('Continue from the last session.')

    def init_new_annotator(self, anno_id, slide_id, vp, db):
        # Save the 'vp'
        self.init_vp = vp

        # Check if the id exists
        flag_exist = 0      # 0 is not exist, and 1 otherwise
        for entity in self.annotator_list:
            if int(entity.annotator_id) == int(anno_id):
                flag_exist = 1
                print('The given annotator id already exists, please use different id.')

        # Add the new annotator and initialize if not exist.
        if not flag_exist:
            self.annotator_list.append(AnnotatorSession(anno_id, slide_id, vp, db))
            print("New annotator as been added to the system. The annotator id is: " + str(anno_id))

    # Getting different objects
    def get_loader_obj(self, anno_id):
        status_find = 0  # 0 = not found, 1 = found
        output_loader_obj = None
        for entity in self.annotator_list:
            if int(entity.annotator_id) == int(anno_id):
                output_loader_obj = entity.loader
                status_find = 1

        if status_find == 0:
            raise ValueError('Cannot find the given annotator id. Please check if the id is valid.')
        else:
            return output_loader_obj

    def get_viewing_position_obj(self, anno_id):
        status_find = 0  # 0 = not found, 1 = found
        output_vp_obj = None
        for entity in self.annotator_list:
            if int(entity.annotator_id) == int(anno_id):
                output_vp_obj = entity.vp
                status_find = 1

        if status_find == 0:
            raise ValueError('Cannot find the given annotator id. Please check if the id is valid.')
        else:
            return output_vp_obj

    def get_slide_id(self, anno_id):
        status_find = 0     # 0 = not found, 1 = found
        output_id = None
        for entity in self.annotator_list:
            if int(entity.annotator_id) == int(anno_id):
                output_id = entity.slide_id
                status_find = 1

        if status_find == 0:
            raise ValueError('Cannot find the given annotator id. Please check if the id is valid.')
        else:
            return output_id

    def set_slide_id(self, anno_id, new_slide_id, db):
        # Operation: [1] Get id in the list, [2] delete that object, [3] create a new one, [4] replace in the list
        status_record = 0     # 0 = not found, 1 = found
        target_id = None
        for idx, entity in enumerate(self.annotator_list):
            if int(entity.annotator_id) == int(anno_id):
                # Update the slide id value
                entity.set_slide_id(new_slide_id)
                target_id = idx
                status_record = 1
                break

        if status_record == 0:
            raise ValueError('Cannot find the given annotator id. Please check if the id is valid.')
        else:
            self.annotator_list[target_id] = AnnotatorSession(anno_id, new_slide_id, self.init_vp, db)
            return status_record

    def get_current_annotation(self, anno_id):
        record_status = 0
        img = None
        for entity in self.annotator_list:
            if int(entity.annotator_id) == int(anno_id):
                img = entity.current_annotation_bin
                record_status = 1
        if record_status == 0:
            raise ValueError('The given annotator id does not exist in the program. Please contact system admin.')
        else:
            return img

    def update_current_annotation(self, anno_id, new_bin):
        # Go to the correct annotator and update the binary image
        update_status = 0   # 0 is not successful and 1 is successful
        for idx, annotator in enumerate(self.annotator_list):
            # To correctly compare the id with the database, we need to convert it to int-type, I don't know why yet.
            if int(annotator.annotator_id) == int(anno_id):
                self.annotator_list[idx].current_annotation_bin = new_bin
                update_status = 1
        if not update_status:
            raise ValueError('The given annotator id has not been initialized yet.')

    def all_save(self):
        print(self)
        with open('static/annotator_list.openhiobj', 'wb') as f:
            pickle.dump(self, f)


class AnnotatorSession:
    def __init__(self, anno_id, slide_id, vp, db):
        # Read framework configuration from INI configuration file.
        # --Check if the local configuration file exists, if not run based on the example file.
        # file_checking = '/home2/sjb/OpenHI/module/legacy/static/OpenHI_conf.ini'
        file_checking = '/home/siri/OpenHI/module/legacy/static/OpenHI_conf.ini'
        print('file_checking:', file_checking)
        checking_status = os.path.isfile(file_checking)
        print('test AnnotatorSessin checking_status:', checking_status)
        if not checking_status:
            open_file_name = '/home/siri/OpenHI/module/legacy/static/OpenHI_conf_example.ini'
        else:
            open_file_name = file_checking

        # --Read the OpenHI INI configuration file.
        conf = configparser.ConfigParser()
        conf.read(open_file_name)

        self.annotator_id = anno_id
        self.slide_id = slide_id
        self.current_annotation_bin = None

        # Viewing position (vp) Class object
        self.vp = vp

        # Initialise the loader configuration (LoaderConf) object for OpenHI
        self.loader = LoaderConf(conf)
        # Reconfigure the loader
        manifest_line_number = slide_id - 1
        print("slide id: " + str(slide_id))
        self.loader.gen_filename(manifest_line_number)
        print('setting r_id...')
        self.loader.get_maxRegionID(manifest_line_number, db)
        print('setting anno_batch...')

        self.loader.get_maxAnnobatch(db, self.loader.current_pslv, anno_id, slide_id)

        for i in range(self.loader.superpixelLv):
            self.loader.bwbown_add(load_bwboundary(self.loader.fullpathBWBoun[i], 0), i)  # Get bw boundary

        self.loader.wsi_update_ptr_with_filename()

    # def reinit_loader_conf(self, anno_id, slide_id, db):
    #     # Reconfigure the loader
    #     manifest_line_number = slide_id - 1
    #     self.loader.gen_filename(manifest_line_number)
    #     print('setting r_id...')
    #     self.loader.get_maxRegionID(manifest_line_number, db)
    #     print('setting anno_batch...')
    #
    #     self.loader.get_maxAnnobatch(db, self.loader.current_pslv, anno_id, slide_id)
    #
    #     for i in range(self.loader.superpixelLv):
    #         self.loader.bwbown_add(load_bwboundary(self.loader.fullpathBWBoun[i], 0), i)  # Get bw boundary
    #
    #     self.loader.wsi_update_ptr_with_filename()

    def set_slide_id(self, new_slide_id):
        self.slide_id = new_slide_id


class Clr:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

    purple = '\033[95m'
    cyan = '\033[96m'
    darkcyan = '\033[36m'
    blue = '\033[94m'
    green = '\033[92m'
    yellow = '\033[93m'
    red = '\033[91m'
    bold = '\033[1m'
    underline = '\033[4m'
    end = '\033[0m'


class WebInter:
    def __init__(self):
        # Create initial URL
        local_id = id_generator()
        self.imgType = '.jpg'
        self.current = '/static/cache/' + local_id + self.imgType

    def get_new_url_rm(self):
        # Delete the current URL and update with the new one
        local_current_url = self.current
        print(local_current_url)
        try:
            os.remove(local_current_url)
            print('Success')
        except Exception as e:
            print('Removing error: ' + ' ... ' + str(e))

        new_id = id_generator()
        self.current = '/static/cache/' + new_id + self.imgType

    def get_new_url(self):
        new_id = id_generator()
        self.current = '/static/cache/' + new_id + self.imgType


def rm_file(url):
    print('Deleting: ' + url + ' ||| ->  ', end='')   # Print without creating a new line
    try:
        c_dir = os.getcwd()
        os.remove(c_dir + url)
        print('Successfully remove previous file.')
    except Exception as e:
        print('Removing error: ' + ' ... ' + str(e))


def id_generator(size=20, chars=string.ascii_letters + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))
