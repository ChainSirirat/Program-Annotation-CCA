# Web framework for whole-slide image annotation.
#   Created on 20180319
import os
import json
import math
import configparser
import logging

from flask import Flask, render_template, jsonify, request, flash, redirect, url_for, session, g
from forms import LoginForm
#import pyvips
from segmentation_algorithm import SegmentationModel
from segmentation_algorithm import water_image
import cv2
import numpy as np
import tensorflow as tf
import mysql.connector
import shutil

from openhi.legacy import anno_img as img
from openhi.legacy import anno_web as web
from openhi.legacy import anno_sql as sql
from openhi.legacy.anno_web import Clr
from openhi.SqliteConnector import SqliteConnector


from collections import OrderedDict
from flask import Flask, abort, make_response, render_template, url_for
from io import BytesIO
import openslide
from openslide import OpenSlide, OpenSlideError
from openslide.deepzoom import DeepZoomGenerator
import os
from optparse import OptionParser
from threading import Lock

# Internal functions
def _configuration_load():
    # Read framework configuration from INI configuration file.
    # --Check if the local configuration file exists, if not run based on the example file.
    file_checking = 'static/OpenHI_conf.ini'
    checking_status_conf_file = os.path.isfile(file_checking)
    if not checking_status_conf_file:
        open_file_name = 'static/OpenHI_conf_example.ini'
    else:
        open_file_name = file_checking

    # --Read the OpenHI INI configuration file.
    _conf = configparser.ConfigParser()
    _conf.read(open_file_name)

    return _conf


# Initialization of the framework
app = Flask(__name__)  # Initialize Flask
app.config['SECRET_KEY'] = '14a194b474e31af27daf2bc52f3a78bf'
app.config['SLIDE_DIR'] = '/home/siri/OpenHI/framework_src/data'
app.config['SLIDE_CACHE_SIZE'] = 10
app.config['DEEPZOOM_FORMAT'] = 'jpeg'
app.config['DEEPZOOM_TILE_SIZE'] = 254
app.config['DEEPZOOM_OVERLAP'] = 1
app.config['DEEPZOOM_LIMIT_BOUNDS'] = True
app.config['DEEPZOOM_TILE_QUALITY'] = 75

# logging configuration
fmt = '%(asctime)s - %(levelname)s - %(filename)s - %(message)s'
logging.basicConfig(level=logging.DEBUG, format=fmt)
logger = logging.getLogger('app')
handler = logging.FileHandler(filename='app.log')
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter(fmt)
handler.setFormatter(formatter)
logging.getLogger('').addHandler(handler)

conf = _configuration_load()  # Load the configuration file
model = SegmentationModel()
#graph = tf.get_default_graph()
# Check for cache directory
dir_checking = 'static/cache'
checking_status = os.path.exists(dir_checking)

print('Did the checking directory exists?: ' + str(checking_status))  # Print the checking result
if not checking_status:
    os.mkdir(dir_checking)
    print(Clr.BOLD + 'The directory "' + dir_checking + '" did not exists and it has been created.' + Clr.end)
else:
    print(Clr.BOLD + 'The directory "' + dir_checking + '" already exists. No further actions are required.' + Clr.end)

loader = img.LoaderConf(conf)
rand_url = web.WebInter()  # rand_url stands for random URL
db = sql.create_connector()  # db stands for database
s_obj = sql.CurrentStaticObject()  # s_obj stands for static object (not changing during every clicks)
pt = sql.PointDynamicObject()  # pt stands for point
pt_success = sql.PointDynamicObject()  # pt_success stands for point pass check before record
pt_false = sql.PointDynamicObject()  # pt_false stands for point fail to pass check before record
asess = web.AnnotatorSessionList()  # asess stands for annotator session

# Initialize annotators
init_vp = img.ViewingPosition()  # vp is 'viewing position'
manifest_line_number = int(conf['viewer_init']['slide_id']) - 1
init_vp.load_from_config(conf['viewer_init']['viewer_coor'])
asess.init_new_annotator(1, int(conf['viewer_init']['slide_id']), init_vp, sql.re_create_connector(db))
asess.init_new_annotator(2, int(conf['viewer_init']['slide_id']), init_vp, sql.re_create_connector(db))
asess.init_new_annotator(3, int(conf['viewer_init']['slide_id']), init_vp, sql.re_create_connector(db))
asess.init_new_annotator(4, int(conf['viewer_init']['slide_id']), init_vp, sql.re_create_connector(db))
asess.init_new_annotator(5, int(conf['viewer_init']['slide_id']), init_vp, sql.re_create_connector(db))
asess.init_new_annotator(6, int(conf['viewer_init']['slide_id']), init_vp, sql.re_create_connector(db))
# asess.init_new_annotator(3, int(conf['viewer_init']['slide_id']), init_vp, sql.re_create_connector(db))

# Ignore human pixel size while calculating virtual magnification
ignore_hps = conf['virtual_mag']['ignore_human_pixel_size'] == 'True'

# Load the first WSIs (no overlaying layer)
img.call_wsi(asess.get_loader_obj(1), init_vp.coor_tl, init_vp.size_viewing, init_vp.size_viewer)
img.wsi_get_thumbnail(asess.get_loader_obj(1), rand_url.current, init_vp.size_viewer)
print('first image is saved as: ' + rand_url.current)


def slide_ID2TCGA_id(line):
    if type(line) is str:
        line = int(line)
    return loader.uuid[line-1]
def slide_ID2filename(line):
    if type(line) is str:
        line = int(line)
    return loader.fn_wsi[line-1]


root = os.getcwd() + "/../../"
if os.path.exists('static/dzi_data'):
    os.remove('static/dzi_data')
try:
    os.symlink(root + 'framework_src/dzi_data', 'static/dzi_data')
except:
    pass


if not os.path.exists('static/dzi_data/' + slide_ID2TCGA_id(int(conf['viewer_init']['slide_id'])) + '.dzi'):
    print('static/dzi_data/' + slide_ID2TCGA_id(int(conf['viewer_init']['slide_id'])) + '.dzi')
    print("dzi file not exist, please run model svs2dzi first")

@app.before_request
def before_request():
    g.user = None
    if 'user' in session:
        g.user = session['user']
    app.basedir = os.path.abspath(app.config['SLIDE_DIR'])
    config_map = {
        'DEEPZOOM_TILE_SIZE': 'tile_size',
        'DEEPZOOM_OVERLAP': 'overlap',
        'DEEPZOOM_LIMIT_BOUNDS': 'limit_bounds',
    }
    opts = dict((v, app.config[k]) for k, v in config_map.items())
    app.cache = _SlideCache(app.config['SLIDE_CACHE_SIZE'], opts)


@app.route('/', methods=['GET', 'POST'])
def index():
    print('g.user:', g.user)
    if g.user:
        page_anno_id = session['user']
        option_ctr_r_appear = False
        print('slide id:' + str(session['slide_id']))

        if option_ctr_r_appear:
            output_slide = '/static/images/OHI_error_image.png'
        else:
            # URL to the blank image
            output_slide = '/static/images/BlankImage.png'

        if os.path.exists('static/dzi_data/' + slide_ID2TCGA_id(asess.get_slide_id(session['user'])) + '.dzi'):
            slide_url = 'static/dzi_data/' + slide_ID2TCGA_id(asess.get_slide_id(session['user'])) + '.dzi'
        else:
            slide_url = 'dzi_online/'+slide_ID2filename(asess.get_slide_id(session['user'])) + '.dzi'
        return render_template(
            'viewer-v2.html',
            slide_url= slide_url,
            #'static/dzi_data/' + slide_ID2TCGA_id(asess.get_slide_id(session['user'])) + '.dzi',
            # 'mytemplate_with_comments.1.html',
            # slide_url=output_slide  ,
            anno_id=page_anno_id
        )

    return redirect(url_for('login'))


@app.route('/login/', methods=['POST', 'GET'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        session.pop('user', None)
        username = form.username.data
        password = form.password.data

        login_sql = 'SELECT annotator_id,password from annotator WHERE annotator_id=\'' + username + "'"
        db1 = sql.re_create_connector(db)
        mycursor = db1.cursor()
        mycursor.execute(login_sql)
        tmp = mycursor.fetchone()
        mycursor.close()

        if password == tmp[1]:
            session['user'] = username

            # Get current viewing slide of the specific user
            anno_id = username
            slide_id = asess.get_slide_id(anno_id)
            session['slide_id'] = slide_id

            # Initialize the annotator's WSI
            img.call_wsi(asess.get_loader_obj(int(anno_id)), init_vp.coor_tl, init_vp.size_viewing, init_vp.size_viewer)
            img.wsi_get_thumbnail(asess.get_loader_obj(int(anno_id)), rand_url.current, init_vp.size_viewer)
            print('first image is saved as: ' + rand_url.current)

            return redirect(url_for('index'))
        else:
            flash("Password is wrong.Please try again.")
            return render_template('login.html', form=form)

    return render_template("login.html", form=form)


@app.route('/_get_info')
def get_info():
    anno_id = session['user']
    local_loader = asess.get_loader_obj(anno_id)
    toggle_status = local_loader.togBounSwitch

    w = local_loader.imgWSIWidth
    h = local_loader.imgWSIHeight
    sending_pslv_number = local_loader.superpixelLv - 1  # In front-end, pslv starts from 0
    print('Returning image size')
    print('Image size is: ' + str(w) + ' by ' + str(h))
    print('Sending JSON')
    max_image_zoom = conf['viewer_init']['max_image_zoom']
    return jsonify(
        img_width=w,
        img_height=h,
        ps_lv=sending_pslv_number,
        um_per_px=0.25,
        max_image_zoom=max_image_zoom,
        toggle_status=toggle_status
    )


@app.route('/_get_WSI_meta')
def get_wsi_meta():
    # ----- [ Getting slide properties ] ----- #
    anno_id = session['user']
    local_loader = asess.get_loader_obj(anno_id)
    # Print all available properties
    list_of_properties = list(local_loader.ptr.properties)  # get the list
    print('\n This is the start of the list: ')
    for line in list_of_properties:
        print(line)

    return jsonify(magnification=local_loader.ptr.properties['aperio.AppMag'],
                   comment=local_loader.ptr.properties['openslide.comment'],
                   height_level0=local_loader.ptr.properties['openslide.level[0].height'],
                   width_level0=local_loader.ptr.properties['openslide.level[0].width'],
                   objective_power=local_loader.ptr.properties['openslide.objective-power'],
                   vendor=local_loader.ptr.properties['openslide.vendor'])

    # -- Example on how to get the value of the property
    # #   Property from the vendor (aperio.*)
    # print('The magnification of the WSI is: ')
    # print(local_loader.ptr.properties['aperio.AppMag'])
    #
    # #   Standard property from OpenSlide (openslide.*)
    # # (full reference: https://openslide.org/api/python/#standard-properties)
    # print('Comment that OpenSlide can read from: ')
    # print(local_loader.ptr.properties['openslide.comment'])
    #
    # print('Original WSI height (level 0): ')
    # print(local_loader.ptr.properties['openslide.level[0].height'])
    #
    # print('Original WSI downsample(level 0):')
    # print(local_loader.ptr.properties['openslide.level[0].downsample'])


@app.route('/get_patient_meta/', methods=['POST'])
def get_patient_meta():
    # Get slide ID from user cookie-session
    slide_id = session['slide_id']
    print('slide_id:', slide_id)
    json_type = request.get_data()
    string_type = json_type.decode()
    db1 = sql.re_create_connector(db)
    result = sql.get_patient_data(string_type, str(slide_id), db1)

    result = json.dumps(result)

    return result


@app.route('/get_bio_meta/', methods=['POST'])
def get_bio_meta():
    # Get slide ID from user cookie-session
    slide_id = session['slide_id']

    json_type = request.get_data()
    string_type = json.loads(json_type)
    db1 = sql.re_create_connector(db)
    result = sql.get_bio_data(string_type, str(slide_id), db1, True)
    result = json.dumps(result)

    return result

graph = tf.get_default_graph()
@app.route('/_update_image')
def update_image():
    # Request for annotator id.
    anno_id = session['user']
    slide_id = session['slide_id']

    # Get local loader
    local_loader = asess.get_loader_obj(anno_id)

    error_message = 'N/A'
    request_status = 0

    v1 = request.args.get('var1', 0, type=int)  # Top-left x coordinate
    v2 = request.args.get('var2', 0, type=int)  # Top-left y coordinate
    v3 = request.args.get('var3', 0, type=int)  # Bottom-right x
    v4 = request.args.get('var4', 0, type=int)  # Bottom-right y
    v5 = request.args.get('var5', 0, type=int)  # Viewer width (pixel)
    v6 = request.args.get('var6', 0, type=int)  # Viewer height (pixel)
    v7 = request.args.get('var7', 0, type=int)  # Viewer height (pixel)

    # (for format, see /documentation/front-back-end_interface.md)

    # Acquire 'var8' and parse the string into a tuple.

    # Update current pslv in the loader. (single-user)
    img_size_w = local_loader.imgWSIWidth
    img_size_h = local_loader.imgWSIHeight

    loc_tl_coor = (v1, v2)
    loc_viewing_size = (v3 - v1 + 1, v4 - v2 + 1)
    loc_viewer_size = (v5, v6)

    local_loader.update_viewersize(loc_viewing_size)

    # Update URL configuration
    last_url = rand_url.current
    rand_url.get_new_url()
    slide_url = rand_url.current

    annotation_root_folder =  '/home/siri/OpenHI/framework_src/annotation_data/' + slide_ID2TCGA_id(slide_id) + '/'
    if not os.path.exists(annotation_root_folder):
        os.mkdir(annotation_root_folder)

    original_pic_url = annotation_root_folder + 'r' + str(v7) + '.png'
    if not os.path.exists(original_pic_url):
        saving_stat = img.call_wsi(local_loader, loc_tl_coor, loc_viewing_size, loc_viewer_size,
                                   write_url= '/home/siri/OpenHI/' + 'framework_src/annotation_data/' +
                                              slide_ID2TCGA_id(slide_id) + '/' + 'r' + str(v7) + '.png')
    original_pic = cv2.imread(original_pic_url)

    region_image_url = annotation_root_folder + 'r' + str(v7) + '.txt'
    if not os.path.exists(region_image_url):
        global graph
        with graph.as_default():
            print('------------------------------------------'+ original_pic_url + '++++++++++++++++++++++++++++++++')
            mask = model.predict(original_pic_url)
        print(mask)
        region_image = model.water_image(mask)
        np.savetxt(region_image_url, region_image, fmt="%d", delimiter=",")
    else:
        region_image = np.loadtxt(region_image_url, delimiter=",", dtype=int)

    annotator_data_url = annotation_root_folder + 'a' + str(anno_id) + '_r' + str(v7) + '.txt'
    if not os.path.exists(annotator_data_url):
        annotator_data = np.zeros(np.max(region_image) + 1)
        print(annotator_data)
        np.savetxt(annotator_data_url, annotator_data, fmt="%d", delimiter=",")
    else:
        annotator_data = np.loadtxt(annotator_data_url, delimiter=",", dtype=int)

    colour = [tuple([124, 252, 0]), tuple([0, 255, 255]), tuple([137, 43, 224]),
              tuple([255 * 0.82, 255 * 0.41, 255 * 0.12]), tuple([255, 0, 0]), tuple([0, 128, 255])]
    color_scheme = [
        [0.49, 0.99, 0], [0, 1, 1], [0.54, 0.17, 0.88], [0.82, 0.41, 0.12],
        [1, 0, 0], [0, 0.5, 1]
    ]
    mask = np.zeros(original_pic.shape)
    mask[region_image == -1] = tuple([0, 0, 0])
    for i, val in enumerate(annotator_data):
        if i != 1 and val != 0:
            # mask[region_image == i] = colour[val - 1]
            mask[region_image == i] = (original_pic[region_image == i] * 2.7 + colour[val - 1]) / 3.3
        # mask[region_image == i][0] = original_pic[region_image == i][0] *color_scheme[val - 1][0]
        # mask[region_image == i][1] = original_pic[region_image == i][1] *color_scheme[val - 1][1]
        # mask[region_image == i][2] = original_pic[region_image == i][2] *color_scheme[val - 1][2]
        else:
            mask[region_image == i] = original_pic[region_image == i]  # web.rm_file(last_url)
    mask[region_image == -1] = tuple([255, 0, 0])
    bound_size = 4
    mask[:, 0:bound_size] = tuple([255, 0, 0])
    mask[:, 511 - bound_size:511] = tuple([255, 0, 0])
    mask[0:bound_size, :] = tuple([255, 0, 0])
    mask[511 - bound_size:511, :] = tuple([255, 0, 0])

    path_onserver = os.getcwd()  # To establish full path
    write_url = path_onserver + slide_url
    cv2.imwrite(write_url, mask)

    # Normal update image routine
    print(Clr.BOLD + 'Sending JSON' + Clr.END)
    print(Clr.BOLD + 'Refreshing information: ' + Clr.END)
    print('  Annotator ID: ' + str(anno_id))
    print('  Slide ID: ' + str(slide_id))
    return jsonify(
        slide_url=slide_url,
        img_size_width=img_size_w,
        img_size_height=img_size_h,
        top_left_x=loc_tl_coor[0],
        top_left_y=loc_tl_coor[1],
        viewing_size_x=loc_viewing_size[0],
        viewing_size_y=loc_viewing_size[1],
        exit_code=request_status,
        error_message=error_message
    )


@app.route('/_record', methods=['GET', 'POST'])
def record():
    anno_id = session['user']
    slide_id = session['slide_id']

    local_loader = asess.get_loader_obj(anno_id)

    # req_data = request.get_json()
    req_form = request.form

    num_of_points = int(len(req_form) / 4)

    # Access info
    pt_list_att = ('[x]', '[y]', '[grading]', '[region_id]')
    print(req_form)
    annotation_root_folder = '/home/siri/OpenHI/framework_src/annotation_data/' + slide_ID2TCGA_id(slide_id) + '/'
    region_image_url = annotation_root_folder + 'r' + str(req_form['0[region_id]']) + '.txt'
    region_image = np.loadtxt(region_image_url, delimiter=",", dtype=int)
    annotator_data_url = annotation_root_folder + 'a' + str(anno_id) + '_r' + req_form['0[region_id]'] + '.txt'
    annotator_data = np.loadtxt(annotator_data_url, delimiter=",", dtype=int)

    temp = 1
    # In one round, the grading and pslv could be updated only once.
    for i in range(num_of_points):
        temp_new = region_image[int(int(req_form[str(i) + '[y]'])), int(int(req_form[str(i) + '[x]']))]
        if temp != temp_new and temp_new > 1:
            region_image[int(int(req_form[str(i) + '[y]'])), int(int(req_form[str(i) + '[x]']))]
            temp = temp_new
            annotator_data[temp] \
                = int(req_form[str(i) + '[grading]'])
    np.savetxt(annotator_data_url, annotator_data, fmt="%d", delimiter=",")

    return jsonify(
        pt_false_x=[],
        pt_false_y=[],
        region_id=req_form['0[region_id]']
    )


@app.route('/_annotator_id')
def annotator_id():
    cmd = request.args.get('id', 0, type=int)
    print('Received command: ' + str(cmd))
    return jsonify(status='successful', result=session['user'])


# Function for image sub-region boundary switch
@app.route('/_tog_boun')
def tog_boun():
    local_loader = asess.get_loader_obj(session['user'])
    local_loader.tog_boun()

    return jsonify(status='toggled', result=local_loader.togBounSwitch)


@app.route('/logout/', methods=['POST', 'GET'])
def logout():
    session.pop('user', None)
    flash("You have logged out.")
    return redirect(url_for('index'))


@app.route('/_calc_mag')
def calc_mag():
    local_loader = asess.get_loader_obj(session['user'])
    width = request.args.get('swidth', 0, type=int)
    height = request.args.get('sheight', 0, type=int)
    screen_res = (width, height)

    size = request.args.get('ssize', 0, type=float)
    screen_size = size

    distance = request.args.get('sdis', 0, type=float)
    screen_dis = distance

    whratio = screen_size * screen_size / (screen_res[0] * screen_res[0] + screen_res[1] * screen_res[1])
    screen_ps = math.sqrt(whratio) * 2.54  # Note: ps = pixel size

    min_angle = 1.22 * 0.55 / 3000
    human_ps = min_angle * screen_dis

    image_ps = float(local_loader.get_mpp()[0])

    show_w = request.args.get('brx', 0, type=float) - request.args.get('tlx', 0, type=float)
    # show_h = request.args.get('bry', 0, type=float) - request.args.get('tly', 0, type=float)

    img_w = request.args.get('iwidth', 0, type=int)
    # img_h = request.args.get('iheight', 0, type=int)

    res = (screen_ps * 10000 / image_ps) * (img_w / show_w)
    if not ignore_hps:
        res = res * (screen_ps / human_ps)

    return jsonify(status='successful', mag=('%.2f' % res), abs=res)


@app.route('/_change_slide_id')
def _change_slide_id():
    anno_id = int(session['user'])
    old_slide_id = session['slide_id']

    # Get the new slide id
    new_slide_id = request.args.get('id', 0, type=int)

    try:
        # Update new slide id into cookie-session
        session['slide_id'] = new_slide_id
        db1 = sql.re_create_connector(db)

        # Use annotator object to initialize new annotator object with specified slide id.
        asess.set_slide_id(anno_id, new_slide_id, db1)

        # report
        print('Slide ID update report: ')
        print('old slide ID: ' + str(old_slide_id))
        print('new slide ID: ' + str(new_slide_id))

        # Re initialize the WSI
        local_loader = asess.get_loader_obj(anno_id)

        # w = local_loader.imgWSIWidth
        # h = local_loader.imgWSIHeight

        # Update URL configuration
        last_url = rand_url.current
        rand_url.get_new_url()
        slide_url = rand_url.current

        # img.gen_image_with_annotation(local_loader, (0, 0), (100, 100), (100, 100), slide_url, db1,
        #                              anno_id, new_slide_id, asess, (), conf)

        # web.rm_file(last_url)
        # rand_url.current = last_url

        img.call_wsi(asess.get_loader_obj(anno_id), init_vp.coor_tl, init_vp.size_viewing, init_vp.size_viewer)
        img.wsi_get_thumbnail(asess.get_loader_obj(anno_id), rand_url.current, init_vp.size_viewer)

        print('first image is saved as: ' + rand_url.current)
        print(asess.get_loader_obj(1).fullpathWSI)
        print('static/dzi_data/' + slide_ID2TCGA_id(new_slide_id) + '.dzi')
        if not os.path.exists('static/dzi_data/' + slide_ID2TCGA_id(new_slide_id) + '.dzi'):
            print('static/dzi_data/' + slide_ID2TCGA_id(new_slide_id) + '.dzi unfind' )

        return jsonify(status='slide id update successful', result=new_slide_id)
    except AttributeError as e:
        print('Update slide error message: ')
        print(e)

        print(Clr.BOLD + 'End of error message' + Clr.END)
        return jsonify(status='Slide ID can only be a number.')


@app.route('/_get_slide_id')
def _get_slide_id():
    slide_id = session['slide_id']
    return jsonify(slide_id=slide_id)


@app.route('/_update_tb_list')
def _update_tb_list():
    slide_id = session['slide_id']
    annotation_root_folder = '/home/siri/OpenHI/framework_src/annotation_data/' + slide_ID2TCGA_id(slide_id) + '/'
    if not os.path.exists(annotation_root_folder):
        os.mkdir(annotation_root_folder)
    tba_list_db =  annotation_root_folder + 'tba_list.db'
    db = SqliteConnector(tba_list_db)
    tba_result = db.get_RegionID_Centre()
    return jsonify(max_region=len(tba_result), reg_list=tba_result)

    #
    # # Get slide ID from user cookie-session
    # slide_id = session['slide_id']
    # print("SLIDE ID: " + str(slide_id))
    # db1 = sql.re_create_connector(db)  # Re-create database object. (fix long idle problem)
    # print("DB: ")
    # print(db1)
    # tba_result = sql.get_tba_list(db1, slide_id)
    # print(['Returning max:' + str(len(tba_result))])
    # return jsonify(max_region=len(tba_result), reg_list=tba_result)


@app.route('/_report_center')
def _report_center():
    slide_id = session['slide_id']
    annotation_root_folder = '/home/siri/OpenHI/framework_src/annotation_data/' + slide_ID2TCGA_id(slide_id) + '/'
    if not os.path.exists(annotation_root_folder):
        os.mkdir(annotation_root_folder)
    tba_list_db = annotation_root_folder + 'tba_list.db'
    db = SqliteConnector(tba_list_db)
    c_x, c_y = db.get_RegionCentre_By_RegionID()
    return jsonify(x=c_x, y=c_y)

    # slide_id = session['slide_id']
    # db1 = sql.re_create_connector(db)  # Re-create database object. (fix long idle problem)
    # reg_id = request.args.get('reg_id', 0, type=int)
    #
    # c_x, c_y = sql.get_center_tb_reg(db1, slide_id, reg_id)
    # return jsonify(x=c_x, y=c_y)


@app.route('/_add_sw')  # Add sub-window
def _add_sw():
    slide_id = session['slide_id']
    anno_id = int(session['user'])

    allowed_annotator = range(5)
    if anno_id in allowed_annotator:
        x = request.args.get('x', 0, type=int)
        y = request.args.get('y', 0, type=int)

        annotation_root_folder = '/home/siri/OpenHI/framework_src/annotation_data/' + slide_ID2TCGA_id(slide_id) + '/'
        if not os.path.exists(annotation_root_folder):
            os.mkdir(annotation_root_folder)
        tba_list_db = annotation_root_folder + 'tba_list.db'
        db = SqliteConnector(tba_list_db)
        db.incert_RegionCentre(-1,x,y)

        # db1 = sql.re_create_connector(db)  # Re-create database object. (fix long idle problem)
        # sql.add_to_tba_list(db1, slide_id, [x, y])
        message = 'Successfully added diagnostic region'
    else:
        message = 'Only annotator ' + str(allowed_annotator) + ' is allowed to add diagnostic region'

    return jsonify(status=message, num_status=1)


@app.route('/_rm_sw')  # Add sub-window
def _rm_sw():
    slide_id = session['slide_id']
    anno_id = int(session['user'])

    allowed_annotator = range(5)

    if anno_id in allowed_annotator:
        sw_id = request.args.get('sw_id', 0, type=int)

        annotation_root_folder = '/home/siri/OpenHI/framework_src/annotation_data/' + slide_ID2TCGA_id(slide_id) + '/'
        if not os.path.exists(annotation_root_folder):
            os.mkdir(annotation_root_folder)
        tba_list_db = annotation_root_folder + 'tba_list.db'
        db = SqliteConnector(tba_list_db)
        db.delete_RegionCentre(sw_id)

        original_pic_url = annotation_root_folder + 'r' + str(sw_id) + '.png'
        if os.path.exists(original_pic_url):
            os.remove(original_pic_url)
        region_image_url = annotation_root_folder + 'r' + str(sw_id) + '.txt'
        if os.path.exists(region_image_url):
            os.remove(region_image_url)
        for annotator_id in range(6):
            annotator_data_url = annotation_root_folder + 'a' + str(annotator_id) + '_r' + str(sw_id) + '.txt'
            if os.path.exists(annotator_data_url):
                os.remove(annotator_data_url)

        # db1 = sql.re_create_connector(db)  # Re-create database object. (fix long idle problem)
        # sql.rm_from_tba_list(db1, slide_id, sw_id)
        message = 'Successfully removed diagnostic region'
    else:
        message = message = 'Only annotator ' + str(allowed_annotator) + ' is allowed to remove diagnostic region'

    return jsonify(status=message, num_status=1)


@app.route('/record_viewing_pos')  # Add sub-window
def record_viewing_pos():
    slide_id = session['slide_id']
    anno_id = int(session['user'])

    allowed_annotator = 2

    if anno_id is allowed_annotator:

        file_checking = 'static/OpenHI_conf.ini'
        checking_status = os.path.isfile(file_checking)

        if not checking_status:
            open_file_name = 'static/OpenHI_conf_example.ini'
        else:
            open_file_name = file_checking

        # Read the OpenHI INI configuration file.
        conf = configparser.ConfigParser()
        conf.read(open_file_name)

        # Read necessary information from dictionary object
        conf_host = conf['db']['host']
        conf_port = conf['db']['port']
        conf_user = conf['db']['user']
        conf_passwd = conf['db']['passwd']
        conf_database = conf['db']['database']

        # Create database object based on the given configuration
        db = mysql.connector.connect(
            host=conf_host,
            port=conf_port,
            user=conf_user,
            passwd=conf_passwd,
            database=conf_database
        )

        upLeft_X = request.args.get('upLeft_X', 0, type=float)
        upLeft_Y = request.args.get('upLeft_Y', 0, type=float)
        downRight_X = request.args.get('downRight_X', 0, type=float)
        downRight_Y = request.args.get('downRight_Y', 0, type=float)

        sql = ' insert into viewingPosition(slide_id, annotator_id, upLeft_X, upLeft_Y, downRight_X, downRight_Y) ' \
              'value(%s,%s,%s,%s,%s,%s);'

        mycursor = db.cursor()
        mycursor.execute(sql, tuple(
            [str(slide_id), str(anno_id), str(upLeft_X), str(upLeft_Y), str(downRight_X), str(downRight_Y)]))
        db.commit()
        mycursor.close()
        message = 'Successfully record_viewing_pos'
    else:
        message = message = 'Only annotator ' + str(allowed_annotator) + ' is allowed to remove diagnostic region'

    return jsonify(status=message, num_status=1)


@app.route('/save_final_result')  # Add sub-window
def save_final_result():
    slide_id = session['slide_id']
    anno_id = int(session['user'])

    final_result = request.args.get('final_result', 0, type=str)

    annotation_root_folder = '/home/siri/OpenHI/framework_src/annotation_data/' + slide_ID2TCGA_id(slide_id) + '/'
    fianl_result_url = annotation_root_folder + 'a' + str(anno_id) + '_whole_slide.txt'
    print(fianl_result_url)
    f = open(fianl_result_url, 'w')
    f.write(final_result)
    f.close()
    message = 'Successfully saving final_result'

    return jsonify(status=final_result, num_status=1)


@app.route('/read_final_result')  # Add sub-window
def read_final_result():
    slide_id = session['slide_id']
    anno_id = int(session['user'])

    annotation_root_folder = '/home/siri/OpenHI/framework_src/annotation_data/' + slide_ID2TCGA_id(slide_id) + '/'
    if not os.path.exists(annotation_root_folder):
        os.mkdir(annotation_root_folder)
    fianl_result_url = annotation_root_folder + 'a' + str(anno_id) + '_whole_slide.txt'

    if not os.path.exists(fianl_result_url):
        shutil.copy('static/final_result_template.txt', fianl_result_url)
    f = open(fianl_result_url, 'r')
    message = f.read()
    f.close()

    return jsonify(result=message, num_status=1)


class PILBytesIO(BytesIO):
    def fileno(self):
        '''Classic PIL doesn't understand io.UnsupportedOperation.'''
        raise AttributeError('Not supported')


class _SlideCache(object):
    def __init__(self, cache_size, dz_opts):
        self.cache_size = cache_size
        self.dz_opts = dz_opts
        self._lock = Lock()
        self._cache = OrderedDict()

    def get(self, path):
        with self._lock:
            if path in self._cache:
                # Move to end of LRU
                slide = self._cache.pop(path)
                self._cache[path] = slide
                return slide

        osr = OpenSlide(path)
        slide = DeepZoomGenerator(osr, **self.dz_opts)
        try:
            mpp_x = osr.properties[openslide.PROPERTY_NAME_MPP_X]
            mpp_y = osr.properties[openslide.PROPERTY_NAME_MPP_Y]
            slide.mpp = (float(mpp_x) + float(mpp_y)) / 2
        except (KeyError, ValueError):
            slide.mpp = 0

        with self._lock:
            if path not in self._cache:
                if len(self._cache) == self.cache_size:
                    self._cache.popitem(last=False)
                self._cache[path] = slide
        return slide


class _Directory(object):
    def __init__(self, basedir, relpath=''):
        self.name = os.path.basename(relpath)
        self.children = []
        for name in sorted(os.listdir(os.path.join(basedir, relpath))):
            cur_relpath = os.path.join(relpath, name)
            cur_path = os.path.join(basedir, cur_relpath)
            if os.path.isdir(cur_path):
                cur_dir = _Directory(basedir, cur_relpath)
                if cur_dir.children:
                    self.children.append(cur_dir)
            elif OpenSlide.detect_format(cur_path):
                self.children.append(_SlideFile(cur_relpath))


class _SlideFile(object):
    def __init__(self, relpath):
        self.name = os.path.basename(relpath)
        self.url_path = relpath


def _get_slide(path):
    path = slide_ID2TCGA_id(asess.get_slide_id(session['user'])) + '/' + path
    path = os.path.abspath(os.path.join(app.basedir, path))
    # if not path.startswith(app.basedir + os.path.sep):
    #     # Directory traversal
    #     abort(404)
    if not os.path.exists(path):
        print(path)
        abort(404)
    try:
        slide = app.cache.get(path)
        slide.filename = os.path.basename(path)
        return slide
    except OpenSlideError:
        abort(404)

@app.route('/dzi_online/<path:path>.dzi')
def dzi(path):
    slide = _get_slide(path)
    format = app.config['DEEPZOOM_FORMAT']
    resp = make_response(slide.get_dzi(format))
    resp.mimetype = 'application/xml'
    return resp


@app.route('/dzi_online/<path:path>_files/<int:level>/<int:col>_<int:row>.<format>')
def tile(path, level, col, row, format):
    slide = _get_slide(path)
    format = format.lower()
    if format != 'jpeg' and format != 'png':
        # Not supported by Deep Zoom
        abort(404)
    try:
        tile = slide.get_tile(level, (col, row))
    except ValueError:
        # Invalid level or coordinates
        abort(404)
    buf = PILBytesIO()
    tile.save(buf, format, quality=app.config['DEEPZOOM_TILE_QUALITY'])
    resp = make_response(buf.getvalue())
    resp.mimetype = 'image/%s' % format
    return resp


if __name__ == '__main__':
    # Read necessary information from the configuration dictionary object
    framework_host = conf['framework']['host']
    framework_port = int(conf['framework']['port'])
    framework_debug = conf['framework']['debug'] == 'True'
    framework_reloader = conf['framework']['reloader'] == 'True'

    app.run(
        host=framework_host,
        debug=framework_debug,  # debug=True is running in debug mode
        port=framework_port,
        use_reloader=framework_reloader,
        threaded=True  # Currently not in the configuration INI file.
    )
