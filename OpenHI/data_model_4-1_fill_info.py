import re
import os
import json
import configparser

import paramiko
import xmltodict
import mysql.connector


def create_connector():
    """
    Connect to MySQL server and database based on preset values. Return database object.
    :return: MySQL database object
    """
    # Check if the local configuration file exists, if not run based on the example file.
    file_checking = '../static/OpenHI_conf.ini'
    checking_status = os.path.isfile(file_checking)

    if not checking_status:
        open_file_name = '../static/OpenHI_conf_example.ini'
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
    mydatabase = mysql.connector.connect(
        host=conf_host,
        port=conf_port,
        user=conf_user,
        passwd=conf_passwd,
        database=conf_database
    )

    return mydatabase


# Copy the data from SFTP server to local host.


def copy_data_to_host(server_path_to_xml, local_path_to_xml):
    """

    :param server_path_to_xml:The path in the server to get XML files.
    :param local_path_to_xml:Local path used to store XML files.
    :return:None.
    """

    # get the configuration information to connect server
    file_checking = '../static/OpenHI_conf.ini'
    checking_status = os.path.isfile(file_checking)

    if not checking_status:
        open_file_name = '../static/OpenHI_conf_example.ini'
    else:
        open_file_name = file_checking

    # Read the OpenHI INI configuration file.
    conf = configparser.ConfigParser()
    conf.read(open_file_name)

    # Read necessary information from dictionary object
    conf_host = conf['ssh']['hostname']
    conf_port = conf['ssh']['port']
    conf_user = conf['ssh']['username']
    conf_passwd = conf['ssh']['password']

    # connect to the server
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(conf_host, conf_port, conf_user, conf_passwd, compress=True)
    # the upper level folder of each XML document in the server
    patient_uuid = client.open_sftp().listdir(server_path_to_xml)

    try:
        for each_uuid in patient_uuid:
            fullpath_of_uuid = server_path_to_xml + '/' + each_uuid
            patient_info = client.open_sftp().listdir(fullpath_of_uuid)
            if re.search('.*' + r'.xml', patient_info[0]):
                fullpath_of_xml = fullpath_of_uuid+'/'+patient_info[0]
                client.open_sftp().get(fullpath_of_xml, local_path_to_xml+'/'+patient_info[0], callback=None)
            elif re.search('.*' + r'.xml', patient_info[1]):
                fullpath_of_xml = fullpath_of_uuid + '/' + patient_info[1]
                client.open_sftp().get(fullpath_of_xml, local_path_to_xml + '/' + patient_info[1], callback=None)
    finally:
        client.close()


# insert clinical data into database.
def clinical_data_insert(local_path_to_xml, pattern):
    """

    :param local_path_to_xml: Local path which stores clinical XML files.
    :param pattern: The pattern of the XML filename from which you need to extract some useful information(like tcga_case_id).
    :return:None.
    """
    xml_list = os.listdir(local_path_to_xml)
    filesep = os.sep

    # Get information from XML documents and store them into mysql.
    for filename in xml_list:
        db = create_connector()

        local_path_of_xml = local_path_to_xml + filesep + filename
        file_object = open(local_path_of_xml)
        # group(2) includes the TCGA code we need in the filename
        tcga_case_id = re.search(pattern, filename, re.I).group(2)

        xml_str = file_object.read()

        # xml to dict
        converted_dict = xmltodict.parse(xml_str)
        # dict to json
        json_str = json.dumps(converted_dict)
        json_str = re.sub(r'@', "", json_str)
        json_str = re.sub(r'#', "", json_str)
        json_str = re.sub(r'\'', " ", json_str)

        sql = "INSERT INTO patient(tcga_case_id,patient_info) VALUES ('%s','%s')" % (tcga_case_id, json_str)
        cursor = db.cursor()
        cursor.execute(sql)
        db.commit()
        db.close()


# insert bio data into database.
def bio_data_insert(local_path_to_xml, pattern):
    """

    :param local_path_to_xml:local_path_to_XML: Local path which stores biospecimen XML files.
    :param pattern:The pattern of the XML filename from which you need to extract some useful information(like tcga_case_id).
    :return:None.
    """
    xml_list = os.listdir(local_path_to_xml)
    filesep = os.sep

    # Get information from XML documents and store them into mysql.
    for filename in xml_list:

        db = create_connector()

        local_path_of_xml = local_path_to_xml + filesep + filename
        file_object = open(local_path_of_xml)

        if os.path.isfile(local_path_of_xml):
            tcga_case_id = re.search(pattern, filename, re.I).group(2)
            # group(2) includes the TCGA code we need in the filename

            try:
                xml_str = file_object.read()
            finally:
                file_object.close()

            # xml to dict
            converted_dict = xmltodict.parse(xml_str)
            # dict to json
            json_str = json.dumps(converted_dict)
            json_str = re.sub(r'@', "", json_str)
            json_str = re.sub(r'#', "", json_str)
            json_str = re.sub(r'\'', " ", json_str)

            sql = "INSERT INTO biospecimen(bio_info,tcga_case_id) VALUES ('%s','%s')" % (json_str, tcga_case_id)

            cursor = db.cursor()
            cursor.execute(sql)
            db.commit()
            db.close()

# insert wsi data into database.


def wsi_data_insert():
    wsi_info = open(os.path.dirname(os.getcwd()) + '/../../framework_src/manifest.txt','r')
    for each_wsi in wsi_info:
        each_wsi_list = each_wsi.split("\t")

        svs_filename = re.search(r'(TCGA-\w{2}-\w{4})(-)([0-9a-zA-Z-]*)(_)([0-9a-zA-Z-]*)(.svs)', each_wsi_list[1])
        if svs_filename is not None:
            uuid = each_wsi_list[0]
            tcga_case_id = svs_filename.group(1)
            tcga_wsi_id = svs_filename.group(1)+svs_filename.group(2)+svs_filename.group(3)
            tcga_wsi_slide_id = svs_filename.group(5)
            filename = svs_filename.group()

            db = create_connector()

            # bio_id_sql="SELECT bio_id from biospecimen WHERE tcga_case_id='"+tcga_case_id+"'"
            #
            # mycursor = db.cursor()
            # mycursor.execute(bio_id_sql)
            # bio_id = mycursor.fetchone()[0]

            sql = "INSERT INTO wsi(tcga_case_id,tcga_wsi_id,tcga_wsi_slide_id,uuid,filename) " \
                  "VALUES ('%s','%s','%s','%s','%s')" % (tcga_case_id, tcga_wsi_id, tcga_wsi_slide_id, uuid, filename)
            cursor = db.cursor()
            cursor.execute(sql)
            db.commit()
            db.close()


def annotator_insert(password,annotator_id=0):   # annotator_id is a default parameter,you can pass it or not.
    db = create_connector()

    if annotator_id == 0:  # If you don't pass it,then this trigger an insert function.

        sql="INSERT INTO annotator(password) VALUES ('%s')"%(password)
        cursor = db.cursor()
        cursor.execute(sql)
        db.commit()
        db.close()

    else:    # If you pass it,then this trigger a password update function.
        sql="UPDATE annotator SET password=\'" + password+"\'" + "where annotator_id=" + str(annotator_id)
        cursor = db.cursor()
        cursor.execute(sql)
        db.commit()
        db.close()


if __name__ == '__main__':
    local_path_to_clinicalXML = '/home/siri/OpenHI/framework_src/clinical_data'
    server_path_to_clinicalXML = "/home1/pp/docs/gdc-downloads/TCGA-KIRC-clinical/gdc_download_20180531_101552"
    clinical_pattern = r'(nationwidechildrens.org_clinical.)(TCGA-\w{2}-\w{4})(.xml)'

    local_path_to_bioXML = os.path.dirname(os.getcwd()) + '/../../framework_src/bio_data'
    server_path_to_bioXML = "/home1/pp/docs/gdc-downloads/TCGA-KIRC_biospec-supplement/gdc_download_20180531_100938"
    bio_pattern = r'(nationwidechildrens.org_biospecimen.)(TCGA-\w{2}-\w{4})(.xml)'

    # copy_data_to_host(server_path_to_clinicalXML, local_path_to_clinicalXML)
    # copy_data_to_host(server_path_to_bioXML, local_path_to_bioXML)
    clinical_data_insert(local_path_to_clinicalXML, clinical_pattern)
    bio_data_insert(local_path_to_bioXML, bio_pattern)
    wsi_data_insert()
    annotator_insert('123456', annotator_id=5)


