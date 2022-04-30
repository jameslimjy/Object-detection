import os
import xml.etree.ElementTree as ET

def xml_parser(file_path):
    '''Rounds float numbers in bndbox to int'''
    xml_files = [name for name in os.listdir(file_path) if name.endswith('.xml')]
    print(xml_files)
    
    for filename in xml_files:
        tree = ET.parse(filename)
        root = tree.getroot()
        for i in root.findall('./object/bndbox/'):
            i.text = str(int(round(float(i.text), 0)))
            
        tree.write(filename)
    return

xml_parser('./')