import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List


def parse_pascal_voc_xml(xml_path: Path) -> Dict:
    """
    Parse Pascal VOC XML annotation file.
    Returns dictionary with image info and annotations.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Extract image information
    filename = root.find('filename').text
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    depth = int(size.find('depth').text)
    
    # Extract objects
    objects = []
    for obj in root.findall('object'):
        obj_data = {
            'name': obj.find('name').text,
            'difficult': int(obj.find('difficult').text) if obj.find('difficult') is not None else 0,
            'truncated': int(obj.find('truncated').text) if obj.find('truncated') is not None else 0,
        }
        
        # Check for bounding box
        bndbox = obj.find('bndbox')
        if bndbox is not None:
            obj_data['bbox'] = {
                'xmin': int(bndbox.find('xmin').text),
                'ymin': int(bndbox.find('ymin').text),
                'xmax': int(bndbox.find('xmax').text),
                'ymax': int(bndbox.find('ymax').text)
            }
        
        # Check for polygon segmentation
        polygon = obj.find('polygon')
        if polygon is not None:
            points = []
            for i in range(1, 100):  # Maximum 100 points
                x_elem = polygon.find(f'x{i}')
                y_elem = polygon.find(f'y{i}')
                if x_elem is not None and y_elem is not None:
                    points.append((float(x_elem.text), float(y_elem.text)))
                else:
                    break
            obj_data['polygon'] = points
        
        objects.append(obj_data)
    
    return {
        'filename': filename,
        'width': width,
        'height': height,
        'depth': depth,
        'objects': objects
    }

def get_dataset_files(dataset_path: Path) -> Dict[str, List[Path]]:
    """
    Get all image and annotation files from a Pascal VOC dataset.
    Returns dictionary with train/test/valid splits.
    """
    splits = {}
    for split in ['train', 'test', 'valid']:
        split_dir = dataset_path / split
        if split_dir.exists():
            # Get all .jpg files
            images = list(split_dir.glob('*.jpg'))
            # Get corresponding .xml files
            annotations = []
            for img_path in images:
                xml_path = img_path.with_suffix('.xml')
                if xml_path.exists():
                    annotations.append(xml_path)
            
            splits[split] = {
                'images': sorted(images),
                'annotations': sorted(annotations)
            }
    
    return splits
