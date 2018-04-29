import json
import numpy as np
from PIL import Image

try:
    import dicom
except:
    import pydicom as dicom

def load_json(filename):
  with open(filename, 'r') as f:
    return json.load(f)

def load_dataset(filename):
  dataset_info = load_json(filename)
  dataset = []
  for filepath, tags in dataset_info.items():
    entry = tags
    dicom_data = dicom.read_file(filepath)
    entry['image'] = dicom_data.pixel_array
    entry['spatial_resolution'] = float(dicom_data.PixelSpacing[0])
    entry['field_of_view'] = np.multiply(entry['spatial_resolution'], dicom_data.pixel_array.shape).max()
    dataset.append(entry)

  return dataset

data = load_dataset('cardiac.json')

fov_info = load_json('fov_test.json')
fov_test_data = []
for filepath, values in fov_info.items():
  image = Image.open(filepath)
  fov_test_data.append({
    'image': np.array(image),
    'field_of_view': values['field_of_view']
  })

views = [
  'axial',
  'two_chamber',
  'three_chamber',
  'four_chamber',
  'short_axis',
]

sequences = [
  'gre',
  'ssfp',
  'black_blood',
  'de',
]

trajectories = [
  'cartesian',
  'spiral'
]
