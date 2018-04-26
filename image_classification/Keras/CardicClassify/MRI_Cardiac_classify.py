import json

def load_json(filename):
  with open(filename, 'r') as f:
    return json.load(f)


def load_dataset(filename):
  dataset_info = load_json(filename)  #deserialized data into JSON dictionary again.  key
#is the ID/filepath.  value is another dictionary that contains the values of trajectory,
#view, and sequence
  #print(dataset_info.items())
  dataset = []
  for filepath, tags in dataset_info.items():  #.items() gives us a list of tuples.  each tuple
        #is a single key-value pair.  string is the key, dictionary is the value

    # tags is the dictionary of the three category values
    dataset.append({ "filepath" : filepath, "trajectory" : tags['trajectory'] })


    #dicom_data = dicom.read_file(filepath)
    #entry['image'] = dicom_data.pixel_array
    #entry['spatial_resolution'] = float(dicom_data.PixelSpacing[0])
    #entry['field_of_view'] = np.multiply(entry['spatial_resolution'], dicom_data.pixel_array.shape).max()
    # we add 3 columns from the above 3 lines to the entry array which originally only contained the 3
    #categories wer'e trying to sort
    #dataset.append(entry)

  return dataset  #dataset is a list of all the entry elements for each file.  each is a
#dictionary of 6 keys

dataset = load_dataset('./CardiacClassification/cardiac.json')
print(len(dataset))
