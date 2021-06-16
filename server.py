from flask import Flask
from file_utils import download_file, delete_file
import os
import requests
from keras import backend as K

#load model
import vgg16_places_365_model 

if not os.path.exists('uploads'):
  os.mkdir('uploads')

app = Flask(__name__)

@app.route('/<filename>')
def predict(filename):
  try:
    #Before prediction
    K.clear_session()
    remote_addr = f"http://images.dwelly.ca/{filename}"
    file_path = download_file(remote_addr)
    prediction = vgg16_places_365_model.predict('uploads/' + file_path)
    #After prediction
    K.clear_session()
    delete_file('uploads/' + filename)
    return prediction, 200
  except requests.exceptions.HTTPError:
    return { "err" : "Image not found", "status": 404}, 404
  except Exception as e:
    print("Error", e)
    return { "err" : "Internall error", "status": 500}, 500

if __name__ == '__main__':
  # app.debug = True 
  app.run(host='0.0.0.0', port=3002)