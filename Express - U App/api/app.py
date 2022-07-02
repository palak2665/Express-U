from flask import Flask
from flask_restful import Resource,Api,reqparse
from prediction import *
import werkzeug

app = Flask(__name__)
api = Api(app)

args = reqparse.RequestParser()
args.add_argument('file',type=werkzeug.datastructures.FileStorage,location='files')

class Prediction(Resource):
  def post(self):
    return Prediction(img) if (img := args.parse_args()) else 404

api.add_resource(Prediction,'/predict/')

if __name__ == '__main__':
  app.run(host='0.0.0.0',port=5000,debug=True)