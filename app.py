import logging
import os

from flask import Flask, request
from flask_cors import CORS, cross_origin
from flask_restful import Api, Resource
from mmaction.apis import init_recognizer, inference_recognizer
from operator import itemgetter

MEDIA_ROOT = "/workspace/tmp/"

API_PREFIX = '/api'

project_dir = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG,
                    format='[%(asctime)s]: {} %(levelname)s %(message)s'.format(os.getpid()),
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[logging.StreamHandler()])
cors = CORS(app, resources={r"/*": {"origins": "*"}})
app.config.update(
    CORS_HEADERS='Content-Type'
)

logger = logging.getLogger()

api = Api(prefix=API_PREFIX)

config_file = 'tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb.py'
checkpoint_file = 'tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb_20220906-2692d16c.pth'
label_file = 'tools/data/kinetics/label_map_k400.txt'
model = init_recognizer(config_file, checkpoint_file, device='cuda:0')  # or device='cuda:0'

def run(model, file_path):
    pred_result = inference_recognizer(model, file_path)

    pred_scores = pred_result.pred_score.tolist()
    score_tuples = tuple(zip(range(len(pred_scores)), pred_scores))
    score_sorted = sorted(score_tuples, key=itemgetter(1), reverse=True)
    top5_label = score_sorted[:5]

    labels = open(label_file).readlines()
    labels = [x.strip() for x in labels]
    return [(labels[k[0]], k[1]) for k in top5_label]

class MMActionAPIView(Resource):
    """POST API class"""
    @cross_origin()
    def post(self):
        """
        (POST)

        upload: <urlstr/image>
        phrase: <str>

        """
        res = {
            "results": {},
            "errors": {},
            "success": False
        }
        # data = request.form
        app.logger.info('new detect')
        upload = request.files["upload"]
        filename = upload.filename
        path = os.path.join(MEDIA_ROOT, filename)
        upload.save(path)

        res["results"] = run(model, path)

        res["success"] = True

        os.remove(path)

        return res


api.add_resource(MMActionAPIView, '/detect')
api.init_app(app)

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=8002, debug=True)