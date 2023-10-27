import logging
import os
import json

from flask import Flask, request
from flask_cors import CORS, cross_origin
from flask_restful import Api, Resource
from mmaction.apis import init_recognizer, inference_recognizer
from operator import itemgetter

MEDIA_ROOT = "/workspace/media/"

API_PREFIX = "/api"

project_dir = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s]: {} %(levelname)s %(message)s".format(os.getpid()),
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler()],
)
cors = CORS(app, resources={r"/*": {"origins": "*"}})
app.config.update(CORS_HEADERS="Content-Type")

logger = logging.getLogger()

api = Api(prefix=API_PREFIX)

models_dict = {}
with open("models.json") as f:
    models_dict = json.load(f)

print(list(models_dict.keys()))


def run(model, file_path, label_file):
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

        upload: <video>
        model: <str>

        """
        res = {"results": {}, "errors": {}, "success": False}
        data = request.form
        app.logger.info("new detect")
        upload = request.files["upload"]
        filename = upload.filename
        path = os.path.join(MEDIA_ROOT, filename)
        upload.save(path)

        model_name = data["model"]
        config_file = models_dict[model_name]["config_file"]
        checkpoint_file = models_dict[model_name]["chkpt_file"]
        label_file = models_dict[model_name]["label_file"]
        model = init_recognizer(config_file, checkpoint_file, device="cuda:0")

        res["results"] = run(model, path, label_file)

        res["success"] = True

        os.remove(path)

        return res


api.add_resource(MMActionAPIView, "/detect")
api.init_app(app)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8002, debug=True)
