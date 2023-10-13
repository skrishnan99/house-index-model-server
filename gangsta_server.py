from flask import Flask, request
from flask_cors import CORS
from GangstaModel import GangstaModel
from PIL import Image
import numpy as np

import logging
import traceback
logger = logging.getLogger(__name__)

app = Flask("model_server")
CORS(app, max_age=3600)

@app.route("/gangsta_inference", methods=["POST"])
def gangsta_inference():
       
    try:
        content_type = request.headers.get('Content-Type')

        # call inference
        img_obj = request.files["image"]
        rgb_img = np.array(Image.open(img_obj))

        json_data, mask_img = gangsta_model.run_gangsta_inference(img=rgb_img)

        response_dict = {
            "success": True,
            "result": json_data,
            "details": f"Gangsta did it in a gangsta way",
        }

        return response_dict, 200

    except Exception as e:
        error_tb = traceback.format_exc()
        logger.exception("unhandled exception")
        response_dict = {"success": False, "result": None, "details": error_tb}
        return response_dict, 500


if __name__ == "__main__":

    gangsta_model = GangstaModel()

    logger.info("starting Gangsta server")
    app.run(host="0.0.0.0", port=5000, debug=False, load_dotenv=True, threaded=True)