from flask import Flask, request
from flask import render_template
from flask_cors import CORS
from get_model_graph import get_model_graph
from backend_settings import avaliable_hardwares,avaliable_models
import argparse

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})


@app.route("/")
def index():
    return "backend server ready."


@app.route("/get_graph", methods=["POST"])
def get_graph():
    inference_config = request.json["inference_config"]
    nodes, edges, total_results, hardware_info = get_model_graph(
        request.json["model_id"],
        request.json["hardware"],
        None,
        inference_config,
    )
    return {
        "nodes": nodes,
        "edges": edges,
        "total_results": total_results,
        "hardware_info": hardware_info,
    }

@app.route("/get_avaliable", methods=["GET"])
def get_avaliable():
    return {
        "avaliable_hardwares": avaliable_hardwares,
        "avaliable_models": avaliable_models,
    }

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args=parser.parse_args()
    host="127.0.0.1" if args.local else "0.0.0.0"
    app.run(debug=args.debug,host=host,port=args.port)
