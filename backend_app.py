from flask import Flask, request
from flask_cors import CORS
from get_ui_graph import analyze_get_ui_graph
from backend_settings import avaliable_hardwares,avaliable_model_ids,avaliable_model_ids_sources
import argparse

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})


@app.route("/")
def index():
    return "backend server ready."


@app.route("/get_graph", methods=["POST"])
def get_graph():
    module_graphs, network_results, hardware_info = analyze_get_ui_graph(
        request.json["model_id"],
        request.json["hardware"],
        request.json["frontend_params_info"],
    )
    return {
        "module_graphs": module_graphs,
        "network_results": network_results,
        "hardware_info": hardware_info,
    }

@app.route("/get_frontend_params_info",methods=["POST"])
def get_frontend_params_info():
    model_id=request.json["model_id"]
    return {
        "frontend_params_info":avaliable_model_ids_sources[model_id][1].frontend_params_info
    }


@app.route("/get_avaliable", methods=["GET"])
def get_avaliable():
    return {
        "avaliable_hardwares": avaliable_hardwares,
        "avaliable_model_ids": avaliable_model_ids,
    }

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args=parser.parse_args()
    host="127.0.0.1" if args.local else "0.0.0.0"
    app.run(debug=args.debug,host=host,port=args.port)
