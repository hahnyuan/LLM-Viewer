from flask import Flask,request
from flask import render_template
from flask_cors import CORS
from get_model_graph import get_model_graph

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})


@app.route("/")
def index():
    return "backend server ready."

@app.route("/get_graph",methods=["POST"])
def get_graph():
    inference_config=request.json["inference_config"]
    nodes, edges,total_results = get_model_graph(
        request.json["model_id"],request.json["hardware"],None,inference_config, 
    )
    return {"nodes": nodes, "edges": edges, "total_results":total_results}

@app.route("/node_info",methods=["POST"])
def node_info():
    pass

if __name__ == "__main__":
    app.run(debug=True)
