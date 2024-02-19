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
    nodes, edges = get_model_graph(
        "meta-llama/Llama-2-7b-hf","nvidia_V100","configs/Llama.py",inference_config, 
    )
    return {"nodes": nodes, "edges": edges}

if __name__ == "__main__":
    app.run(debug=True)
