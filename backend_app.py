from flask import Flask
from flask import render_template
from flask_cors import CORS
from get_model_graph import get_model_graph

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/')
def get_graph():
    nodes=[
        {"label":"LayerNorm", "id": "1", "description":"OPs:xx,\nvv:ds", "panels":[{"title":"OPs","value":"59M"},{"title":"Access","value":"34MB"}]},
        {"label":"q_proj", "id": "2", "panels":[{"title":"OPs","value":"59M"},{"title":"Access","value":"34MB"}]},
    ]

    edges=[
        {"source":"1","target":"2"}
    ]
    nodes,edges=get_model_graph("meta-llama/Llama-2-7b-hf",1,2048,"configs/llama-2.py")
    return {"nodes":nodes, "edges":edges}

if __name__ == '__main__':
    app.run(debug=True)