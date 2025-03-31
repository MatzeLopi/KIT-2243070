import queue
import threading
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from flask import Flask, render_template, Response

from models import hammerstein, utils

app = Flask(__name__)

# Queue to store progress updates
progress_queue = queue.Queue()

# Callback function to push updates to the queue
def progress_callback(msg: str):
    progress_queue.put(msg)

# Flask route: renders the frontend
@app.route('/')
def index():
    return render_template('index.html')

# Flask route: streams progress updates via SSE
@app.route('/progress')
def progress():
    def generate():
        while True:
            msg = progress_queue.get()
            yield f"data: {msg}\n\n"
    return Response(generate(), mimetype='text/event-stream')

# Optimizer function to run in a background thread
def run_optimizer():
    data1 = utils.read_mat(r'C:\Users\matze\workspaces\KIT-2243070\raw\ex1_data.mat')
    data2 = utils.read_mat(r'C:\Users\matze\workspaces\KIT-2243070\raw\ex2_data.mat')

    data1.y = data1.y.flatten()
    data1.u = data1.u.flatten()
    data1.t = data1.t.flatten()

    t = jnp.array(data2.t1)
    y = jnp.array(data2.y1)
    u = jnp.array(data2.u1)

    hp = hammerstein.optimize(
        data1.y, data1.u,
        (7, 8), (7, 8),
        (1, 2), (1, 2),
        (0.1, 0.9),
        callback=progress_callback
    )

    with open("result.txt", "w") as f:
        f.write(f"Best parameters:\n {hp}\n")
    
    progress_queue.put("Optimization complete!")

# Run the Flask app and start the optimizer in a separate thread
if __name__ == '__main__':
    threading.Thread(target=run_optimizer, daemon=True).start()
    app.run(debug=True, threaded=True)