from flask import Flask, render_template, request
from models.hallucination import hallucination_guard

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():

    result = None

    if request.method == "POST":

        user_query = request.form["query"]

        result = hallucination_guard(user_query)

    return render_template(
        "index.html",
        result=result
    )

if __name__ == "__main__":

    app.run(debug=True)
    