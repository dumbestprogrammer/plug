from flask import Flask, request, jsonify
from langchain_core.prompts import PromptTemplate
from langchain.llms import HuggingFaceHub
import os

app = Flask(__name__)

# Set your Hugging Face API key
os.environ["myplug"] = "hf_xjHxWCkX0nHkIMQCRXnpRGMHMN0oHnFySg"

# Load LLM from Hugging Face
llm = HuggingFaceHub(
    repo_id="codellama/CodeLlama-7b-Instruct-hf",  # Code Llama 7B Instruct model
    model_kwargs={"temperature": 0.5, "max_length": 512}
)


# Create a structured prompt template
prompt_template = PromptTemplate.from_template(
    "Analyze the following Java code and suggest improvements:\n\n{code}\n\nProvide detailed suggestions."
)

@app.route("/analyze", methods=["POST"])
def analyze_code():
    data = request.json
    code = data.get("code", "")

    if not code.strip():
        return jsonify({"error": "No Java code provided"}), 400

    # Use LangChain to structure the request
    prompt = prompt_template.format(code=code)
    response = llm.invoke(prompt)

    return jsonify({"suggestions": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)