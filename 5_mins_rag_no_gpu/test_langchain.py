import langchain
from langchain.chains import SimpleSequentialChain
from langchain.prompts import PromptTemplate

# Define the prompts
prompt_template = PromptTemplate.from_template(
    "Write a short note on naxalism?"
)

# Define the LLM model
model = langchain.LLM("mixtral-8x7b-instruct-v0-1")

# Create the chain
chain = SimpleSequentialChain(
    llm=model,
    prompt=prompt_template,
    max_tokens=1024,
    top_p=1,
    n=1,
    stream=False,
    stop="string",
    frequency_penalty=0.0,
)

# Run the chain
response = chain.run()

# Process the response as needed
for c in response:
    print(c["message"]["content"])