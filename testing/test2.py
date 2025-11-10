from openai import OpenAI
client = OpenAI()

response = client.responses.create(
    model="gpt-5-nano",
    input="what is diffusion language model?"
)

print(response.output_text)




