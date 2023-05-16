import openai
import os

# OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")


def fill_template(vqa_prompts, vqa_responses):
    prefix = "Give a caption to a meme, as detailed and long as possible, given the following BLIP-generated response to multiple VQA prompts:\n"
    vqa = "\n".join(
        [
            f"VQA prompt: {p}, VQA response: {r}"
            for p, r in zip(vqa_prompts, vqa_responses)
        ]
    )
    suffix = "\nNow, give a caption to the meme:"
    return [{"role": "user", "content": prefix + vqa + suffix}]


def query_gpt(vqa_prompts, vqa_responses, temperature=0):
    message = fill_template(vqa_prompts, vqa_responses)
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=message,
        temperature=temperature,
    )
    return response
