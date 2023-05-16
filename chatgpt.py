import openai
import os

# OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")


def query_gpt(vqa_prompts, vqa_responses, temperature=0):
    prefix1 = "Describe a meme in detail, given the following BLIP-generated response to multiple VQA prompts:\n"
    prefix2 = "Perform OCR on a meme, given the following BLIP-generated response to multiple VQA prompts:\n"
    vqa = "\n".join(
        [
            f"VQA prompt: {p}, VQA response: {r}"
            for p, r in zip(vqa_prompts, vqa_responses)
        ]
    )
    suffix1 = "\nNow, describe what the meme's about:"
    suffix2 = "\nWhat is the exact text in the meme? Say No if there is no text:"
    message1 = [{"role": "user", "content": prefix1 + vqa + suffix1}]
    message2 = [{"role": "user", "content": prefix2 + vqa + suffix2}]
    response1 = openai.ChatCompletion.create(
        model="gpt-4",
        messages=message1,
        temperature=temperature,
    )["choices"][0]["message"]["content"]
    response2 = openai.ChatCompletion.create(
        model="gpt-4",
        messages=message2,
        temperature=temperature,
    )["choices"][0]["message"]["content"]
    return response1, response2
