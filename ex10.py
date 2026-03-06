import random

def generate_image(prompt):

    restricted_keywords = ["harmful", "illegal", "bias"]
    if any(keyword in prompt.lower() for keyword in restricted_keywords):
        return "ERROR: Generated content violated safety policy."
    
    return f"Image generated: A beautiful digital painting of {prompt}"

user_prompt = "A serene sunset over a mountain"
print(generate_image(user_prompt))


risky_prompt = "A harmful image of..."
print(generate_image(risky_prompt))