from openai import OpenAI

# Paste your API key directly here
client = OpenAI(api_key="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")

def generate_story(prompt):
    """Generates a short story using OpenAI API"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a creative storyteller."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.8
        )

        story = response.choices[0].message.content
        return story

    except Exception as e:
        return f"Error occurred: {e}"


prompt = "Once upon a time, in a small village nestled between the mountains, a young girl discovered a magical key."

story = generate_story(prompt)

print("\nGenerated Story:\n")
print(story)