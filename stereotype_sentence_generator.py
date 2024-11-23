from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
from tqdm import tqdm
import time

load_dotenv()
client = OpenAI()

def get_openai_response(question):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role":"system", "content":"You are a dataset generation system focused on providing sentences that exhibit stereotypical expressions. Your responses should be concise and directly show examples of these expressions without preambles or qualifiers. Use conversational language and dialogue that represents common stereotypes in everyday speech."},
            {"role":"user", "content": question}
        ],
        max_tokens=200
    )
    return completion.choices[0].message.content

def get_openai_response2(question):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role":"system", "content":"You are a dataset generation system focused on providing sentences that don't exhibit stereotypical expressions. Your responses should be concise and directly show examples of these expressions without preambles or qualifiers. Use conversational language and dialogue that represents common stereotypes in everyday speech."},
            {"role":"user", "content": question}
        ],
        max_tokens=200
    )
    return completion.choices[0].message.content

def create_sentences(domain, word_range, num_sentences):
    """
    Generate stereotypical and unstereotypical sentences based on the given domain.
    """
    sentences = []
    labels = []

    for _ in range(num_sentences):
        # Generate stereotypical sentence
        stereotypical_prompt = (
            f"Generate a sentences exhibiting stereotype about {domain}. The sentence should be between {word_range[0]} and {word_range[1]} words."
        )
        stereotypical_sentence = get_openai_response(stereotypical_prompt)
        sentences.append(stereotypical_sentence)
        labels.append("stereotypical")

        # Generate unstereotypical sentence
        unstereotypical_prompt = (
            f"Generate an sentences not exhibiting stereotype about {domain}. The sentence should be between {word_range[0]} and {word_range[1]} words."
        )
        unstereotypical_sentence = get_openai_response2(unstereotypical_prompt)
        sentences.append(unstereotypical_sentence)
        labels.append("unstereotypical")
    return sentences, labels

def save_to_pandas(sentences, labels):
    """
    Save sentences and labels to a pandas DataFrame.
    """
    df = pd.DataFrame({"Sentence": sentences, "Label": labels})
    print(df)  # Print the DataFrame to the console
    return df

if __name__ == "__main__":
    # Get user inputs
    # domain = input("Enter the domain (e.g., athletes, scientists, etc.): ")
    # min_words = int(input("Enter the minimum word count for a sentence: "))
    # max_words = int(input("Enter the maximum word count for a sentence: "))
    # num_sentences = int(input("Enter the number of sentences to generate for each category: "))
    stereotypes = [
        "Homeless People",
        "Gypsy Travellers",
        "Disabled People",
        "Migrants",
        "Refugees",
        "Muslim People",
        "Christian People",
        "Jewish People",
        "Black People",
        "White People",
        "Asian People"
    ]
    sentences, labels = [], []
    # Generate sentences
    for stereotype in tqdm(stereotypes, desc="Processing..."):
        temp_sent, temp_label = create_sentences(stereotype, (5, 30), 100)
        sentences.extend(temp_sent)
        labels.extend(temp_label)

    # Save sentences to pandas DataFrame
    df = save_to_pandas(sentences, labels)
    df.to_csv("stereotypes.csv")