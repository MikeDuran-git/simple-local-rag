import streamlit as st
import torch
import textwrap
import numpy as np
import pandas as pd
import fitz
import matplotlib.pyplot as plt
from sentence_transformers import util, SentenceTransformer
from time import perf_counter as timer
import time
# libraries
from CleanRecipeMethods import *
from UnitTestRecipe import *
import warnings
warnings.filterwarnings("ignore")

# Load models and data
pdf_path = "greg_doucette_cookbook_2_0.pdf"
recipe_dataset = pd.read_csv("recipe_dataset.csv")
embedding_model = torch.load("embedding_model.pt")
model = "gpt-4o-mini"
text_chunks_and_embedding_df = pd.read_csv("text_chunks_and_embeddings_df.csv")
text_chunks_and_embedding_df["embedding"] = text_chunks_and_embedding_df["embedding"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
pages_and_chunks = text_chunks_and_embedding_df.to_dict(orient="records")
embeddings = torch.tensor(np.array(text_chunks_and_embedding_df["embedding"].tolist()), dtype=torch.float32).to("cuda" if torch.cuda.is_available() else "cpu")

# Navigation Function
def navigate():
    page = st.sidebar.selectbox("Select Page", ["Main Board", "Create Recipe"])
    return page

# Main Board Layout
def main_board():
    st.title("Main Board")
    st.write("### Welcome to the AI Recipe Generator!")
    if st.button("Create Recipe"):
        st.session_state.page = "Create Recipe"
    if st.button("Create Menu"):
        st.write("Menu creation not implemented yet.")

# Recipe Creation Steps
def create_recipe():
    st.title("Create Recipe")

    st.write("### Step 1: Select Dish Type")
    dish_type = st.selectbox("What kind of dish do you want to make?", ["Breakfast", "Lunch", "Dinner", "Dessert", "Appetizer", "Snacks"])
    st.session_state.dish_type = dish_type

    st.write("### Step 2: Number of People")
    number_of_people = st.slider("For how many people? (between 1 and 8)", 1, 8, 1)
    st.session_state.number_of_people = number_of_people

    st.write("### Step 3: Dietary Preferences and Restrictions")
    diet_options = ["none", "vegan", "vegetarian", "vegetalian", "pescetarian", "no gluten", "no lactose", "no porc"]
    diets = []
    restrictions = []
    for i in range(number_of_people):
        st.write(f"#### Person {i+1}")
        col1, col2 = st.columns(2)
        with col1:
            diet = st.selectbox(f"Diet", diet_options, key=f"diet_{i}")
        with col2:
            restriction = st.text_input(f"Restrictions or dislikes (list ingredients, separated by commas)", key=f"restriction_{i}")
        diets.append(diet)
        restrictions.append([r.strip().lower() for r in restriction.split(',')])
    st.session_state.diets = diets
    st.session_state.restrictions = restrictions

    st.write("### Step 4: Ingredients")
    ingredients = st.text_input("What ingredients would you like the recipe to have? (list up to 5 ingredients, separated by commas)").split(',')
    ingredients = [ing.strip().lower() for ing in ingredients]
    st.session_state.ingredients = ingredients

    st.write("### Step 5: Preparation Time")
    time_options = ["at most 15 min", "between 15-30 min", "30 min or more"]
    max_time = st.selectbox("How much time would you like the recipe to take to prepare?", time_options)
    st.session_state.max_time = max_time

    st.write("### Step 6: Cooking Tools")
    tool_options = ["none", "stovetop", "oven", "blender", "microwave", "automatic cooker", "fryer"]
    cooking_tools = st.multiselect("What are your cooking tools?", tool_options)
    st.session_state.cooking_tools = cooking_tools

    if st.button("Generate Recipe"):
        with st.spinner('Generating recipe...'):
            start_time = timer()
            recipe_prompt, generated_recipe = generate_recipe_based_on_questions_with_RAG(model=model)
            end_time = timer()
            elapsed_time = end_time - start_time
            st.write("### Recipe Prompt")
            st.text_area("Prompt", recipe_prompt, height=300)
            st.write("### Generated Recipe")
            st.success(f'Recipe generated in {elapsed_time:.2f} seconds.')
            st.text_area("Recipe", format_recipe(generated_recipe), height=300)

    if st.button("Return to Main Page"):
        st.session_state.page = "Main Board"

def suggest_alternative(ingredient, diet):
    """Suggests an alternative ingredient based on the diet."""
    meat_alternatives = {
        'chicken': 'vegan chicken',
        'beef': 'vegan beef',
        'pork': 'vegan pork',
        'lamb': 'vegan lamb',
        'turkey': 'vegan turkey',
        'fish': 'vegan fish',
        'shrimp': 'vegan shrimp',
        'sausage': 'vegan sausage',
        'bacon': 'vegan bacon',
        'ham': 'vegan ham'
    }

    alternatives = {
        'vegan': meat_alternatives,
        'vegetarian': {
            'chicken': 'tofu',
            'beef': 'soy protein',
            'fish': 'tempeh',
            'gelatin': 'agar-agar'
        },
        'no gluten': {
            'wheat': 'gluten-free flour',
            'soy sauce': 'tamari',
            'barley': 'quinoa'
        },
        'no lactose': {
            'milk': 'lactose-free milk',
            'cheese': 'lactose-free cheese',
            'cream': 'coconut cream'
        }
    }

    # Handle vegan meat alternatives
    if diet == 'vegan' and ingredient.lower() in meat_alternatives:
        return meat_alternatives[ingredient.lower()]
    return alternatives.get(diet, {}).get(ingredient.lower(), ingredient)

def retrieve_relevant_recipe(query: str,
                    embeddings: torch.tensor,
                    pages_and_chunks: list[dict]=pages_and_chunks,
                    n_resources_to_return: int=5):
    scores, indices = retrieve_relevant_resources(query=query,
                                                  embeddings=embeddings,
                                                  n_resources_to_return=n_resources_to_return)
    
    recipes = []
    for score, index in zip(scores, indices):
        recipes.append(recipe_dataset.loc[recipe_dataset['page_number'] == pages_and_chunks[index]['page_number'], 'text'].values[0])
    
    context= "\n-------------------------------------\n".join(recipes)
    
    return context

def retrieve_relevant_resources(query: str,
                                embeddings: torch.tensor,
                                model: SentenceTransformer=embedding_model,
                                n_resources_to_return: int=5,
                                print_time: bool=True):
    """Embeds a query with model and returns top k scores and indices from embeddings."""
    query_embedding = model.encode(query, convert_to_tensor=True)
    start_time = timer()
    dot_scores = util.dot_score(query_embedding, embeddings)[0]
    end_time = timer()
    if print_time:
        print(f"[INFO] Time taken to get scores on {len(embeddings)} embeddings: {end_time-start_time:.5f} seconds.")
    scores, indices = torch.topk(input=dot_scores, k=n_resources_to_return)
    return scores, indices

def prompt_formatter(query: str, context_items: str) -> str:
    """Formats the prompt for the recipe model."""
    context = context_items
    base_prompt = """
    You are an expert chef. Create a recipe based on the following criteria:
    {context}
    
    The recipe must strictly adhere to the following guidelines:
    1. Respect all dietary restrictions and dislikes. Do not include any ingredients that are listed as restrictions. 
       If a restriction such as "red fruits" is specified, ensure no red fruits are included.
    2. Use the preferred ingredients where possible, substituting alternatives if they conflict with any dietary restrictions or dislikes.
    3. The preparation time should not exceed the specified maximum time.
    4. The recipe should be suitable for the number of people specified.
    5. The recipe should only require the specified cooking tools.

    Please provide the recipe with the following details:
    - Title
    - Ingredients (including quantities)
    - Directions
    - Nutritional information (calories, fat, carbs, protein)
    - Preparation time
    - Category
    - Diet
    """
    prompt = base_prompt.format(context=context, query=query)
    return prompt

def generate_recipe_with_RAG(query, embeddings, pages_and_chunks, model='ft:gpt-3.5-turbo-1106:personal:recipecreatorv7b:9XU88CKN'):
    """Generates a recipe using a Retrieval Augmented Generation (RAG) approach."""
    context = retrieve_relevant_recipe(query=query, embeddings=embeddings, pages_and_chunks=pages_and_chunks)

    prompt = query + "\n Inspire yourself from the retreived recipe if possible:\n" + context

    recipe = send_message_to_recipe_model(prompt, model=model)
    return prompt, recipe

def generate_recipe_based_on_questions_with_RAG(model='ft:gpt-3.5-turbo-1106:personal:recipecreatorv7b:9XU88CKN'):
    dish_type = st.session_state.dish_type
    number_of_people = st.session_state.number_of_people
    diets = st.session_state.diets
    restrictions = st.session_state.restrictions
    ingredients = st.session_state.ingredients
    max_time = st.session_state.max_time
    cooking_tools = st.session_state.cooking_tools

    # Check for incompatible ingredients and suggest alternatives
    compatible_ingredients = []
    meat_alternatives = {
        'chicken': 'vegan chicken',
        'beef': 'vegan beef',
        'pork': 'vegan pork',
        'lamb': 'vegan lamb',
        'turkey': 'vegan turkey',
        'fish': 'vegan fish',
        'shrimp': 'vegan shrimp',
        'sausage': 'vegan sausage',
        'bacon': 'vegan bacon',
        'ham': 'vegan ham'
    }

    for ingredient in ingredients:
        original_ingredient = ingredient
        for diet in diets:
            alternative = suggest_alternative(ingredient, diet)
            if alternative != ingredient:
                st.write(f"For the {diet} diet, replacing {ingredient} with {alternative}.")
            ingredient = alternative
        if ingredient in meat_alternatives.values():
            compatible_ingredients.append(f"(vegan) {ingredient}")
        else:
            compatible_ingredients.append(ingredient)

    # Build the context for the recipe
    context = f"""
    Dish Type: {dish_type}
    Number of People: {number_of_people}
    Diets: {', '.join(diets)}
    Restrictions:
    """
    for i in range(number_of_people):
        context += f"Person {i+1}: {', '.join(restrictions[i])}\n"
    context += f"Preferred Ingredients: {', '.join(compatible_ingredients)}\n"
    context += f"Maximum Preparation Time: {max_time}\n"
    context += f"Available Cooking Tools: {', '.join(cooking_tools)}"

    # Format the prompt for the recipe model
    prompt = f"""
    You are an expert chef. Create a recipe based on the following criteria:
    {context}
    
    The recipe must strictly adhere to the following guidelines:
    1. Respect all dietary restrictions and dislikes. Do not include any ingredients that are listed as restrictions. 
       If a restriction such as "red fruits" is specified, ensure no red fruits are included.
    2. Use the preferred ingredients where possible, substituting alternatives if they conflict with any dietary restrictions or dislikes.
    3. The preparation time should not exceed the specified maximum time.
    4. The recipe should be suitable for the number of people specified.
    5. The recipe should only require the specified cooking tools.

    Please provide the recipe with the following details:
    - Title
    - Ingredients (including quantities)
    - Directions
    - Nutritional information (calories, fat, carbs, protein)
    - Preparation time
    - Category
    - Diet
    """

    # Generate the recipe using the RAG method
    try:
        recipe_prompt, recipe = generate_recipe_with_RAG(prompt, embeddings, pages_and_chunks, model=model)
    except Exception as e:
        print(f"Error generating recipe: {e}")
        return "An error occurred while generating the recipe."

    return recipe_prompt, recipe
def format_recipe(recipe):
    """
    This function formats the recipe string into a more readable format.
    """
    recipe = recipe.replace("<recipe_start>", "\n").replace("<title_start>", "TITLE: ").replace("<ingredient_start>", "INGREDIENTS: \n-").replace("<ingredient_next>", "\n-").replace("<directions_start>", "DIRECTIONS: \n-").replace("<directions_next>", "\n-").replace("<calories_start>", "CALORIES: ").replace("<fatcontent_start>", "FAT: ").replace("<carbohydratecontent_start>", "CARBS: ").replace("<proteincontent_start>", "PROTEIN: ").replace("<prep_time_min_start>", "PREP TIME: ").replace("<type_start>", "TYPE: ").replace("<diet_start>", "DIET: ").replace("<title_end>", "\n").replace("<ingredient_end>", "\n").replace("<directions_end>", "\n").replace("<calories_end>", "\n").replace("<fatcontent_end>", "\n").replace("<carbohydratecontent_end>", "\n").replace("<proteincontent_end>", "\n").replace("<prep_time_min_end>", "\n").replace("<type_end>", "\n").replace("<diet_end>", "\n").replace("<recipe_end>", "\n")

    return recipe

def main():
    if 'page' not in st.session_state:
        st.session_state.page = "Main Board"
    
    page = st.session_state.page
    
    if page == "Main Board":
        main_board()
    elif page == "Create Recipe":
        create_recipe()

if __name__ == "__main__":
    main()
