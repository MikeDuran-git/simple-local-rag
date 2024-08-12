import streamlit as st
import torch
import numpy as np
import pandas as pd
import json
from sentence_transformers import util, SentenceTransformer
from time import perf_counter as timer
import tiktoken
from CleanRecipeMethods import *
from UnitTestRecipe import *
import warnings

warnings.filterwarnings("ignore")

# Load models and data
pdf_path = "greg_doucette_cookbook_2_0.pdf"
model="ft:gpt-4o-mini-2024-07-18:personal:recipecreatorv8:9vLqipXh" #replace with gpt-4o-mini to have more accurate data.
recipe_dataset = pd.read_csv("recipe_dataset.csv")
embedding_model = torch.load("embedding_model.pt")
text_chunks_and_embedding_df = pd.read_csv("text_chunks_and_embeddings_df.csv")
text_chunks_and_embedding_df["embedding"] = text_chunks_and_embedding_df["embedding"].apply(
    lambda x: np.fromstring(x.strip("[]"), sep=" ")
)
pages_and_chunks = text_chunks_and_embedding_df.to_dict(orient="records")
# Load embeddings to the correct device
device = "cuda" if torch.cuda.is_available() else "cpu"
embeddings = torch.tensor(
    np.array(text_chunks_and_embedding_df["embedding"].tolist()),
    dtype=torch.float32
).to(device)


# Load saved recipes
def load_recipes():
    try:
        with open("saved_recipes.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []


# Save recipes
def save_recipes(recipes):
    with open("saved_recipes.json", "w") as f:
        json.dump(recipes, f)


# Load saved menus
def load_menus():
    try:
        with open("saved_menus.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []


# Save menus
def save_menus(menus):
    with open("saved_menus.json", "w") as f:
        json.dump(menus, f)


# Load saved plans
def load_plans():
    try:
        with open("saved_plans.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []


# Save plans
def save_plans(plans):
    with open("saved_plans.json", "w") as f:
        json.dump(plans, f)


# Navigation Function
def navigate():
    page = st.sidebar.selectbox("Select Page", ["Main Board", "Create Recipe", "Create Menu", "My Recipes", "My Menus", "Create Weekly Plan", "My Plans"])
    return page


# Main Board Layout
def main_board():
    st.title("Recipe CreAItor")
    st.markdown("### Welcome to your AI buffed Recipe generator!")
    st.markdown("_Don't laugh, your name isn't any better!_")
    st.caption("Create New Recipes, Menus, and Weekly Plans")
    st.markdown("---")

    st.header("Create New")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Create Recipe"):
            st.session_state.page = "Create Recipe"
            st.rerun()
    with col2:
        if st.button("Create Menu"):
            st.session_state.page = "Create Menu"
            st.rerun()
    with col3:
        if st.button("Create Weekly Plan"):
            st.session_state.page = "Create Weekly Plan"
            st.rerun()
    
    st.markdown("---")
    st.header("My Creations")
    col4, col5, col6 = st.columns(3)
    with col4:
        if st.button("My Recipes"):
            st.session_state.page = "My Recipes"
            st.rerun()
    with col5:
        if st.button("My Menus"):
            st.session_state.page = "My Menus"
            st.rerun()
    with col6:
        if st.button("My Weekly Plans"):
            st.session_state.page = "My Plans"
            st.rerun()


def calculate_token_count(text, model_name=model):
    encoder = tiktoken.encoding_for_model(model_name)
    tokens = encoder.encode(text)
    return len(tokens)


def suggest_alternative(ingredient, diet):
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

    if diet == 'vegan' and ingredient.lower() in meat_alternatives:
        return meat_alternatives[ingredient.lower()]
    return alternatives.get(diet, {}).get(ingredient.lower(), ingredient)


def retrieve_relevant_recipe(query: str,
                             embeddings: torch.tensor,
                             pages_and_chunks: list[dict] = pages_and_chunks,
                             n_resources_to_return: int = 5):
    scores, indices = retrieve_relevant_resources(query=query,
                                                  embeddings=embeddings,
                                                  n_resources_to_return=n_resources_to_return)

    recipes = []
    for score, index in zip(scores, indices):
        recipes.append(recipe_dataset.loc[recipe_dataset['page_number'] == pages_and_chunks[index]['page_number'], 'text'].values[0])

    context = "\n-------------------------------------\n".join(recipes)

    return context


def retrieve_relevant_resources(query: str,
                                embeddings: torch.tensor,
                                model: SentenceTransformer = embedding_model,
                                n_resources_to_return: int = 5,
                                print_time: bool = True):
    device = embeddings.device
    query_embedding = model.encode(query, convert_to_tensor=True).to(device)

    start_time = timer()
    dot_scores = util.dot_score(query_embedding, embeddings)[0]
    end_time = timer()
    if print_time:
        print(f"[INFO] Time taken to get scores on {len(embeddings)} embeddings: {end_time - start_time:.5f} seconds.")
    scores, indices = torch.topk(input=dot_scores, k=n_resources_to_return)
    return scores, indices


def prompt_formatter(query: str, context_items: str) -> str:
    context = context_items
    base_prompt = """
    You are an expert culinary chef with more than 10 years of experience. Create a recipe based on the following criteria:
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


def generate_recipe_based_on_questions_with_RAG():
    dish_type = st.session_state.dish_type
    number_of_people = st.session_state.number_of_people
    diets = st.session_state.diets
    restrictions = st.session_state.restrictions
    ingredients = st.session_state.ingredients
    max_time = st.session_state.max_time
    cooking_tools = st.session_state.cooking_tools

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

    context = f"""
    Dish Type: {dish_type}
    Number of People: {number_of_people}
    Diets: {', '.join(diets)}
    Restrictions:
    """
    for i in range(number_of_people):
        context += f"Person {i + 1}: {', '.join(restrictions[i])}\n"
    context += f"Preferred Ingredients: {', '.join(compatible_ingredients)}\n"
    context += f"Maximum Preparation Time: {max_time}\n"
    context += f"Available Cooking Tools: {', '.join(cooking_tools)}"

    prompt = f"""
    You are an expert culinary chef. Create a recipe based on the following criteria:
    {context}

    The recipe must strictly adhere to the following guidelines:
    1. Respect all dietary restrictions and dislikes. Do not include any ingredients that are listed as restrictions. If a restriction such as "red fruits" is specified, ensure no red fruits are included.
    2. Use the preferred ingredients where possible, substituting alternatives if they conflict with any dietary restrictions or dislikes.
    3. The preparation time should not exceed the specified maximum time.
    4. The recipe should be suitable for the number of people specified.
    5. The recipe should only require the specified cooking tools.

    For the recipe, provide the following details:
    - Title
    - Ingredients (including quantities)
    - Directions
    - Nutritional information for the whole recipe (calories, fat, carbs, protein)
    - Preparation time
    - Category
    - Diet
    """

    RAG_context = retrieve_relevant_recipe(query=f"{', '.join(compatible_ingredients)}", embeddings=embeddings, pages_and_chunks=pages_and_chunks)
    try:
        recipe_prompt = prompt + "\nInspire yourself from the these recipe to make a high protein and healthy recipe if possible:\n" + RAG_context

        recipe = send_message_to_recipe_model(recipe_prompt, model=model)
    except Exception as e:
        print(f"Error generating recipe: {e}")
        return "An error occurred while generating the recipe."

    return recipe_prompt, recipe


def format_recipe(recipe):
    recipe = recipe.replace("<recipe_start>", "").replace("<title_start>", "TITLE: ").replace("<ingredient_start>", "INGREDIENTS: \n-").replace("<ingredient_next>", "\n-").replace("<directions_start>", "DIRECTIONS: \n-").replace("<directions_next>", "\n-").replace("<calories_start>", "CALORIES: ").replace("<fatcontent_start>", "FAT: ").replace("<carbohydratecontent_start>", "CARBS: ").replace("<proteincontent_start>", "PROTEIN: ").replace("<prep_time_min_start>", "PREP TIME: ").replace("<type_start>", "TYPE: ").replace("<diet_start>", "DIET: ").replace("<title_end>", "\n").replace("<ingredient_end>", "\n").replace("<directions_end>", "\n").replace("<calories_end>", "\n").replace("<fatcontent_end>", "\n").replace("<carbohydratecontent_end>", "\n").replace("<proteincontent_end>", "\n").replace("<prep_time_min_end>", "\n").replace("<type_end>", "\n").replace("<diet_end>", "\n").replace("<recipe_end>", "")
    return recipe


# Recipe Creation Steps
def create_recipe():
    if st.button("Return to Main Page"):
        st.session_state.page = "Main Board"
        st.rerun()
    st.title("Create Recipe")
    st.write("### Step 1: Select Dish Type")
    dish_type = st.selectbox("What kind of dish do you want to make?", ["Breakfast", "Lunch", "Dinner", "Dessert", "Appetizer", "Snacks"])
    st.session_state.dish_type = dish_type

    st.write("### Step 2: Number of People")
    number_of_people = st.slider("For how many people? (between 1 and 8)", 1, 8, 1)
    st.session_state.number_of_people = number_of_people

    st.write("### Step 3: Dietary Preferences and Restrictions")
    diet_options = ["none", "vegan", "vegetarian", "pescetarian", "no gluten", "no lactose", "no pork"]
    diets = []
    restrictions = []
    for i in range(number_of_people):
        st.write(f"#### Person {i + 1}")
        col1, col2 = st.columns(2)
        with col1:
            diet = st.selectbox(f"Diet", diet_options, key=f"diet_{i}")
        with col2:
            restriction = st.text_input(f"Restrictions or dislikes (ex: 'no red fruits, no nuts...')", key=f"restriction_{i}")
        diets.append(diet)
        restrictions.append([r.strip().lower() for r in restriction.split(',')])
    st.session_state.diets = diets
    st.session_state.restrictions = restrictions

    st.write("### Step 4: Ingredients")
    ingredients = st.text_input("What ingredients would you like the recipe to have? (list up to 5 ingredients separated by commas, you can also just say 'fruits' or 'vegetables')").split(',')
    ingredients = [ing.strip().lower() for ing in ingredients]
    st.session_state.ingredients = ingredients

    st.write("### Step 5: Preparation Time")
    time_options = ["at most 15 min", "between 15-30 min", "30 min or more"]
    max_time = st.selectbox("How much time would you like the recipe to take to prepare?", time_options)
    st.session_state.max_time = max_time

    st.write("### Step 6: Cooking Tools")
    tool_options = ["stovetop", "oven", "blender", "microwave", "automatic cooker", "fryer"]
    cooking_tools = st.multiselect("What are your cooking tools?", tool_options)
    st.session_state.cooking_tools = cooking_tools

    if st.button("Generate Recipe"):
        with st.spinner('Generating recipe...'):
            start_time = timer()
            recipe_prompt, generated_recipe = generate_recipe_based_on_questions_with_RAG()
            end_time = timer()
            elapsed_time = end_time - start_time
            input_token_count = calculate_token_count(recipe_prompt)
            output_token_count = calculate_token_count(generated_recipe)
            st.write("### Recipe Prompt")
            st.text_area("Prompt", recipe_prompt, height=300)
            st.write("### Generated Recipe")
            formatted_recipe = format_recipe(generated_recipe)
            st.session_state.formatted_recipe = formatted_recipe
            st.text_area("Recipe", formatted_recipe, height=300)
            st.success(f'Recipe generated in {elapsed_time:.2f} seconds.')
            st.write(f"Input token count: {input_token_count}")
            st.write(f"Output token count: {output_token_count}")

    if 'formatted_recipe' in st.session_state:
        if st.button("Save Recipe"):
            saved_recipes = load_recipes()
            saved_recipes.append(st.session_state.formatted_recipe)
            save_recipes(saved_recipes)
            st.success("Recipe saved successfully!")
            st.session_state.page = "Main Board"
            st.rerun()


def create_menu():
    if st.button("Return to Main Page"):
        st.session_state.page = "Main Board"
        st.rerun()

    st.title("Create Menu")

    st.write("### Step 1: Menu Name")
    menu_name = st.text_input("Enter a name for your menu")
    st.session_state.menu_name = menu_name

    st.write("### Step 2: Number of Recipes")
    num_recipes = st.slider("How many recipes do you want to have in your menu? (2 to 6)", 2, 6, 2)
    st.session_state.num_recipes = num_recipes

    st.write("### Step 3: Number of People")
    number_of_people = st.slider("For how many people? (between 1 and 8)", 1, 8, 1)
    st.session_state.number_of_people = number_of_people

    st.write("### Step 4: Dietary Preferences and Restrictions")
    diet_options = ["none", "vegan", "vegetarian", "pescetarian", "no gluten", "no lactose", "no pork"]
    diets = []
    restrictions = []
    for i in range(number_of_people):
        st.write(f"#### Person {i + 1}")
        col1, col2 = st.columns(2)
        with col1:
            diet = st.selectbox(f"Diet", diet_options, key=f"menu_diet_{i}")
        with col2:
            restriction = st.text_input(f"Restrictions or dislikes (list ingredients, separated by commas)", key=f"menu_restriction_{i}")
        diets.append(diet)
        restrictions.append([r.strip().lower() for r in restriction.split(',')])
    st.session_state.diets = diets
    st.session_state.restrictions = restrictions

    st.write("### Step 5: Cooking Tools")
    tool_options = ["stovetop", "oven", "blender", "microwave", "automatic cooker", "fryer"]
    cooking_tools = st.multiselect("What are your cooking tools?", tool_options)
    st.session_state.cooking_tools = cooking_tools

    recipes = []
    for i in range(num_recipes):
        st.write(f"### Recipe {i + 1}")
        recipe_type = st.selectbox("What type of recipe is it?", ["Breakfast", "Lunch", "Dinner", "Dessert", "Appetizer", "Snacks"], key=f"recipe_type_{i}")
        ingredients = st.text_input(f"What ingredients will be in recipe {i + 1}? (list up to 5 ingredients, separated by commas)", key=f"recipe_ingredients_{i}").split(',')
        ingredients = [ing.strip().lower() for ing in ingredients]
        time_options = ["at most 15 min", "between 15-30 min", "30 min or more"]
        max_time = st.selectbox(f"In how much time would you like recipe {i + 1} to be made?", time_options, key=f"recipe_time_{i}")
        recipes.append((recipe_type, ingredients, max_time))
    st.session_state.recipes = recipes

    if st.button("Generate Menu"):
        with st.spinner('Generating menu...'):
            start_time = timer()
            menu_prompt, generated_menu = generate_menu_based_on_questions_with_RAG()
            end_time = timer()
            elapsed_time = end_time - start_time
            input_token_count = calculate_token_count(menu_prompt)
            output_token_count = calculate_token_count("".join(generated_menu))
            st.write("### Menu Prompt")
            st.text_area("Prompt", menu_prompt, height=300)
            st.write("### Generated Menu")
            st.success(f'Menu generated in {elapsed_time:.2f} seconds.')
            st.write(f'Input token count: {input_token_count}')
            st.write(f'Output token count: {output_token_count}')
            generated_menu = generated_menu.split("<recipe_end>")
            generated_menu = [m.strip() for m in generated_menu if m.strip() != ""]
            st.write(f"Number of recipes generated: {len(generated_menu)}")
            formatted_menu = [format_recipe(recipe) for recipe in generated_menu]
            st.session_state.formatted_menu = formatted_menu
            for i, recipe in enumerate(formatted_menu):
                st.write(f"### Recipe {i + 1}")
                st.text_area(f"Recipe {i + 1}", recipe, height=300)

    if 'formatted_menu' in st.session_state:
        if st.button("Save Menu"):
            saved_menus = load_menus()
            saved_menus.append({"name": st.session_state.menu_name, "recipes": st.session_state.formatted_menu})
            save_menus(saved_menus)
            st.success("Menu saved successfully!")
            st.session_state.page = "Main Board"
            st.rerun()


def generate_menu_based_on_questions_with_RAG():
    num_recipes = st.session_state.num_recipes
    number_of_people = st.session_state.number_of_people
    diets = st.session_state.diets
    restrictions = st.session_state.restrictions
    cooking_tools = st.session_state.cooking_tools
    recipes = st.session_state.recipes

    menu_context = f"""
    Number of Recipes: {num_recipes}
    Number of People: {number_of_people}
    Diets: {', '.join(diets)}
    Restrictions:
    """
    for i in range(number_of_people):
        menu_context += f"Person {i + 1}: {', '.join(restrictions[i])}\n"
    menu_context += f"Available Cooking Tools: {', '.join(cooking_tools)}\n\n"

    menu = []
    for i, (recipe_type, ingredients, max_time) in enumerate(recipes):
        compatible_ingredients = []
        for ingredient in ingredients:
            original_ingredient = ingredient
            for diet in diets:
                alternative = suggest_alternative(ingredient, diet)
                if alternative != ingredient:
                    print(f"For the {diet} diet, replacing {ingredient} with {alternative}.")
                ingredient = alternative
            compatible_ingredients.append(ingredient)

        recipe_context = f"""
        Recipe {i + 1}:
        Type: {recipe_type}
        Preferred Ingredients: {', '.join(compatible_ingredients)}
        Maximum Preparation Time: {max_time}
        """
        menu_context += recipe_context

    prompt = f"""You are an expert culinary chef. Create a cohesive and harmonious menu based on the following criteria:
    {menu_context}

    Instructions:
    1. You must respect the number of recipes specified. For example, if the number of recipes is 2, then only return 2 recipes, not more, not less.
    2. Each recipe must strictly adhere to the following guidelines:
        a. Respect all dietary restrictions and dislikes. Do not include any ingredients that are listed as restrictions. If a restriction such as "red fruits" is specified, ensure no red fruits are included.
        b. Use the preferred ingredients where possible, substituting alternatives if they conflict with any dietary restrictions or dislikes.
        c. The preparation time should not exceed the specified maximum time.
        d. The recipes should be suitable for the number of people specified.
        e. The recipes should only require the specified cooking tools.
        f. Ensure the dishes are logically sequenced and have a harmonious flow from one to the next.
        g. if the user provides you for example with "vegetables" as an ingredient, you can use any type of vegetables that based on your knowledge fits well with the other informations provided and your knowledge (preferably of season).
        h. If the user provides you for example fruits, and it is not clear which fruits, you can use any type of fruits that based on your knowledge fits well with the other informations provided and your knowledge (preferably of season).
        i. Specify the macros per serving per person (calories, fat, carbs, protein).

    For each recipe, provide the following details:
    - Title
    - Ingredients (including quantities)
    - Directions
    - Nutritional information (calories, fat, carbs, protein)
    - Preparation time
    - Category
    - Diet
    """

    RAG_context = retrieve_relevant_recipe(query=f"{', '.join(compatible_ingredients)}", embeddings=embeddings, pages_and_chunks=pages_and_chunks)
    try:
        menu_prompt = prompt + "\nInspire yourself from the these recipe to make a high protein and healthy recipe if possible:\n" + RAG_context

        menu = send_message_to_recipe_model(menu_prompt, model=model)
        print(f"Generated menu: {menu}")
    except Exception as e:
        print(f"Error generating recipe: {e}")
        return "An error occurred while generating the recipe."

    return menu_prompt, menu


def extract_title(recipe):
    if isinstance(recipe, dict):
        return recipe.get("TITLE", "")
    title_start = recipe.find("TITLE:") + len("TITLE: ")
    title_end = recipe.find("\n", title_start)
    return recipe[title_start:title_end]

def extract_ingredients(recipe):
    if isinstance(recipe, dict):
        return recipe.get("INGREDIENTS", "")
    ingredients_start = recipe.find("INGREDIENTS:") + len("INGREDIENTS: ")
    ingredients_end = recipe.find("DIRECTIONS:")
    return recipe[ingredients_start:ingredients_end].strip()

def extract_directions(recipe):
    if isinstance(recipe, dict):
        return recipe.get("DIRECTIONS", "")
    directions_start = recipe.find("DIRECTIONS:") + len("DIRECTIONS: ")
    directions_end = recipe.find("CALORIES:")
    return recipe[directions_start:directions_end].strip()

def extract_nutritional_info(recipe):
    if isinstance(recipe, dict):
        calories = recipe.get("CALORIES", "")
        fat = recipe.get("FAT", "")
        carbs = recipe.get("CARBS", "")
        protein = recipe.get("PROTEIN", "")
    else:
        calories_start = recipe.find("CALORIES:") + len("CALORIES: ")
        calories_end = recipe.find("FAT:")
        fat_start = recipe.find("FAT:") + len("FAT: ")
        fat_end = recipe.find("CARBS:")
        carbs_start = recipe.find("CARBS:") + len("CARBS: ")
        carbs_end = recipe.find("PROTEIN:")
        protein_start = recipe.find("PROTEIN:") + len("PROTEIN: ")
        protein_end = recipe.find("PREP TIME:")
        calories = recipe[calories_start:calories_end].strip()
        fat = recipe[fat_start:fat_end].strip()
        carbs = recipe[carbs_start:carbs_end].strip()
        protein = recipe[protein_start:protein_end].strip()

    return calories, fat, carbs, protein

def extract_prep_time(recipe):
    if isinstance(recipe, dict):
        return recipe.get("PREP TIME", "")
    prep_time_start = recipe.find("PREP TIME:") + len("PREP TIME: ")
    prep_time_end = recipe.find("TYPE:")
    return recipe[prep_time_start:prep_time_end].strip()

def extract_type(recipe):
    if isinstance(recipe, dict):
        return recipe.get("TYPE", "")
    type_start = recipe.find("TYPE:") + len("TYPE: ")
    type_end = recipe.find("DIET:")
    return recipe[type_start:type_end].strip()

def extract_diet(recipe):
    if isinstance(recipe, dict):
        return recipe.get("DIET", "")
    diet_start = recipe.find("DIET:") + len("DIET: ")
    diet_end = recipe.find("\n", diet_start)
    return recipe[diet_start:diet_end].strip()


# My Recipes Page
# My Recipes Page
def my_creations():
    if st.button("Return to Main Page"):
        st.session_state.page = "Main Board"
        st.rerun()
    st.title("My Recipes")

    search_query = st.text_input("Search for a recipe by title")
    saved_recipes = load_recipes()
    selected_recipes = []
    if not saved_recipes:
        st.write("No recipes saved yet.")
    else:
        filtered_recipes = [recipe for recipe in saved_recipes if search_query.lower() in extract_title(recipe).lower()] if search_query else saved_recipes
        for idx, recipe in enumerate(filtered_recipes):
            title = extract_title(recipe)
            ingredients = extract_ingredients(recipe)
            directions = extract_directions(recipe)
            calories, fat, carbs, protein = extract_nutritional_info(recipe)
            prep_time = extract_prep_time(recipe)
            type_ = extract_type(recipe)
            diet = extract_diet(recipe)

            st.write(f"### {title}")
            st.write(f"**Ingredients:**\n{ingredients}")
            st.write(f"**Directions:**\n{directions}")
            st.write(f"**Nutritional Information:**")
            st.write(f"- Calories: {calories}")
            st.write(f"- Fat: {fat}")
            st.write(f"- Carbs: {carbs}")
            st.write(f"- Protein: {protein}")
            st.write(f"**Prep Time:** {prep_time}")
            st.write(f"**Type:** {type_}")
            st.write(f"**Diet:** {diet}")

            if st.checkbox(f"Select {title} for Grocery List", key=f"select_{idx}"):
                selected_recipes.append(recipe)

            if st.button(f"Delete Recipe {idx + 1}", key=f"delete_{idx}"):
                del saved_recipes[idx]
                save_recipes(saved_recipes)
                st.session_state.page = "My Recipes"
                st.rerun()

    if selected_recipes:
        if st.button("Generate Grocery List"):
            grocery_list = generate_grocery_list(selected_recipes)
            st.session_state.grocery_list = grocery_list
            st.session_state.page = "Grocery List"
            st.rerun()



# My Menus Page
# My Menus Page
def my_menus():
    if st.button("Return to Main Page"):
        st.session_state.page = "Main Board"
        st.rerun()
    st.title("My Menus")

    search_query = st.text_input("Search for a menu by name")
    saved_menus = load_menus()
    if not saved_menus:
        st.write("No menus saved yet.")
    else:
        filtered_menus = [menu for menu in saved_menus if search_query.lower() in menu['name'].lower()] if search_query else saved_menus
        for idx, menu in enumerate(filtered_menus):
            st.write(f"### Menu {idx + 1}: {menu['name']}")
            for j, recipe in enumerate(menu['recipes']):
                title = extract_title(recipe)
                ingredients = extract_ingredients(recipe)
                directions = extract_directions(recipe)
                calories, fat, carbs, protein = extract_nutritional_info(recipe)
                prep_time = extract_prep_time(recipe)
                type_ = extract_type(recipe)
                diet = extract_diet(recipe)

                st.write(f"#### Recipe {j + 1}: {title}")
                st.write(f"**Ingredients:**\n{ingredients}")
                st.write(f"**Directions:**\n{directions}")
                st.write(f"**Nutritional Information:**")
                st.write(f"- Calories: {calories}")
                st.write(f"- Fat: {fat}")
                st.write(f"- Carbs: {carbs}")
                st.write(f"- Protein: {protein}")
                st.write(f"**Prep Time:** {prep_time}")
                st.write(f"**Type:** {type_}")
                st.write(f"**Diet:** {diet}")

            if st.button(f"Generate Grocery List for Menu {idx + 1}", key=f"menu_select_{idx}"):
                grocery_list = generate_grocery_list(menu['recipes'])
                st.session_state.grocery_list = grocery_list
                st.session_state.page = "Grocery List"
                st.rerun()

            if st.button(f"Delete Menu {idx + 1}", key=f"delete_menu_{idx}"):
                del saved_menus[idx]
                save_menus(saved_menus)
                st.session_state.page = "My Menus"
                st.rerun()


def generate_grocery_list(selected_recipes):
    grocery_list = {}
    for recipe in selected_recipes:
        ingredients = extract_ingredients(recipe).split('\n-')
        for ingredient in ingredients:
            ingredient = ingredient.strip()
            if ingredient:
                if ingredient in grocery_list:
                    grocery_list[ingredient] += 1
                else:
                    grocery_list[ingredient] = 1
    return grocery_list


# Create Weekly Plan Page
def create_weekly_plan():
    if st.button("Return to Main Page", key="return_main_page"):
        st.session_state.page = "Main Board"
        st.rerun()

    st.title("Create Weekly Meal Plan")

    days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    daily_plans = {}

    for day in days_of_week:
        st.write(f"### {day}")
        daily_plans[day] = daily_plans.get(day, [])

        add_more = True
        plan_index = 0
        while add_more:
            plan_type = st.selectbox(
                f"Select an option for {day}", 
                ["None", "Add Existing Menu", "Add Existing Recipe"], 
                key=f"{day}_plan_type_{plan_index}"
            )

            if plan_type == "Add Existing Menu":
                existing_menus = load_menus()
                if existing_menus:
                    menu_selection = st.selectbox(
                        f"Select a menu for {day}", 
                        [f"Menu {i+1}" for i in range(len(existing_menus))], 
                        key=f"{day}_menu_selection_{plan_index}"
                    )
                    selected_menu = existing_menus[int(menu_selection.split()[-1]) - 1]
                    daily_plans[day].append(selected_menu)
                else:
                    st.write("No existing menus available.")

            elif plan_type == "Add Existing Recipe":
                existing_recipes = load_recipes()
                if existing_recipes:
                    recipe_titles = [extract_title(recipe) for recipe in existing_recipes]
                    recipe_selection = st.selectbox(
                        f"Select a recipe for {day}", 
                        recipe_titles, 
                        key=f"{day}_recipe_selection_{plan_index}"
                    )
                    selected_recipe = existing_recipes[recipe_titles.index(recipe_selection)]
                    daily_plans[day].append(selected_recipe)
                else:
                    st.write("No existing recipes available.")

            plan_index += 1
            add_more = st.checkbox(f"Add another plan for {day}", key=f"{day}_add_more_{plan_index}")

    plan_name = st.text_input("Name your weekly plan", key="plan_name_input")
    save_plan_button = st.button("Save Weekly Plan", key="save_plan_button")

    if save_plan_button and plan_name:
        weekly_plan = {
            "plan_name": plan_name,
            "days": daily_plans
        }
        saved_plans = load_plans()
        saved_plans.append(weekly_plan)
        save_plans(saved_plans)
        st.success("Weekly plan saved successfully!")
        st.session_state.page = "Main Board"
        st.rerun()
    elif save_plan_button:
        st.error("Please provide a name for the weekly plan before saving.")


def display_recipe(recipe):
    title = extract_title(recipe)
    ingredients = extract_ingredients(recipe)
    directions = extract_directions(recipe)
    calories, fat, carbs, protein = extract_nutritional_info(recipe)
    prep_time = extract_prep_time(recipe)
    type_ = extract_type(recipe)
    diet = extract_diet(recipe)

    st.write(f"**Title:** {title}")
    st.write(f"**Ingredients:**\n{ingredients}")
    st.write(f"**Directions:**\n{directions}")
    st.write(f"**Nutritional Information:**")
    st.write(f"- Calories: {calories}")
    st.write(f"- Fat: {fat}")
    st.write(f"- Carbs: {carbs}")
    st.write(f"- Protein: {protein}")
    st.write(f"**Prep Time:** {prep_time}")
    st.write(f"**Type:** {type_}")
    st.write(f"**Diet:** {diet}")


# My Plans Page
def my_plans():
    if st.button("Return to Main Page", key="return_main_page"):
        st.session_state.page = "Main Board"
        st.rerun()
    st.title("My Plans")
    saved_plans = load_plans()
    if not saved_plans:
        st.write("No plans saved yet.")
    else:
        for idx, plan in enumerate(saved_plans):
            st.write(f"### Plan {idx + 1}: {plan['plan_name']}")
            for day, items in plan["days"].items():
                st.write(f"#### {day}")
                for item in items:
                    if isinstance(item, list):
                        for recipe in item:
                            display_recipe(recipe)
                    else:
                        display_recipe(item)

            if st.button(f"Generate Grocery List for Plan {idx + 1}", key=f"plan_select_{idx}"):
                all_recipes = [recipe for day_items in plan["days"].values() for item in day_items for recipe in (item if isinstance(item, list) else [item])]
                grocery_list = generate_grocery_list(all_recipes)
                st.session_state.grocery_list = grocery_list
                st.session_state.page = "Grocery List"
                st.rerun()

            if st.button(f"Delete Plan {idx + 1}", key=f"delete_plan_{idx}"):
                del saved_plans[idx]
                save_plans(saved_plans)
                st.session_state.page = "My Plans"
                st.rerun()


# Grocery List Page
def grocery_list():
    if st.button("Return to Main Page"):
        st.session_state.page = "Main Board"
        st.rerun()
    st.title("Grocery List")
    grocery_list = st.session_state.grocery_list
    for ingredient, quantity in grocery_list.items():
        st.write(f"{ingredient}: {quantity}")


# Main Function
def main():
    if 'page' not in st.session_state:
        st.session_state.page = "Main Board"
    page = st.session_state.page

    if page == "Main Board":
        main_board()
    elif page == "Create Recipe":
        create_recipe()
    elif page == "Create Menu":
        create_menu()
    elif page == "Create Weekly Plan":
        create_weekly_plan()
    elif page == "My Recipes":
        my_creations()
    elif page == "My Menus":
        my_menus()
    elif page == "My Plans":
        my_plans()
    elif page == "Grocery List":
        grocery_list()


if __name__ == "__main__":
    main()