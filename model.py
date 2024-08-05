import torch
# Define helper function to print wrapped text 
import textwrap
import numpy as np 
import pandas as pd
import fitz
import matplotlib.pyplot as plt
from time import perf_counter as timer
from sentence_transformers import util
#libraries
from CleanRecipeMethods import *
from UnitTestRecipe import *
import warnings 
warnings.filterwarnings("ignore")

pdf_path = "greg_doucette_cookbook_2_0.pdf"
recipe_dataset = pd.read_csv("recipe_dataset.csv")

def print_wrapped(text, wrap_length=80):
    wrapped_text = textwrap.fill(text, wrap_length)
    print(wrapped_text)

def display_relevant_pages(pdf_path, most_relevant_pages, query):
    # Open the PDF
    doc = fitz.open(pdf_path)

    # Iterate through the relevant pages
    for page_number in most_relevant_pages:
        page = doc.load_page(page_number-1) # Load the page
        
        # Get the image of the page
        img = page.get_pixmap(dpi=300)

        # Convert the Pixmap to a numpy array
        img_array = np.frombuffer(img.samples, dtype=np.uint8).reshape((img.height, img.width, img.n))

        # Display the image using Matplotlib
        plt.figure(figsize=(13, 10))
        plt.imshow(img_array)
        plt.title(f"Query: '{query}' | Most relevant page: {page_number}")
        plt.axis('off') # Turn off axis
        plt.show()

    # Close the PDF document
    doc.close()




#we load the embedding model from our local directory
from sentence_transformers import SentenceTransformer
embedding_model = torch.load("embedding_model.pt")

device = "cuda" if torch.cuda.is_available() else "cpu"

# Import texts and embedding df
text_chunks_and_embedding_df = pd.read_csv("text_chunks_and_embeddings_df.csv")

# Convert embedding column back to np.array (it got converted to string when it got saved to CSV)
text_chunks_and_embedding_df["embedding"] = text_chunks_and_embedding_df["embedding"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))

# Convert texts and embedding df to list of dicts
pages_and_chunks = text_chunks_and_embedding_df.to_dict(orient="records")

# Convert embeddings to torch tensor and send to device (note: NumPy arrays are float64, torch tensors are float32 by default)
embeddings = torch.tensor(np.array(text_chunks_and_embedding_df["embedding"].tolist()), dtype=torch.float32).to(device)


# 1. Define the query
# Note: This could be anything. But since we're working with a nutrition textbook, we'll stick with nutrition-based queries.
# 3. Get similarity scores with the dot product (we'll time this for fun)


def retrieve_relevant_resources(query: str,
                                embeddings: torch.tensor,
                                model: SentenceTransformer=embedding_model,
                                n_resources_to_return: int=5,
                                print_time: bool=True):
    """
    Embeds a query with model and returns top k scores and indices from embeddings.
    """

    # Embed the query
    query_embedding = model.encode(query, 
                                   convert_to_tensor=True) 

    # Get dot product scores on embeddings
    start_time = timer()
    dot_scores = util.dot_score(query_embedding, embeddings)[0]
    end_time = timer()

    if print_time:
        print(f"[INFO] Time taken to get scores on {len(embeddings)} embeddings: {end_time-start_time:.5f} seconds.")

    scores, indices = torch.topk(input=dot_scores, 
                                 k=n_resources_to_return)

    return scores, indices

def print_top_results_and_scores(query: str,
                                 embeddings: torch.tensor,
                                 pages_and_chunks: list[dict]=pages_and_chunks,
                                 n_resources_to_return: int=5):
    """
    Takes a query, retrieves most relevant resources and prints them out in descending order.

    Note: Requires pages_and_chunks to be formatted in a specific way (see above for reference).
    """
    
    scores, indices = retrieve_relevant_resources(query=query,
                                                  embeddings=embeddings,
                                                  n_resources_to_return=n_resources_to_return)
    
    print(f"Query: {query}\n")
    print("Results:")
    # Loop through zipped together scores and indicies
    for score, index in zip(scores, indices):
        print(f"Score: {score:.4f}")
        # Print relevant sentence chunk (since the scores are in descending order, the most relevant chunk will be first)
        print_wrapped(pages_and_chunks[index]["sentence_chunk"])
        # Print the page number too so we can reference the textbook further and check the results
        print(f"Page number: {pages_and_chunks[index]['page_number']}")
        print("\n")


def print_top_recipes(query: str,
                    embeddings: torch.tensor,
                    pages_and_chunks: list[dict]=pages_and_chunks,
                    n_resources_to_return: int=5):
    """
    Takes a query, retrieves most relevant resources and prints them out in descending order.

    Note: Requires pages_and_chunks to be formatted in a specific way (see above for reference).
    """
    
    scores, indices = retrieve_relevant_resources(query=query,
                                                  embeddings=embeddings,
                                                  n_resources_to_return=n_resources_to_return)
    print(f"Query: {query}\n")
    print("Results:")
    # Loop through zipped together scores and indicies
    for score, index in zip(scores, indices):
        print(f"Score: {score:.4f}")
        # Print the page number too so we can reference the textbook further and check the results
        print(f"Page number: {pages_and_chunks[index]['page_number']}")
        # Print the recipe
        print_wrapped(recipe_dataset.loc[recipe_dataset['page_number'] == pages_and_chunks[index]['page_number'], 'text'].values[0])
        print("\n")


#print_top_recipes("bread", embeddings, pages_and_chunks, 5)

### Setup prompt formatter 
def prompt_formatter(query: str, 
                     context_items: str) -> str:
    """
    Augments query with text-based context from context_items.
    """
    # Join context items into one dotted paragraph
    #context = "- " + "\n- ".join([item["sentence_chunk"] for item in context_items])
    context=context_items
    # Create a base prompt with examples to help the model
    # Note: this is very customizable, I've chosen to use 3 examples of the answer style we'd like.
    # We could also write this in a txt file and import it in if we wanted.
    base_prompt = """
    Inspire yourself on the following prompt, return a recipe that answers the query. Give yourself room to think. Don't return the thinking, only return the answer.

    You must include every detail in the following format:
    - Title
    - Macros (calories, fat, carbs, fiber, protein)
    - Directions
    - Ingredients
    - Prep time
    - Ready in time
    - Diet (vegan, vegetarian, etc.)
    - Category

    Make sure your answers are as explanatory as possible.
    Use the following examples as reference for the ideal answer style.

    \nExample 1:
    Query: Can you generate me a high protein breakfast recipe?
    Answer: 
    TITLE: Anabolic Apple Pie Breakfast Bake - Entire Batch
    CALORIES (in kcal): 3350
    FAT (in g): 17
    CARBS (in g): 464
    FIBER (in g): 46
    PROTEIN (in g): 265


    DIRECTIONS:
    1. Pre-heat the oven to 400°F (204°C).
    2. Chop the apples into small pieces.
    3. In a bowl, whisk egg whites, cinnamon, sweetener, and vanilla.
    4. Tear the bread into small pieces and place in a bowl with the egg whites, cinnamon, sweetener, and vanilla. Mix with your hands until the bread pieces are well soaked with the batter.
    5. Spray a casserole dish with cooking spray for 1 second. Pour the egg white/bread mixture into the casserole dish.
    6. Place the casserole dish uncovered in the middle rack and cook in the oven at 400°F/204°C for 40-50 minutes.

    INGREDIENTS:
    18 slices regular bread (or one loaf [570g] of regular bread)
    1920g (4 cartons/2000ml) egg whites
    21g (3 tbsp) cinnamon
    15g (1 tbsp) vanilla extract
    15 packets (⅝ cup) sweetener
    1500g or ~10 apples of your choice
    Cooking spray

    PREP TIME (in m): 20

    READY IN (in m): 60

    CATEGORY: Breakfast

    DIET: non-vegan , vegetarian

    \nExample 2:
    Query: Make me a snack, I like apples?
    Answer: 
    TITLE: Apple Cinnamon Protein Rice Cakes
    CALORIES (in kcal): 580
    FAT (in g): 7
    CARBS (in g): 85
    FIBER (in g): 10
    PROTEIN (in g): 45

    DIRECTIONS:
    1. Mix the chocolate protein powder and powdered peanut butter in a bowl. Slowly add water to make a liquid paste consistency. Add sweetener if you desire more sweetness.
    2. Spread the liquid paste over the rice cakes.
    3. Wash the apple and cut into thin slices, place on top of the rice cake.
    4. Sprinkle with cinnamon. Enjoy!

    INGREDIENTS:
    2 rice cakes
    30g chocolate protein powder
    20g powdered peanut butter
    Water as needed
    1 apple
    Cinnamon to taste

    PREP TIME (in m): 10

    READY IN (in m): 15

    CATEGORY: Treats

    DIET: non-vegan , vegetarian

    \nExample 3:
    Query: I want pizza for dinner, I have cauliflower?
    Answer: 
    TITLE: Cauliflower Pizza Crust - Total
    CALORIES (in kcal): 800
    FAT (in g): 5
    CARBS (in g): 128
    FIBER (in g): 21
    PROTEIN (in g): 56

    TITLE: Cauliflower Pizza Crust - Per Crust
    CALORIES (in kcal): 200
    FAT (in g): 1
    CARBS (in g): 32
    FIBER (in g): 5
    PROTEIN (in g): 14

    DIRECTIONS:
    1. OPTIONAL: Prep cauliflower rice (either see the recipe in this book on page 124 or purchase pre-cooked cauliflower rice.)
    2. Pre-heat the oven to 400°F/204°C.
    3. In a bowl mix flour, guar/xanthan gum, garlic powder, salt, oregano, and basil.
    4. Add in the Greek yogurt and fold together to form a ball.
    5. In another bowl combine cooked cauliflower rice and egg whites. Mix well.
    6. Add the cauliflower mixture to the flour mixture and mix well. You can use your hands or a hand blender.
    7. Let stand at room temperature for 20 minutes.
    8. Divide the mixture into six 150g portions.
    9. Cover a baking sheet with parchment paper and spread the mixture into a ‘circle’.
    10. Bake at 400°F/204°C for 30-35 minutes or until lightly browned.
    11. Remove from the oven and let cool for a few minutes.

    INGREDIENTS:
    100g (~7/8 cup) self-raising flour
    700g (3 cups) cooked cauliflower rice
    180g (¾ cup) egg whites
    250g (1 cup) 0% fat Greek yogurt
    9g (1 tbsp) guar/xanthan gum
    1 tsp garlic powder
    ¼ tsp Kosher salt
    ½ tsp oregano
    ½ tsp basil

    PREP TIME (in m): 10

    READY IN (in m): 45

    CATEGORY: Dinner

    DIET: non-vegan , vegetarian

    Now, Create a recipe inspired by the following context to answer the user query:
    {context}
    User query: {query}
    Answer:
    """


    # Update base prompt with context items and query   
    prompt = base_prompt.format(context=context, query=query)
    return prompt

def retrieve_relevant_recipe(query: str,
                    embeddings: torch.tensor,
                    pages_and_chunks: list[dict]=pages_and_chunks,
                    n_resources_to_return: int=5):
    scores, indices = retrieve_relevant_resources(query=query,
                                                  embeddings=embeddings,
                                                  n_resources_to_return=n_resources_to_return)
    
    recipes = []
    print(f"Query: {query}\n")
    print("Results:")
    # Loop through zipped together scores and indicies
    for score, index in zip(scores, indices):
        # Print the recipe
        recipes.append(recipe_dataset.loc[recipe_dataset['page_number'] == pages_and_chunks[index]['page_number'], 'text'].values[0])
    
    context= "\n-------------------------------------\n".join(recipes)
    
    return context



def generate_recipe_with_RAG(query, embeddings, pages_and_chunks, model='ft:gpt-3.5-turbo-1106:personal:recipecreatorv7b:9XU88CKN'):
    """
    Generates a recipe using a Retrieval Augmented Generation (RAG) approach. It retrieves relevant context based on the query
    and generates a recipe using a language model.

    Args:
        query (str): The user's query for the recipe (e.g., "high protein breakfast").
        embeddings (torch.tensor): The embeddings of the text data.
        pages_and_chunks (list[dict]): The list of dictionaries containing the text chunks and associated metadata.
        model (str): The identifier of the language model to use.

    Returns:
        str: The generated recipe.
    """
    # Step 1: Retrieve relevant context based on the query
    context = retrieve_relevant_recipe(query=query, embeddings=embeddings, pages_and_chunks=pages_and_chunks)

    # Step 2: Format the prompt using the retrieved context
    prompt = prompt_formatter(query=query, context_items=context)

    # Step 3: Generate the recipe using the language model
    recipe = send_message_to_recipe_model(prompt, model=model)

    # Step 4: Return the generated recipe
    return recipe


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

def validate_number_of_people():
    """Validates and returns the number of people."""
    while True:
        try:
            number_of_people = int(input("For how many people? (between 1 and 8): "))
            if 1 <= number_of_people <= 8:
                return number_of_people
            else:
                print("Number of people must be between 1 and 8.")
        except ValueError:
            print("Invalid input. Please enter a number between 1 and 8.")

def get_user_input():
    """Gathers user input for recipe generation."""
    # Provide options for dish types
    dish_type = input("What kind of dish do you want to make? (Options: Breakfast, Lunch, Dinner, Dessert, Appetizer, Snacks): ").strip().capitalize()
    
    # Validate number of people
    number_of_people = validate_number_of_people()

    # Provide dietary options
    diet_options = ["vegan", "vegetarian", "vegetalian", "pescetarian", "no gluten", "no lactose", "no porc", "none"]
    diets = []
    for i in range(number_of_people):
        while True:
            diet = input(f"What is the diet for person {i+1}? (Options: {', '.join(diet_options)}): ").strip().lower()
            if diet in diet_options:
                diets.append(diet)
                break
            else:
                print("Invalid diet option. Please choose from the provided options.")
    
    restrictions = []
    for i in range(number_of_people):
        restriction = input(f"What are the restrictions or dislikes for person {i+1}? (list ingredients, separated by commas): ")
        restrictions.append([r.strip().lower() for r in restriction.split(',')])
    
    ingredients = [ing.strip().lower() for ing in input("What ingredients would you like the recipe to have? (list up to 5 ingredients, separated by commas): ").split(',')]
    
    # Provide preparation time options
    time_options = ["at most 15 min", "between 15-30 min", "30 min or more"]
    max_time = ""
    while max_time not in time_options:
        max_time = input(f"How much time would you like the recipe to take to prepare? (Options: {', '.join(time_options)}): ").strip().lower()
        if max_time not in time_options:
            print("Invalid time option. Please choose from the provided options.")
    
    # Provide cooking tool options
    tool_options = ["stovetop", "oven", "blender", "microwave", "automatic cooker", "fryer", "none"]
    cooking_tools = []
    while not cooking_tools:
        cooking_tools_input = input(f"What are your cooking tools? (Options: {', '.join(tool_options)}, separated by commas): ").split(',')
        cooking_tools = [tool.strip().lower() for tool in cooking_tools_input if tool.strip().lower() in tool_options]
        if not cooking_tools:
            print("Invalid tool options. Please choose from the provided options.")

    return dish_type, number_of_people, diets, restrictions, ingredients, max_time, cooking_tools

def generate_recipe_based_on_questions_with_RAG(model='gpt-4o-mini'):
    # Gather user input
    user_input = get_user_input()
    if not user_input:
        return "Invalid input received. Cannot generate recipe."

    dish_type, number_of_people, diets, restrictions, ingredients, max_time, cooking_tools = user_input

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
                print(f"For the {diet} diet, replacing {ingredient} with {alternative}.")
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
        recipe = generate_recipe_with_RAG(prompt, embeddings, pages_and_chunks, model=model)
    except Exception as e:
        print(f"Error generating recipe: {e}")
        return "An error occurred while generating the recipe."

    return recipe

# Example usage
recipe = generate_recipe_based_on_questions_with_RAG()
print(print_recipe(recipe))
