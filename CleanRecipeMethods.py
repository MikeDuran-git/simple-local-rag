import difflib
import re
from colorama import Fore, Style
from openai import OpenAI # type: ignore
import json
import unittest
import re
class TestRecipeGeneration(unittest.TestCase):
    """
    This class contains the test methods for the recipe generation. It will receive a recipe as an input and will test the 
    structure of the recipe, the title, the ingredients, the directions, the calories, the fat content, the carbohydrate content, the protein content, the preparation time, the type, and the diet.
    """
    def __init__(self, methodName='runTest', recipe_output=None):
        super(TestRecipeGeneration, self).__init__(methodName)
        self.recipe_output = recipe_output

    def setUp(self):
        self.assertIsNotNone(self.recipe_output, "Recipe output should not be None")

    def run_recipe_test(recipe):
        suite = unittest.TestSuite()
        for method in dir(TestRecipeGeneration):
            if method.startswith("test_"):
                suite.addTest(TestRecipeGeneration(methodName=method, recipe_output=recipe))
        runner = unittest.TextTestRunner()
        runner.run(suite)

    def test_structure(self):
        required_sections = [
            '<recipe_start>', '<title_start>', '<title_end>',
            '<ingredient_start>', '<ingredient_next>', '<ingredient_end>',
            '<directions_start>', '<directions_end>',
            '<calories_start>', '<calories_end>',
            '<fatcontent_start>', '<fatcontent_end>',
            '<carbohydratecontent_start>', '<carbohydratecontent_end>',
            '<proteincontent_start>', '<proteincontent_end>',
            '<prep_time_min_start>', '<prep_time_min_end>',
            '<type_start>', '<type_end>',
            '<diet_start>', '<diet_end>',
            '<recipe_end>'
        ]

        for section in required_sections:
            with self.subTest(section=section):
                self.assertIn(section, self.recipe_output, f"Missing section: {section}")

        # Check for <directions_next> only if there are multiple directions
        directions_pattern = re.compile(r'<directions_start>(.*?)<directions_end>', re.DOTALL)
        match = directions_pattern.search(self.recipe_output)
        if match:
            directions_content = match.group(1).strip()
            if '<directions_next>' in directions_content:
                self.assertIn('<directions_next>', self.recipe_output, "Missing section: <directions_next>")

    def test_title(self):
        pattern = re.compile(r'<title_start>(.*?)<title_end>', re.DOTALL)
        match = pattern.search(self.recipe_output)
        self.assertIsNotNone(match, "Title section is incorrectly formatted")
        
        title_content = match.group(1).strip()
        self.assertNotEqual(title_content, "", "Title should not be empty")
        
    def test_ingredients(self):
        pattern = re.compile(r'<ingredient_start>(.*?)<ingredient_end>', re.DOTALL)
        match = pattern.search(self.recipe_output)
        self.assertIsNotNone(match, "Ingredients section is incorrectly formatted")

        ingredients_section = match.group(1).strip()
        self.assertNotEqual(ingredients_section, "", "Ingredients section should not be empty")

        # Split ingredients by <ingredient_next>
        ingredients = ingredients_section.split('<ingredient_next>')

        # Check if there are no leading or trailing <ingredient_next> tokens
        self.assertFalse(ingredients_section.startswith('<ingredient_next>'), "Ingredients section should not start with <ingredient_next>")
        self.assertFalse(ingredients_section.endswith('<ingredient_next>'), "Ingredients section should not end with <ingredient_next>")

        # Check for correct number of <ingredient_next> tokens
        num_ingredients = len(ingredients)
        num_next_tokens = ingredients_section.count('<ingredient_next>')
        self.assertEqual(num_next_tokens, num_ingredients - 1, "Incorrect number of <ingredient_next> tokens")

        # Updated pattern to be more flexible
        ingredient_pattern = re.compile(r'^\d*\.?\d*\s?\w+.*$')

        for ingredient in ingredients:
            ingredient = ingredient.strip()
            # Ensure ingredient is not just numbers
            self.assertFalse(ingredient.isdigit(), f"Ingredient should not be only numbers: {ingredient}")
            # Ensure ingredient follows a consistent format
            self.assertTrue(ingredient_pattern.match(ingredient), f"Ingredient format is incorrect: {ingredient}")


with open('./openaiCredentials.json') as f:
    data = json.load(f)
    api_key = data['OPENAI_API_KEY']
    organization = data['ORGANIZATION_ID']

client = OpenAI(
    api_key=api_key,
    organization=organization
)

def send_message_to_recipe_model(msg, 
                                 content=(
                    "You are a helpful assistant that generates a new recipe based on a given list of ingredients. "
                    "Follow these guidelines strictly: "
                    "1. Ensure the generated recipes include only the specified elements in the following order and format, all in a single line: "
                    "'<recipe_start> <title_start>title_name<title_end> <ingredient_start>ingredient1<ingredient_next>ingredient2<ingredient_next>...<ingredient_end> "
                    "<directions_start>direction1<directions_next>direction2<directions_next>...<directions_end> <calories_start>calories_value<calories_end> "
                    "<fatcontent_start>fat_content_value<fatcontent_end> <carbohydratecontent_start>carbohydrate_content_value<carbohydratecontent_end> "
                    "<proteincontent_start>protein_content_value<proteincontent_end> <prep_time_min_start>prep_time_value<prep_time_min_end> "
                    "<type_start>type_value<type_end> <diet_start>diet_value<diet_end> <recipe_end>'. 2. The user will only provide a list of ingredients. Based on your training, "
                    "you must define the dosage of each ingredient. 3. Correct diet misclassification issues by ensuring the diet matches one of the following categories: vegan, vegetarian, contains_meat. "
                    "4. Maintain proper formatting in the output, including consistent spacing. 5. Accurately classify ingredients into meal types and diets based on enhanced training datasets. "
                    "6. Verify caloric and macronutrient calculations using the formula: Calories = 9 * fatContent + 4 * carbohydrateContent + 4 * proteinContent. "
                    "7. Ensure that the correct units are used for each ingredient. Solids should be measured in grams (g) and liquids in milliliters (ml). "
                    "8. Strictly enforce the expected output format by using the specified tokens: <title_start><title_end>, <ingredient_start><ingredient_end>, <directions_start><directions_end>, etc. "
                    "9. Ensure that the units are consistent and appropriate for the type of ingredient. 10. Normalize text to ensure consistent capitalization and remove unnecessary punctuation or characters. "
                    "11. Ensure no additional tokens or unexpected characters are included in the output. "
                    "12. Correct type misclassification issues by ensuring the type matches one of the following categories: appetizer, dinner, lunch, breakfast_and_brunch, desert "
                    "13. If a vegan recipe is requested and you see that there is an ingredient that is not vegan, you can specify the vegan version of it. For example, if the ingredient is Honey, you can specify (vegan) Honey, or (vegan) chicken."
                    "14. if you see 14 bananas, then that must mean 1/4 bananas"
                    "Example Input: ingredients: [mayonnaise, package knorr leek mix, sour cream, package spinach, chopped, drained well, loaf] "
                    "Expected Output: <recipe_start> <title_start> Bread Bowl Spinach Dip <title_end> <ingredient_start> 118.29 ml mayonnaise <ingredient_next> 26.57 g package knorr leek mix <ingredient_next> 118.29 ml sour cream "
                    "<ingredient_next> 141.74 g package spinach, chopped, drained well <ingredient_next> round sourdough loaf <ingredient_end> <directions_start> Mix first four ingredients well and refrigerate for 6 hours. "
                    "<directions_next> Create cavity inside French or sourdough bread loaf. <directions_next> Reserve pieces for dip. <directions_next> Fill the cavity with spinach dip. <directions_next> Makes 2 cups. "
                    "<directions_end> <calories_start> 84.9 <calories_end> <fatcontent_start> 7.8 <fatcontent_end> <carbohydratecontent_start> 2.8 <carbohydratecontent_end> <proteincontent_start> 2.1 <proteincontent_end> "
                    "<prep_time_min_start> 365 <prep_time_min_end> <type_start> appetizer <type_end> <diet_start> vegetarian <diet_end> <recipe_end>'"
                ) ,
                model='ft:gpt-3.5-turbo-1106:personal:recipecreatorv7b:9XU88CKN'):
    """
    This method sends a message to the recipe model and returns the response message.
    The form of the message is as follows: 
    ingredients: [ingredient1, ingredient2, ingredient3, ...]
    The model used by default is the RecipeCreatorV7b model, which is the model with the best performance.
    """
    response = client.chat.completions.create(
        model=model,
        response_format={"type": "text"},
        messages=[
            {
                "role": "system",
                "content": content
            },
            {
                "role": "user",
                "content": msg
            },
        ],
    )
    return response.choices[0].message.content

def print_recipe(recipe):
    """
    This method formats and prints the recipe in a more readable way with added colors.
    """
    recipe = recipe.replace("<recipe_start>", "\n").replace("<title_start>", "TITLE: ").replace("<ingredient_start>", "INGREDIENTS: \n-").replace("<ingredient_next>", "\n-").replace("<directions_start>", "DIRECTIONS: \n-").replace("<directions_next>", "\n-").replace("<calories_start>", "CALORIES: ").replace("<fatcontent_start>", "FAT: ").replace("<carbohydratecontent_start>", "CARBS: ").replace("<proteincontent_start>", "PROTEIN: ").replace("<prep_time_min_start>", "PREP TIME: ").replace("<type_start>", "TYPE: ").replace("<diet_start>", "DIET: ").replace("<title_end>", "\n").replace("<ingredient_end>", "\n").replace("<directions_end>", "\n").replace("<calories_end>", "\n").replace("<fatcontent_end>", "\n").replace("<carbohydratecontent_end>", "\n").replace("<proteincontent_end>", "\n").replace("<prep_time_min_end>", "\n").replace("<type_end>", "\n").replace("<diet_end>", "\n").replace("<recipe_end>", "\n")

    # Adding color to specific parts
    recipe = recipe.replace("TITLE: ", "\033[1;32;40m TITLE: \033[0m").replace("INGREDIENTS: \n-", "\033[1;34;40m INGREDIENTS: \033[0m\n-").replace("DIRECTIONS: \n-", "\033[1;33;40m DIRECTIONS: \033[0m\n-").replace("CALORIES: ", "\033[1;31;40m CALORIES: \033[0m").replace("FAT: ", "\033[1;31;40m FAT: \033[0m").replace("CARBS: ", "\033[1;31;40m CARBS: \033[0m").replace("PROTEIN: ", "\033[1;31;40m PROTEIN: \033[0m").replace("PREP TIME: ", "\033[1;36;40m PREP TIME: \033[0m").replace("TYPE: ", "\033[1;36;40m TYPE: \033[0m").replace("DIET: ", "\033[1;36;40m DIET: \033[0m")
    
    return recipe

def colored_recipe_printer(recipe_dict):
    #we print the recipe
    return  Fore.GREEN + "TITLE: " + Style.RESET_ALL + recipe_dict['TITLE'] + '\n'+ \
            Fore.BLUE + "INGREDIENTS:" + Style.RESET_ALL + '\n' + recipe_dict['INGREDIENTS'] + '\n'+ \
            Fore.YELLOW + "DIRECTIONS:" + Style.RESET_ALL + '\n' + recipe_dict['DIRECTIONS'] + '\n'+\
            Fore.RED + "CALORIES: " + Style.RESET_ALL + recipe_dict['CALORIES'] + '\n'+\
            Fore.RED + "FAT CONTENT: " + Style.RESET_ALL + recipe_dict['FAT CONTENT'] + '\n'+\
            Fore.RED + "CARBOHYDRATES CONTENT: " + Style.RESET_ALL + recipe_dict['CARBOHYDRATES CONTENT'] + '\n'+\
            Fore.RED + "PROTEIN CONTENT: " + Style.RESET_ALL + recipe_dict['PROTEIN CONTENT'] + '\n'+\
            Fore.RED + "PREP TIME: " + Style.RESET_ALL + recipe_dict['PREP TIME'] + '\n'+\
            Fore.CYAN + "TYPE: " + Style.RESET_ALL + recipe_dict['TYPE'] + '\n'+\
            Fore.CYAN + "DIET: " + Style.RESET_ALL + recipe_dict['DIET']

def clean_main_structure(recipe):
    """
    This method cleans the main structure of the recipe by ensuring that the tokens are in the correct order and format.
    It also removes any extra spaces and newlines.
    """
    tokens = [
        '<recipe_start>', '<title_start>', '<title_end>', '<ingredient_start>', '<ingredient_next>', '<ingredient_end>',
        '<directions_start>', '<directions_next>', '<directions_end>', '<calories_start>', '<calories_end>',
        '<fatcontent_start>', '<fatcontent_end>', '<carbohydratecontent_start>', '<carbohydratecontent_end>',
        '<proteincontent_start>', '<proteincontent_end>', '<prep_time_min_start>', '<prep_time_min_end>',
        '<type_start>', '<type_end>', '<diet_start>', '<diet_end>', '<recipe_end>'
    ]

    def add_missing_token(recipe, token, prev_token):
        """
        Helper function to add missing tokens in the correct position.
        """
        if token == '<recipe_start>':
            return f'{token} {recipe}'
        elif token == '<recipe_end>':
            return f'{recipe} {token}'
        elif token == '<ingredient_end>':
            next_token = '<directions_start>'
            return recipe.replace(next_token, f'{token} {next_token}')
        elif token == '<directions_end>':
            next_token = '<calories_start>'
            return recipe.replace(next_token, f'{token} {next_token}')
        elif token == '<title_end>':
            next_token = '<ingredient_start>'
            return recipe.replace(next_token, f'{token} {next_token}')
        else:
            return recipe.replace(prev_token, f'{prev_token} {token}', 1)

    # Ensure all tokens are present exactly once
    for i, token in enumerate(tokens):
        token_count = recipe.count(token)
        if token_count == 0:
            prev_token = tokens[i - 1]
            recipe = add_missing_token(recipe, token, prev_token)
        elif token_count > 1 and token not in ['<ingredient_next>', '<directions_next>']:
            recipe = recipe.replace(token, '', token_count - 1)

    # Ensure no extra characters between tokens
    for i in range(len(tokens) - 1):
        pattern = rf'{tokens[i]}\s*{tokens[i + 1]}'
        replacement = f'{tokens[i]} {tokens[i + 1]}'
        recipe = re.sub(pattern, replacement, recipe)

    # Remove any extra spaces and newlines
    recipe = re.sub(r'\s+', ' ', recipe).strip()
    return recipe

def clean_title(recipe):
    """
    This method cleans the title section of the recipe by removing substrings that resemble <title_start> and <title_end> tokens.
    If there are paired occurrences of '<' and '>', the content inside along with them is removed.
    If there are unpaired occurrences, they are simply deleted.
    """
    title_pattern = re.compile(r'<title_start>(.*?)<title_end>', re.DOTALL)
    match = title_pattern.search(recipe)
    
    if match:
        title_content = match.group(1).strip()
        
        # Count the occurrences of '<' and '>'
        num_less_than = title_content.count('<')
        num_greater_than = title_content.count('>')
        
        if num_less_than == num_greater_than:
            # Paired occurrences: Remove content inside along with '<' and '>'
            title_content = re.sub(r'<.*?>', '', title_content)
        else:
            # Unpaired occurrences: Simply delete '<' and '>'
            title_content = title_content.replace('<', '').replace('>', '')
        
        # Remove extra spaces
        title_content = re.sub(r'\s+', ' ', title_content).strip()
        
        # Update the recipe with the cleaned title content
        recipe = recipe.replace(match.group(0), f'<title_start> {title_content} <title_end>')
    else:
        return "The title section is empty."
    return recipe

def clean_ingredients(recipe):
    """
    This method cleans the ingredients section of the recipe by ensuring proper formatting and removing unnecessary characters.
    """
    pattern = re.compile(r'<ingredient_start>(.*?)<ingredient_end>', re.DOTALL)
    match = pattern.search(recipe)

    if match:
        ingredients_section = match.group(1).strip()
        ingredients_section = re.sub(r'\s+', ' ', ingredients_section).strip()

        ingredients = ingredients_section.split('<ingredient_next>')
        ingredients = [ingredient for ingredient in ingredients if not ingredient.isspace() and ingredient != ""]

        for i, ingredient in enumerate(ingredients):
            ingredient = re.sub(r'\s+', ' ', ingredient).strip()
            ingredient = re.sub(r'[^\w\s,.:;!?()-]', '', ingredient)
            ingredient = ingredient.strip()
            ingredients[i] = ingredient

        cleaned_ingredients_section = ' <ingredient_next> '.join(ingredients)
        recipe = re.sub(rf'<ingredient_start>.*?<ingredient_end>', f'<ingredient_start> {cleaned_ingredients_section} <ingredient_end>', recipe)
    else:
        return "The ingredients section is empty."

    return recipe

def clean_directions(recipe):
    """
    This method cleans the directions section of the recipe by ensuring proper formatting and removing unnecessary characters.
    """
    pattern = re.compile(r'<directions_start>(.*?)<directions_end>', re.DOTALL)
    match = pattern.search(recipe)

    if match:
        directions_section = match.group(1).strip()
        directions_section = re.sub(r'\s+', ' ', directions_section).strip()

        directions = directions_section.split('<directions_next>')
        directions = [direction for direction in directions if not direction.isspace() and direction != ""]

        for i, direction in enumerate(directions):
            direction = re.sub(r'\s+', ' ', direction).strip()
            direction = re.sub(r'[^\w\s,.:;!?()-]', '', direction)
            direction = direction.strip()
            directions[i] = direction

        cleaned_directions_section = ' <directions_next> '.join(directions)
        recipe = re.sub(rf'<directions_start>.*?<directions_end>', f'<directions_start> {cleaned_directions_section} <directions_end>', recipe)
    else:
        return "The directions section is empty."

    return recipe

def clean_type(recipe):
    """
    This method cleans the type section of the recipe by ensuring that the type value is valid. If the type value is not valid, it is replaced with the closest valid type.
    For example, if the type value is "din3r" it will be replaced with "dinner".
    """
    allowed_types = ['appetizer', 'dinner', 'lunch', 'breakfast_and_brunch', 'desert']

    # Find the type section
    pattern = re.compile(r'<type_start>(.*?)<type_end>', re.DOTALL)
    match = pattern.search(recipe)
    
    if match:
        type_value = match.group(1).strip()
        # Check if the type value is valid
        if type_value not in allowed_types:
            # Find the closest match
            closest_match = difflib.get_close_matches(type_value, allowed_types, n=1)
            if closest_match:
                correct_type = closest_match[0]
                # Replace the incorrect type with the closest valid type
                recipe = re.sub(rf'<type_start>.*?<type_end>', f'<type_start> {correct_type} <type_end>', recipe)
                # Recursively call the function to ensure the replacement is valid
                return clean_type(recipe)
    else:
        return "The type value is empty."
    return recipe

def clean_diet(recipe):
    """
    This method cleans the diet section of the recipe by ensuring that the diet value is valid. If the diet value is not valid, it is replaced with the closest valid diet.
    """
    allowed_diets = ['vegan', 'vegetarian', 'contains_meat']

    # Broader set of terms indicating a meat-inclusive diet
    meat_indicating_terms = [
        'omnivorous', 'paleo', 'keto', 'mediterranean', 'carnivorous',
        'meat', 'meaty', 'meat_included',
        'non_vegetarian', 'non_veg', 'flexitarian'
    ]
    # Find the diet section
    pattern = re.compile(r'<diet_start>(.*?)<diet_end>', re.DOTALL)
    match = pattern.search(recipe)
    
    # Check if the diet section is present    
    if match:
        diet_value = match.group(1).strip()
        # Check if diet_value is composed of only spaces or is empty
        if diet_value == "" or diet_value.isspace():
            return "The diet value is empty."
        # Check if the diet value is valid
        if diet_value not in allowed_diets:
            # Find the closest match or classify meat-indicating terms as 'contains_meat'
            if diet_value.lower() in meat_indicating_terms:
                correct_diet = 'contains_meat'
            else:
                closest_match = difflib.get_close_matches(diet_value, allowed_diets, n=1)
                correct_diet = closest_match[0] if closest_match else 'contains_meat'
            
            # Replace the incorrect diet with the closest valid diet
            recipe = re.sub(r'<diet_start>.*?<diet_end>', f'<diet_start> {correct_diet} <diet_end>', recipe)
    else:
        return "The diet value is empty."
    return recipe

def clean_numbers(recipe, section_name):
    """
    This method cleans the numbers in the recipe by removing all the non-numeric characters from the section specified by the section_name.
    For example, if the section_name is "calories" and the recipe is "<calories_start>300H<calories_end>" the method will return "<calories_start>300<calories_end>"
    """
    # Dictionary mapping section names to their corresponding tags
    sections = {
        'prep_time': 'prep_time_min',
        'calories': 'calories',
        'fat_content': 'fatcontent',
        'carbohydrate_content': 'carbohydratecontent',
        'protein_content': 'proteincontent'
    }
    
    section = sections.get(section_name)
    
    recipe = recipe.replace("\n", " ")
    section_start_count = recipe.count(f'<{section}_start>')
    section_end_count = recipe.count(f'<{section}_end>')

    if section_start_count == 0: # section_start is missing
        recipe = recipe.replace('<recipe_start>', f'<recipe_start> <{section}_start>', 1)

    if section_end_count == 0: # section_end is missing
        recipe = recipe.replace('<ingredient_start>', f'<{section}_end> <ingredient_start>', 1)
    
    if section_start_count > 1: # section_start is present more than once
        recipe = recipe.replace(f'<{section}_start>', '', section_start_count - 1)

    if section_end_count > 1: # section_end is present more than once
        recipe = recipe.replace(f'<{section}_end>', '', section_end_count - 1)

    if section:
        pattern = re.compile(rf'<{section}_start>(.*?)<{section}_end>')
        match = pattern.search(recipe)
        if match:
            section_value = match.group(1).strip()
            if section_value == "" or section_value.isspace():
                return f"The {section_name} value is empty."
            clean_section_value = re.sub(r'\D', '', section_value)
            recipe = re.sub(rf'<{section}_start>.*?<{section}_end>', f'<{section}_start> {clean_section_value} <{section}_end>', recipe)
        else:
            return f"The {section_name} section is empty."
    return recipe

def clean_recipe(recipe):
    """
    This method cleans the recipe by calling the individual cleaning functions for each section of the recipe.
    """
    cleaned_recipe = clean_main_structure(recipe)
    cleaned_recipe = clean_title(cleaned_recipe)
    cleaned_recipe = clean_ingredients(cleaned_recipe)
    cleaned_recipe = clean_directions(cleaned_recipe)
    cleaned_recipe = clean_numbers(cleaned_recipe, 'calories')
    cleaned_recipe = clean_numbers(cleaned_recipe, 'fat_content')
    cleaned_recipe = clean_numbers(cleaned_recipe, 'carbohydrate_content')
    cleaned_recipe = clean_numbers(cleaned_recipe, 'protein_content')
    cleaned_recipe = clean_numbers(cleaned_recipe, 'prep_time')
    cleaned_recipe = clean_type(cleaned_recipe)
    cleaned_recipe = clean_diet(cleaned_recipe)
    return cleaned_recipe



def find_recipe_errors(recipe):
    errors = []
    for method in dir(TestRecipeGeneration):
        if method.startswith("test_"):
            recipeTester= TestRecipeGeneration(recipe_output=recipe)
            recipeTester.setUp()
            try:
                getattr(recipeTester, method)()
            except AssertionError as e:
                errors.append(str(e))
    return errors

def request_and_clean_recipe(ingredients, max_attempts=3,model='ft:gpt-3.5-turbo-1106:personal:recipecreatorv7b:9XU88CKN'):
    """
    Requests a recipe based on the given ingredients, cleans it, and checks for issues.
    If issues are found, requests corrections from the model up to a maximum number of attempts.
    
    Args:
        ingredients (list): List of ingredients.
        max_attempts (int): Maximum number of attempts to request corrections from the model.
    
    Returns:
        str: The final cleaned recipe.
    """
    attempt = 0
    recipe = ""
    
    while attempt < max_attempts:
        attempt += 1
        if attempt == 1:
            recipe = send_message_to_recipe_model(f"ingredients: {ingredients}",model=model)
        else:#
            #we request from the model the exact same recipe but with the corrections
            recipe = send_message_to_recipe_model(f"the recipe is: {recipe} and it has the following errors {errors} please correct them and return me the exact same recipe that i just gave you with the corrections",model=model)
        
        #we clean the recipe
        cleaned_recipe = clean_recipe(recipe)

        errors = find_recipe_errors(cleaned_recipe)
        if not errors:
            return cleaned_recipe

    return f"Failed to generate a clean recipe after {max_attempts} attempts. Last attempt:\n{cleaned_recipe}"

def generate_recipe_with_diet(ingredients, diet, max_attempts=3,model='ft:gpt-3.5-turbo-1106:personal:recipecreatorv7b:9XU88CKN'):
    """
    Generates a recipe based on the given list of ingredients and specified diet,
    cleans it, and checks for issues. If issues are found, requests corrections 
    from the model up to a maximum number of attempts.

    Args:
        ingredients (list): List of ingredients.
        diet (str): Specified diet (e.g., 'vegetarian', 'vegan', 'contains_meat').
        max_attempts (int): Maximum number of attempts to request corrections from the model.

    Returns:
        str: The final cleaned recipe.
    """
    # Ensure the diet is specified correctly in the request
    request_message = f"ingredients: {ingredients}\ndiet: {diet}"
    
    cleaned_recipe = request_and_clean_recipe(request_message, max_attempts,model=model)
    
    # If the final result indicates failure, handle it here as needed
    if "Failed to generate a clean recipe" in cleaned_recipe:
        print("Could not generate a clean recipe after maximum attempts.")
        return cleaned_recipe

    # Otherwise, return the successfully cleaned recipe
    return cleaned_recipe

def generate_recipe_with_type(ingredients, type, max_attempts=3,model='ft:gpt-3.5-turbo-1106:personal:recipecreatorv7b:9XU88CKN'):
    """
    Generates a recipe based on the given list of ingredients and specified type,
    cleans it, and checks for issues. If issues are found, requests corrections 
    from the model up to a maximum number of attempts.

    Args:
        ingredients (list): List of ingredients.
        type (str): Specified type (e.g., 'vegetarian', 'vegan', 'contains_meat').
        max_attempts (int): Maximum number of attempts to request corrections from the model.

    Returns:
        str: The final cleaned recipe.
    """
    # Ensure the type is specified correctly in the request
    request_message = f"ingredients: {ingredients}\ntype: {type}"
    
    cleaned_recipe = request_and_clean_recipe(request_message, max_attempts,model=model)
    
    # If the final result indicates failure, handle it here as needed
    if "Failed to generate a clean recipe" in cleaned_recipe:
        print("Could not generate a clean recipe after maximum attempts.")
        return cleaned_recipe

    # Otherwise, return the successfully cleaned recipe
    return cleaned_recipe

def generate_recipe(ingredients, diet=None, recipe_type=None, max_attempts=3, model='ft:gpt-3.5-turbo-1106:personal:recipecreatorv7b:9XU88CKN'):
    """
    Generates a recipe based on the given list of ingredients, specified diet, and recipe type.
    Cleans it, and checks for issues. If issues are found, requests corrections from the model
    up to a maximum number of attempts.

    Args:
        ingredients (list): List of ingredients.
        diet (str, optional): Specified diet (e.g., 'vegetarian', 'vegan', 'contains_meat').
        recipe_type (str, optional): Specified type (e.g., 'appetizer', 'dinner', 'lunch', 'breakfast_and_brunch', 'dessert').
        max_attempts (int): Maximum number of attempts to request corrections from the model.

    Returns:
        str: The final cleaned recipe.
    """
    
    # Build the request message based on the provided parameters
    request_message = f"ingredients: {ingredients}\n"
    if diet:
        request_message += f"it must have the diet, and you can change the ingredients to a vegan version (example: vegan Honey, or vegan chicken) if based on your knowledge they are not: {diet}\n"
    if recipe_type:
        request_message += f"and it must be of type: {recipe_type}"

    cleaned_recipe = request_and_clean_recipe(request_message, max_attempts, model=model)
    
    # If the final result indicates failure, handle it here as needed
    if "Failed to generate a clean recipe" in cleaned_recipe:
        print("Could not generate a clean recipe after maximum attempts.")
        return cleaned_recipe

    # Validate the diet in the cleaned recipe, if specified
    if diet:
        given_diet = re.search(r'<diet_start>(.*?)<diet_end>', cleaned_recipe).group(1).strip()
        if given_diet != diet:
            # Request the model to correct the diet
            recipe = send_message_to_recipe_model(
                f"the recipe is: {cleaned_recipe} and it has the following error: wrong diet is given, expected '{diet}', "
                f"but received '{given_diet}' please correct it and return me the recipe adapted to this diet. "
                f"Note that if you see that the ingredients are not vegan then you can specify the vegan version of them."
            ,model=model)
            cleaned_recipe = clean_diet(recipe)
            given_diet = re.search(r'<diet_start>(.*?)<diet_end>', cleaned_recipe).group(1).strip()
            if given_diet != diet:
                print(f"Could not correct the diet to '{diet}'.")
                return cleaned_recipe

    # Validate the type in the cleaned recipe, if specified
    if recipe_type:
        given_type = re.search(r'<type_start>(.*?)<type_end>', cleaned_recipe).group(1).strip()
        if given_type != recipe_type:
            # Request the model to correct the type
            recipe = send_message_to_recipe_model(
                f"the recipe is: {cleaned_recipe} and it has the following error: wrong type is given, expected '{recipe_type}', "
                f"but received '{given_type}' please correct it and return me the recipe adapted to this type."
            ,model=model)
            cleaned_recipe = clean_type(recipe)
            given_type = re.search(r'<type_start>(.*?)<type_end>', cleaned_recipe).group(1).strip()
            if given_type != recipe_type:
                print(f"Could not correct the type to '{recipe_type}'.")
                return cleaned_recipe

    # Otherwise, return the successfully cleaned recipe
    return cleaned_recipe



### Unit tests

