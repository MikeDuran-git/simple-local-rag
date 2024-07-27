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