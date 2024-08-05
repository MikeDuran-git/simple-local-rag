import torch
import os
import glob
import re
import fitz # (pymupdf, found this is better than pypdf for our use case, note: licence is AGPL-3.0, keep that in mind if you want to use any code commercially)
from tqdm.auto import tqdm # for progress bars, requires !pip install tqdm 
import pandas as pd
from tqdm import tqdm
from spacy.lang.en import English
from sentence_transformers import SentenceTransformer

tqdm.pandas()

# Get PDF document
pdf_path = "greg_doucette_cookbook_2_0.pdf"



#check gpu access
if torch.cuda.is_available():
    print("Using GPU: "+torch.cuda.get_device_name(0) + " is available")
else: #we are using cpu
    print("Using CPU")


# Download PDF if it doesn't already exist
if not os.path.exists(pdf_path):
  print("File doesn't exist, go to the link in the README to download it.")
else:
  print(f"File {pdf_path} exists.")


#preprocessing code :

def text_formatter(text: str) -> str:
    """Performs minor formatting on text."""
    cleaned_text = text.replace("\n", " ").strip() # note: this might be different for each doc (best to experiment)

    # Other potential text formatting functions can go here
    return cleaned_text

# Open PDF and get lines/pages
# Note: this only focuses on text, rather than images/figures etc
def open_and_read_pdf(pdf_path: str) -> list[dict]:
    """
    Opens a PDF file, reads its text content page by page, and collects statistics.

    Parameters:
        pdf_path (str): The file path to the PDF document to be opened and read.

    Returns:
        list[dict]: A list of dictionaries, each containing the page number
        (adjusted), character count, word count, sentence count, token count, and the extracted text
        for each page.
    """
    doc = fitz.open(pdf_path)  # open a document
    pages_and_texts = []
    for page_number, page in tqdm(enumerate(doc)):  # iterate the document pages
        text = page.get_text()  # get plain text encoded as UTF-8
        #text = text_formatter(text)
        pages_and_texts.append({"page_number": page_number+1,
                                 "page_char_count": len(text),
                                 "page_word_count": len(text.split(" ")),
                                 "page_sentence_count_raw": len(text.split(". ")),
                                "page_token_count": len(text) / 4,  # 1 token = ~4 chars, see: https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
                                "text": text})
    return pages_and_texts


def extract_directions(text: str) -> str:
    # List of separators to use in the search
    separators = [
        "N u t r i t i o n",
        "N o t e",
        "N U T R I T I O N",
        "N O T E",
        "P R E P",
        "T O T A L",
        "V E G E T A R I A N",
        "V E G A N",
        "\nIngredients\n",
        "R E A D Y"
    ]
    
    # Construct a regex pattern to match any of the separators
    separator_pattern = r'|'.join([re.escape(separator) for separator in separators])
    
    # Remove the "number.\n" -> "number. " since it is not useful
    text = re.sub(r'(\d+)\.\n', r'\1. ', text)
    
    # Search for the section "Directions"
    directions_match = re.search(r'Directions\s*([\s\S]*?)\s*(' + separator_pattern + r')', text, re.IGNORECASE)
    if directions_match:
        # Return the found directions
        return directions_match.group(1).strip()
    return None

def extract_ingredients(text: str) -> str:
    # List of separators to use in the search
    separators = [
        "N u t r i t i o n",
        "N o t e",
        "N U T R I T I O N",
        "N O T E",
        "P R E P",
        "T O T A L",
        "V E G E T A R I A N",
        "V E G A N",
        "R E A D Y",
        "C L I C K",
    ]
    
    # Construct a regex pattern to match any of the separators
    separator_pattern = r'|'.join([re.escape(separator) for separator in separators])
    
    # Remove the "number.\n" -> "number. " since it is not useful
    text = re.sub(r'(\d+)\.\n', r'\1. ', text)
    
    # Search for the section "Directions"
    ingredients_match = re.search(r'\nIngredients\n\s*([\s\S]*?)\s*(' + separator_pattern + r')', text, re.IGNORECASE)
    if ingredients_match:
        # Return the found ingredients
        return ingredients_match.group(1).strip()
    return None

# Function to extract prep time and ready in time
def extract_prep_time_and_or_ready_in(text: str) -> dict:
    prep_time = None
    ready_in = None

    # Find all occurrences of "MINUTES" and "HOURS" in uppercase along with their preceding numbers
    times = re.findall(r'(\d+)\s*(MINUTES|HOUR)', text)

    # Find the occurrence of "P R E P  T I M E" and "R E A D Y  I N"
    prep_time_index = text.find("\nP R E P  T I M E")
    ready_in_index = text.find("\nR E A D Y  I N")

    if prep_time_index != -1 and ready_in_index != -1:
        # Determine the order of prep time and ready in time based on their positions in the text
        if prep_time_index < ready_in_index:
            if len(times) > 0:
                prep_time = times[0][0] + " " + times[0][1] if times[0][1] == "MINUTES" else None
            if len(times) > 1:
                ready_in = times[1][0] + " " + times[1][1]
        else:
            if len(times) > 0:
                ready_in = times[0][0] + " " + times[0][1]
            if len(times) > 1:
                prep_time = times[1][0] + " " + times[1][1] if times[1][1] == "MINUTES" else None
    elif prep_time_index != -1:
        prep_time = times[0][0] + " " + times[0][1] if len(times) > 0 and times[0][1] == "MINUTES" else None
    elif ready_in_index != -1:
        ready_in = times[0][0] + " " + times[0][1] if len(times) > 0 else None

    return {
        "prep_time": prep_time,
        "ready_in": ready_in
    }

# Function to remove specific phrases from text
def remove_phrases(text, phrases):
    for phrase in phrases:
        text = text.replace(phrase, '')
    return text

#definition of the method to retreive the dataframe
def get_dataframe_from_table_page(text):

    #first off we remove the lines until we find the first occurence of "Vegetarian" and we drop it along with everything that came before him.
    text = text[text.find("Vegetarian"):].replace("Vegetarian","")

    #when we see the pattern of any \n next to "-" we remove the "\n"
    text = text.replace("\n-","-").replace("\n -"," -").replace("-\n","- ").replace("\n-\n","-").replace("- \n","- ")

    # if we see \n next to serving or servings we remove it
    text = re.sub(r'\n(?=\b[sS]erving[s]?\b)', '', text)

    # #we remove the first \n
    text = text.replace("\n"," ",1)

    #we remove "\nY\n"
    text = text.replace("\nY","")

    array=text.split("\n")
    #we remove unwanted spaces
    array = [i.strip() for i in array]
    #we remove empty strings
    array = list(filter(None, array))

    #on parcours le tableau, et nous calculons la longueur de chaque element. Si la le voisin i+1 est plus long que 3 c'est que c'est un text qui est coupé en deux, on le merge avec le voisin i. SAUF si i a une taille infieur ou égale à 3, car dans ce cas i est un nombre.
    for i in range(len(array) - 1):
        if len(array[i+1]) > 3 and len(array[i]) > 3 and not array[i+1].isnumeric() and not is_number_without_period(array[i]):
            array[i] = array[i] + " " + array[i+1]
            array[i+1] = ""

    # we remove the empty strings
    array = list(filter(None, array))
    #we now that the first 7 elements are 1 recipe, so we can append by chunks of 7. The first element is the page, the second the recipe, the third the calories, the fourth the fat, the fifth the carbs, the sixth the fiber, the seventh the protein.

    pages=[]
    recipes=[]
    calories=[]
    fat=[]
    carbs=[]
    fiber=[]
    protein=[]

    for i in range(0,len(array),7):
        pages.append(array[i])
        recipes.append(array[i+1])
        calories.append(array[i+2])
        fat.append(array[i+3])
        carbs.append(array[i+4])
        fiber.append(array[i+5])
        protein.append(array[i+6])
        

    #we create a dictionnary to store the data
    nutrition_data = {
        "Page": pages,
        "Recipe": recipes,
        "Calories Per Serving": calories,
        "Fat (g) per serving": fat,
        "Carbs (g) per serving": carbs,
        "Fiber (g) per serving": fiber,
        "Protein (g) per serving": protein,
    }

    #we create a dataframe
    page_dataframe = pd.DataFrame(nutrition_data)

    return page_dataframe


# to remove . in the page number
def is_number_without_period(s):
    return s.replace(".", "").isnumeric()


#### create Recipe Dataset
def get_recipe_from_page(page_number):
    page_data = P_D_I_P_R_dataframe[P_D_I_P_R_dataframe['page_number'] == page_number]
    if page_data.empty:
        return "No data found for the given page number."

    extract_directions = page_data['directions'].values[0]
    extract_ingredients = page_data['ingredients'].values[0]
    extracted_prep_time = page_data['prep_time_minutes'].values[0]
    extracted_ready_in = page_data['ready_in_minutes'].values[0]

    text = f"DIRECTIONS:\n{extract_directions}\n\nINGREDIENTS:\n{extract_ingredients}\n\nPREP TIME (in m): {extracted_prep_time}\n\nREADY IN (in m): {extracted_ready_in}"
    return text

def get_title_and_macros(page_number):
    recipes = P_T_C_F_C_F_P_dataframe[P_T_C_F_C_F_P_dataframe['Page'] == page_number]
    if recipes.empty:
        return "No data found for the given page number."

    text = ""
    for _, recipe in recipes.iterrows():
        title = recipe['Recipe']
        calories = recipe['Calories Per Serving']
        fat = recipe['Fat (g) per serving']
        carbs = recipe['Carbs (g) per serving']
        fiber = recipe['Fiber (g) per serving']
        protein = recipe['Protein (g) per serving']
        text += f"TITLE: {title}\nCALORIES (in kcal): {calories}\nFAT (in g): {fat}\nCARBS (in g): {carbs}\nFIBER (in g): {fiber}\nPROTEIN (in g): {protein}\n\n"
    return text

def get_recipe(page_number):
    title_and_macros = get_title_and_macros(page_number)
    recipe = get_recipe_from_page(page_number)

    page_data = P_T_C_F_C_F_P_dataframe[P_T_C_F_C_F_P_dataframe['Page'] == page_number]
    if page_data.empty:
        return "No data found for the given page number."

    diet = "vegan" if page_data['Vegan'].values[0] == "Yes" else "non-vegan"
    diet += " , vegetarian" if page_data['Vegetarian'].values[0] == "Yes" else " , non-vegetarian"

    category = page_data['Category'].values[0]

    text = f"{title_and_macros}{recipe}\n\nCATEGORY: {category}\n\nDIET: {diet}"
    return text



def merge_numbered_elements(dic):
    merged_list = []
    i = 0
    while i < len(dic):
        if  dic[i].strip().endswith('.') and len(dic[i]) <=3:
            merged_list.append(dic[i] + ' ' + dic[i + 1])
            i += 2
        else:
            merged_list.append(dic[i])
            i += 1
    return merged_list

def get_sentences(_str):
  chunks = _str.split('\n')
  sentences = []
  nlp = English()
  nlp.add_pipe("sentencizer")
  for chunk in chunks:
    doc = nlp(chunk)
    sentences += [sent.text.strip() for sent in doc.sents]

  #we delete the empty strings
  sentences = list(filter(None, sentences))

  # if we see the "number.", for example "1.", we know that it is a step in the direction and the next sentence is the continuation of the step, so we merge them
  for i in range(len(sentences) - 1):
    if sentences[i].isnumeric() and sentences[i+1] != "":
      sentences[i] = sentences[i] + " " + sentences[i+1]
      sentences[i+1] = ""

  sentences = merge_numbered_elements(sentences)
  return sentences

# Create a function that recursively splits a list into desired sizes
def split_list(input_list: list, 
               slice_size: int) -> list[list[str]]:
    """
    Splits the input_list into sublists of size slice_size (or as close as possible).

    For example, a list of 17 sentences would be split into two lists of [[10], [7]]
    """
    return [input_list[i:i + slice_size] for i in range(0, len(input_list), slice_size)]


pages_and_texts = open_and_read_pdf(pdf_path=pdf_path)

#### Data cleaning 
table_of_contents = pages_and_texts[len(pages_and_texts)-19:len(pages_and_texts)-2]
recipes = pages_and_texts[16:len(pages_and_texts)-21]
vegetables_raw_and_legumes_servings_reference_table=table_of_contents[-3:-1]
fruits_servings_reference_table=table_of_contents[-4]

# PDIPR DATASET CREATION
df = pd.DataFrame(recipes)
df.to_csv('cookbook_data.csv', index=False)

#get the rows that contain ingredients and Directions since everything else is plain explanation
df = df[df.text.str.contains('Ingredients|Directions')]



#we test it on the first 5 rows
df['directions'] = df.text.progress_apply(extract_directions)

df['ingredients'] = df.text.progress_apply(extract_ingredients)

# Apply the function to the dataframe
df[['prep_time', 'ready_in']] = df['text'].apply(lambda x: pd.Series(extract_prep_time_and_or_ready_in(x)))

# List of phrases to remove
click_to_and_get_strings = [
    "CLICK TO ORDER ICON MEALS PROTEIN BREAD",
    "CLICK TO ORDER LOW-CALORIE SYRUP",
    "CLICK TO PURCHASE GUAR GUM",
    "CLICK TO ORDER WALDEN FARMS SYRUP",
    "CLICK TO PURCHASE A NINJA BLENDER",
    "CLICK TO PURCHASE MUSCLE EGG",
    "CLICK TO PURCHASE LIQUID MUSCLE",
    "CLICK TO PURCHASE MISSION CARB BALANCE TORTILLA",
    "CLICK TO PURCHASE YVES VEGGIE TOFU DOGS",
    "CLICK TO PURCHASE PALMINI LOW-CARB LASAGNA",
    "CLICK TO PURCHASE VEGGIE GROUND \"MEAT\"",
    "CLICK TO PURCHASE SUGAR-FREE CHOCOLATE SAUCE",
    "GET SUGAR-FREE CHOCOLATE JELL-O PUDDING",
    "GET CHOCOLATE SUGAR-FREE JELLO PUDDING MIX",
    "GET GUAR GUM",
    "GET PB2 POWDERED PEANUT BUTTER",
    "CLICK TO PURCHASE PB2 POWDERED PEANUT BUTTER",
    "CLICK TO PURCHASE PUMPKIN PURÉE",
    "CLICK TO PURCHASE PB2 (POWDERED PEANUT BUTTER)",
    "CLICK TO PURCHASE FIBER ONE BROWNIE BAR",
    "CLICK TO PURCHASE CHOCOLATE PB2 POWDER",
    "GET BANANA SUGAR-FREE JELLO PUDDING MIX",
    "CLICK TO PURCHASE WALDEN FARMS MAPLE WALNUT SYRUP",
    "CLICK TO PURCHASE HERSHEY'S HEALTH SHELL TOPPING",
    "CLICK TO PURCHASE SUGAR-FREE JELLO CHEESECAKE PUDDING POWDER",
    "CLICK TO PURCHASE LIBBY’S 100% PURE PUMPKIN",
    "CLICK TO PURCHASE SUGAR-FREE VANILLA PUDDING JELL-O",
    "GET CHOCOLATE PB2 POWDERED PEANUT BUTTER",
]


# Apply cleaning to all texts in the dataset
df['text'] = df['text'].apply(lambda x: remove_phrases(x, click_to_and_get_strings))

# Save the cleaned data to a new CSV file
cleaned_file_path = 'cleaned_cookbook_data.csv'
df.to_csv(cleaned_file_path, index=False)

#we replace de 1 hour by 60 minutes
df['ready_in'] = df['ready_in'].str.replace('1 HOUR', '60 MINUTES')
#we replace the column name ready_in by ready_in_minutes and prep_time by prep_time_minutes
df.rename(columns={'ready_in': 'ready_in_minutes', 'prep_time': 'prep_time_minutes'}, inplace=True)

#we remove the words MINUTES and HOUR
df['ready_in_minutes'] = df['ready_in_minutes'].str.replace('MINUTES', '')
df['prep_time_minutes'] = df['prep_time_minutes'].str.replace('MINUTES', '')
#we remove useless spaces
df['ready_in_minutes'] = df['ready_in_minutes'].str.strip()
df['prep_time_minutes'] = df['prep_time_minutes'].str.strip()

#we drop the useless columns LIKE page_number, page_char_count, page_word_count, page_sentence_count_raw, page_token_count
df.drop(columns=['text','page_char_count', 'page_word_count', 'page_sentence_count_raw', 'page_token_count'], inplace=True)
df.to_csv('P_D_I_P_R.csv', index=False)


##### PTCFCGP creation
master_recipe_nutrition_table = table_of_contents[:-4]



for index, page in enumerate(master_recipe_nutrition_table):
    print(f"Processing page {index}...")
    df = get_dataframe_from_table_page(page['text'])
    df.to_csv(f'nutrition_table_{index}.csv', index=False)



#we load the pages one by one
page_1_dataframe = pd.read_csv('nutrition_table_0.csv')
page_2_dataframe = pd.read_csv('nutrition_table_1.csv')
page_3_dataframe = pd.read_csv('nutrition_table_2.csv')
page_4_dataframe = pd.read_csv('nutrition_table_3.csv')
page_5_dataframe = pd.read_csv('nutrition_table_4.csv')
page_6_dataframe = pd.read_csv('nutrition_table_5.csv')
page_7_dataframe = pd.read_csv('nutrition_table_6.csv')
page_8_dataframe = pd.read_csv('nutrition_table_7.csv')
page_9_dataframe = pd.read_csv('nutrition_table_8.csv')
page_10_dataframe = pd.read_csv('nutrition_table_9.csv')
page_11_dataframe = pd.read_csv('nutrition_table_10.csv')
page_12_dataframe = pd.read_csv('nutrition_table_11.csv')
page_13_dataframe = pd.read_csv('nutrition_table_12.csv')

#we concatenate all the dataframes
master_recipe_nutrition_table = pd.concat([page_1_dataframe, page_2_dataframe, page_3_dataframe, page_4_dataframe, page_5_dataframe, page_6_dataframe, page_7_dataframe, page_8_dataframe, page_9_dataframe, page_10_dataframe, page_11_dataframe, page_12_dataframe, page_13_dataframe])

# we order it by the page number
P_T_C_F_C_F_P_dataframe = master_recipe_nutrition_table.sort_values(by='Page')

#from every row starting from 51 we reduce the page number by 1
#we replace the page column by the new values
for i in range(51,len(P_T_C_F_C_F_P_dataframe)):
    P_T_C_F_C_F_P_dataframe.iloc[i,0] = int(P_T_C_F_C_F_P_dataframe.iloc[i,0])-1

#what we now need to do is is to manually set the vegan and vegetarian column, we can check in the cookbook each recipe to see if it is vegan or vegetarian. We can see that on page 187 there are all recipe are vegetaria (except 29,30,31) and all ar non vegan. So we can just write a python hard code that does that.

#pages that are vegan:
vegan_pages = [51,66,76,78,91,121,142,152,164,169,196,170,175]

#we create the vegan column
P_T_C_F_C_F_P_dataframe['Vegan'] = "No"

#we set the vegan column to Yes for the pages that are vegan
for page in vegan_pages:
    P_T_C_F_C_F_P_dataframe.loc[P_T_C_F_C_F_P_dataframe['Page'] == page, 'Vegan'] = "Yes"

# we set the ChocolateProtein Mug Cake to no, because it is not vegan
P_T_C_F_C_F_P_dataframe.loc[P_T_C_F_C_F_P_dataframe['Recipe'] == 'Chocolate Protein Mug Cake', 'Vegan'] = "No"

#we set the vegetarian column

#we set every column as vegetarian
P_T_C_F_C_F_P_dataframe['Vegetarian'] = "Yes"

#now we just set to no the recipes that are not vegetarian
non_vegetarian_pages=[29,30,31,
                      
                      63,64,68,70,
                      
                      75,
                      80,
                      85,
                      86,87,89,90,92,93,98,99,101,102,

                      105,107,108,110,111,113,115,116,118,

                      126,

                      152,

                      ]

for page in non_vegetarian_pages:
    P_T_C_F_C_F_P_dataframe.loc[P_T_C_F_C_F_P_dataframe['Page'] == page, 'Vegetarian'] = "No"


#Egg White Wrap & Cauliflower PIzza Crust - Per 2 Meat Lovers Pizza is not vegetarian 
P_T_C_F_C_F_P_dataframe.loc[P_T_C_F_C_F_P_dataframe['Recipe'] == 'Cauliflower PIzza Crust - Per 2 Meat Lovers Pizza', 'Vegetarian'] = "No"
# Cauliflower PIzza Crust - Per 2 Meat Lovers Pizza is not vegetarian
P_T_C_F_C_F_P_dataframe.loc[P_T_C_F_C_F_P_dataframe['Recipe'] == 'Cauliflower PIzza Crust - Per 2 Meat Lovers Pizza', 'Vegetarian'] = "No"
# Sloppy Greg Sandwich - Total is not vegetarian
P_T_C_F_C_F_P_dataframe.loc[P_T_C_F_C_F_P_dataframe['Recipe'] == 'Sloppy Greg Sandwich - Total', 'Vegetarian'] = "No"
# Sloppy Greg Sandwich - Per Serving is not vegetarian
P_T_C_F_C_F_P_dataframe.loc[P_T_C_F_C_F_P_dataframe['Recipe'] == 'Sloppy Greg Sandwich - Per Serving', 'Vegetarian'] = "No"
# Grilled Chicken Wrap with Mango Relish - 1 Wrap is not vegetarian
P_T_C_F_C_F_P_dataframe.loc[P_T_C_F_C_F_P_dataframe['Recipe'] == 'Grilled Chicken Wrap with Mango Relish - 1 Wrap', 'Vegetarian'] = "No"
# Egg Whites on Flatout Light OR La Tortilla OR 90-110 Calorie Wrap of Choice is vegetarian
P_T_C_F_C_F_P_dataframe.loc[P_T_C_F_C_F_P_dataframe['Recipe'] == 'Egg Whites on Flatout Light OR La Tortilla OR 90-110 Calorie Wrap of Choice', 'Vegetarian'] = "Yes"
# Egg Whites on Joseph's Lavash is vegetarian
P_T_C_F_C_F_P_dataframe.loc[P_T_C_F_C_F_P_dataframe['Recipe'] == 'Egg Whites on Joseph\'s Lavash', 'Vegetarian'] = "Yes"

# Now we succeded, we can create a category based on the table of content. 
# * Breakfast: all pages from 17 to 60
# * Appetizer: all pages from 62 to 67
# * Tacos, Wraps and Sandwiches : all pages from 70 to 92
# * Dinner: 95 to 122
# * Treats: 125 to 156
# * Dessert: 158 to 185

#we create the category column
P_T_C_F_C_F_P_dataframe['Category'] = "Breakfast"
#we set the category column to the right category
P_T_C_F_C_F_P_dataframe.loc[P_T_C_F_C_F_P_dataframe['Page'].between(62, 67), 'Category'] = "Appetizer"
P_T_C_F_C_F_P_dataframe.loc[P_T_C_F_C_F_P_dataframe['Page'].between(70, 92), 'Category'] = "Tacos, Wraps and Sandwiches"
P_T_C_F_C_F_P_dataframe.loc[P_T_C_F_C_F_P_dataframe['Page'].between(95, 122), 'Category'] = "Dinner"
P_T_C_F_C_F_P_dataframe.loc[P_T_C_F_C_F_P_dataframe['Page'].between(125, 156), 'Category'] = "Treats"
P_T_C_F_C_F_P_dataframe.loc[P_T_C_F_C_F_P_dataframe['Page'].between(158, 186), 'Category'] = "Dessert"
#we save the dataframe
P_T_C_F_C_F_P_dataframe.to_csv('P_T_C_F_C_F_P.csv', index=False)

#Optional, we can delete the nutrition_table csv files
files = glob.glob('nutrition_table_*.csv')
for f in files:
    os.remove(f)

P_D_I_P_R_dataframe = pd.read_csv('P_D_I_P_R.csv')
P_T_C_F_C_F_P_dataframe = pd.read_csv('P_T_C_F_C_F_P.csv')

recipe_dataset = pd.DataFrame(columns=['page_number',
                                       'page_char_count',
                                        'page_word_count',
                                        'page_sentence_count_raw',
                                        'page_token_count',
                                       'text'])

recipe_dataset['page_number'] = P_D_I_P_R_dataframe['page_number']
recipe_dataset['text'] = recipe_dataset['page_number'].progress_apply(get_recipe)
recipe_dataset['page_char_count'] = recipe_dataset['text'].progress_apply(lambda x: len(x))
recipe_dataset['page_word_count'] = recipe_dataset['text'].progress_apply(lambda x: len(x.split(" ")))
recipe_dataset['page_sentence_count_raw'] = recipe_dataset['text'].progress_apply(lambda x: len(x.split(". ")))
                                                                         
recipe_dataset['page_token_count'] = recipe_dataset['page_char_count'] / 4



recipe_dataset['sentences'] = recipe_dataset['text'].progress_apply(get_sentences)

#we add a column that counts the page_sentence_count_spacy
recipe_dataset['page_sentence_count_spacy'] = recipe_dataset['sentences'].progress_apply(lambda x: len(x))

# Define split size to turn groups of sentences into chunks
num_sentence_chunk_size = 10 


# Loop through pages and texts and split sentences into chunks
recipe_dataset["sentence_chunks"] = recipe_dataset["sentences"].progress_apply(lambda x: split_list(input_list=x,
                                         slice_size=num_sentence_chunk_size))
recipe_dataset["num_chunks"] = recipe_dataset["sentence_chunks"].progress_apply(lambda x: len(x))



recipe_dataset.to_csv('recipe_dataset.csv', index=False)


# Split each chunk into its own item
pages_and_chunks = []
for item in tqdm(recipe_dataset.to_dict(orient="records")):
    for sentence_chunk in item["sentence_chunks"]:
        chunk_dict = {}
        chunk_dict["page_number"] = item["page_number"]
        
        # Join the sentences together into a paragraph-like structure, aka a chunk (so they are a single string)
        joined_sentence_chunk = "".join(sentence_chunk).replace("  ", " ").strip()
        joined_sentence_chunk = re.sub(r'\.([A-Z])', r'. \1', joined_sentence_chunk) # ".A" -> ". A" for any full-stop/capital letter combo 
        chunk_dict["sentence_chunk"] = joined_sentence_chunk

        # Get stats about the chunk
        chunk_dict["chunk_char_count"] = len(joined_sentence_chunk)
        chunk_dict["chunk_word_count"] = len([word for word in joined_sentence_chunk.split(" ")])
        chunk_dict["chunk_token_count"] = len(joined_sentence_chunk) / 4 # 1 token = ~4 characters
        
        pages_and_chunks.append(chunk_dict)

# Requires !pip install sentence-transformers
embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2", 
                                      device="cpu") # choose the device to load the model to (note: GPU will often be *much* faster than CPU)
# Create embeddings one by one on the GPU
for item in tqdm(pages_and_chunks):
    item["embedding"] = embedding_model.encode(item["sentence_chunk"])

df=pd.DataFrame(pages_and_chunks)
#we save the dataframe : 
df.to_csv('text_chunks_and_embeddings_df.csv', index=False)

#we save the embedding model to use it later
torch.save(embedding_model, 'embedding_model.pt')