import boto3
import json
from statistics import mean
import re
import ast

# initialisation du client boto3
client = boto3.client("bedrock-runtime", region_name="us-west-2")
model_id = "us.anthropic.claude-haiku-4-5-20251001-v1:0" #inference profile

### ----- AJOUTER l'USER PROMPT
def add_user_message(messages, text):
    user_message = {
        "role": "user",
        "content": [
            {"text": text}
        ]
    }
    messages.append(user_message)

### ---- CLAUDE 
def add_assistant_message(messages, text):
    assistant_message = {
        "role": "assistant", 
        "content": [
            {"text": text}
        ]
    }
    messages.append(assistant_message)

### résultat de claude
def chat(messages, system=None, temperature=1, stop_sequences=[]): # None rend le paramètre system optionnel
    
    params = {
            
            "modelId": model_id, "messages": messages,
              
            "inferenceConfig": {
                "temperature": temperature,
                "stopSequences": stop_sequences
            }, # température de 0 à 0.3 = deterministe -> pour du code. max 1 -->(créatif)
                
            }
    
    if system:
        params["system"] = [{"text": system}]
    
    response = client.converse(**params) # ** pour récupérer les éléments d'un dictionnaire
    return response["output"]["message"]["content"][0]["text"] 

# générer dataset en utilisant le prefilling
def generate_dataset():
    prompt = """
    Generate 3 AWS-related tasks that require Python, JSON, or Regex solutions.
    
    Focus on tasks that can be solved by writing a single Python function, 
    a single JSON object, or tasks that do not require writing much code.
    
    Examples output:
    [
        {
            "task": "Description of task",
            "format": "json" or "python" or "regex"
            "solution_criteria": "key criteria for evaluating the solution"
        },     
    ]
    
    Please generate 3 objects.
    """
    messages = []

    add_user_message(messages,prompt)
    add_assistant_message(messages,"```json")
    text = chat(messages, stop_sequences=["```"])
    return json.loads(text)

#messages = []
#add_user_message(messages,"prompt")
#add_assistant_message(messages,"```json:")
#stream = client.converse_stream(messages=messages, modelId=model_id)
#reponse = chat(messages, stop_sequences=["```"])
#texte = ""
#for event in stream['stream']: #on doit itérer sur la clé stream pour afficher chaque évènement
#    if "contentBlockDelta" in event:
#        chunk = event["contentBlockDelta"]["delta"]["text"]
#        texte += chunk=


# noter la sortie de claude par rapport à la task du dataset en entrée. Retourner au format json en sortie

def validate_json(text):
    try:
        json.loads(text.strip())
        return 10
    except json.JSONDecodeError:
        return 0

def validate_python(text):
    try:
        ast.parse(text.strip())
        return 10
    except SyntaxError:
        return 0

def validate_regex(text):
    try:
        re.compile(text.strip())
        return 10
    except re.error:
        return 0

def grade_syntax(response, test_case):
    format = test_case["format"]
    if format == "json":
        return validate_json(response)
    if format == "python":
        return validate_python(response)    
    else:
        return validate_regex(response)

def grade_by_model(test_case,claude_output):
    # Create evaluation prompt.
    # Le score seulement est insuffisant, il faut rajouter forces/faiblesses/reasoning sinon le résultat tendera souvenet vers le même score.
    eval_prompt = f"""
    You are an expert AWS code reviewer. Your task is to evaluate the following AI-generated solution.
    
    Original Task:
    <task>
    {test_case["task"]}
    </task>
    
    Solution to Evaluate:
    <solution>
    Solution: {claude_output}
    </solution>
    
    Criteria you should use to evaluate the solution:
    <criteria>
    {test_case["solution_criteria"]}
    </criteria>
    
    Output format
    Provide your evaluation as a structured JSON object with:
    - "strengths": An array of 1-3 key strengths
    - "weaknesses": An array of 1-3 key areas for improvement  
    - "reasoning": A concise explanation of your assessment
    - "score": A number between 1-10
    """
    
    messages = []
    add_user_message(messages, eval_prompt)
    add_assistant_message(messages, "```json")
    
    eval_text = chat(messages, stop_sequences=["```"])
    return json.loads(eval_text)

# formate la réponse de claude pour n'obtenir que du code
def run_prompt(test_case):
    prompt = f"""
    Please solve the following task:

    {test_case["task"]}
    * Respond only with Python, JSON, or a plain Regex.
    * Do NOT add any comments or commentary or explaination.
    """
    messages = []
    add_user_message(messages,prompt)
    add_assistant_message(messages,"```code")
    claude_output = chat(messages,stop_sequences=["```"])
    return claude_output    

# teste et retourne les résultats par task
def run_test_case(test_case):
    claude_output = run_prompt(test_case)
    
    # Get model evaluation
    model_grade = grade_by_model(test_case, claude_output)
    score = model_grade["score"]
    reasoning = model_grade["reasoning"]
    
    syntax_score = grade_syntax(claude_output,test_case)
    final_score = (score + syntax_score) / 2
    print(f"le score est de {score} et le syntax score est de {syntax_score} et le score finale est de {final_score}")
    return {
        "output": claude_output, 
        "test_case": test_case, 
        "score": final_score,
        "reasoning": reasoning
    }

# retourne les résultats des tests pour tous les dataset et affiche la moyenne du score
def run_eval(dataset):
    results = []
    
    for test_case in dataset:
        result = run_test_case(test_case)
        results.append(result)
    
    average_score = mean([result["score"] for result in results])
    print(f"score final: {average_score}")
    
    return results
    
dataset = generate_dataset()
with open("dataset.json","w") as f: # crée le fichier
       json.dump(dataset,f,indent=2) # convertit dataset en json

#with open("dataset.json", "r") as f:
 #   dataset = json.load(f)

print(dataset)
resultat = run_eval(dataset)

print(json.dumps(resultat,indent=2))

