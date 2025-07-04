from transformers import TFAutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline
from googletrans import Translator
import json

model_save_directory = "saved_model"
tokenizer_save_directory = "saved_tokeniser"
num_classes = 24

# Loading the model
loaded_model = TFAutoModelForSequenceClassification.from_pretrained(model_save_directory)

# Loading the tokenizer
loaded_tokenizer = AutoTokenizer.from_pretrained(tokenizer_save_directory)

translator = Translator()


file_path = 'diseases.json'
# Loading the JSON data from the file
with open(file_path, 'r') as file:
    diseases_info = json.load(file)
print(diseases_info)
# Example: Print the overview of Psoriasis
print(diseases_info['Psoriasis']['Overview'])

def predict(prompt):
	# Creating the pipeline with the loaded model and tokenizer
	pipe = TextClassificationPipeline(model=loaded_model, tokenizer=loaded_tokenizer, top_k=num_classes)

	pred1 = pipe(prompt)

	print(pred1[0][:2])
	return pred1[0][0]

def translaste_and_predict(prompt, lang):
	print(lang)
	eng_prompt = translator.translate(prompt, dest='en').text
	print(eng_prompt)
	predicted = predict(eng_prompt)
	# ans = "I have " + str(int(predicted['score']*100)) + "% confidence that you might be suffering from the " + predicted['label']
	# print(ans)
	ans = create_html_response(predicted)
	disease_local_lang = translator.translate(ans, dest=lang).text
	print(disease_local_lang)
	return disease_local_lang

def detect_lang(prompt):
	lang_detected = translator.detect(prompt).lang
	print('prompt: ', prompt, 'language detected: ', translator.detect(prompt).lang)
	return lang_detected

def get_disease_info(predicted_label):
    """
    Fetches the disease information for the given label from the loaded JSON data.
    """
    disease_data = diseases_info.get(predicted_label, {})
    if not disease_data:
        return "I couldn't find information on this condition."
    
    overview = disease_data.get('Overview', 'No overview available.')
    precautions = disease_data.get('Precautions', 'No precautions available.')
    suggestions = disease_data.get('Suggestions', 'No suggestions available.')
    
    return overview, precautions, suggestions

def create_response(predicted):
    """
    Creates a response string incorporating the disease prediction and information.
    """
    overview, precautions, suggestions = get_disease_info(predicted['label'])
    confidence = str(int(predicted['score'] * 100))
    
    response = (f"I have {confidence}% confidence that you might be suffering from {predicted['label']}. "
                f"\n\nOverview: {overview} "
                f"\n\nPrecautions: {precautions} "
                f"\n\nSuggestions: {suggestions}")
    
    return response

def create_html_response(predicted):
    """
    Creates a response string formatted with HTML line breaks.
    """
    overview, precautions, suggestions = get_disease_info(predicted['label'])
    confidence = str(int(predicted['score'] * 100))
    
    response = (f"I have {confidence}% confidence that you might be suffering from {predicted['label']}. "
                f"<br><br>Overview: {overview} "
                f"<br><br>Precautions: {precautions} "
                f"<br><br>Suggestions: {suggestions}")
    
    return response
# # Example usage
# predicted = {'label': 'Psoriasis', 'score': 0.85}  # Example predicted output
# response = create_response(predicted)
# print(response)

# prompt = "I am having severe stomach ache and vomitings"
# prompt_telugu= "నాకు విపరీతమైన కడుపునొప్పి, వాంతులు అవుతున్నాయి"
# prompt_hindi = "मुझे पेट में तेज़ दर्द और उल्टी हो रही है"
# translaste_and_predict(prompt_hindi)