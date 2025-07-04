from flask import Flask, render_template, request, jsonify
import infer

app = Flask(__name__)

ALLOWED_LANGUAGES = ['en', 'hi', 'te']
LANGUAGE = 'en'

def get_followup_questions(language):
    if language == 'en':
        return ["What symptomns are you facing?", "From how many days you are experiencing it?", "How severe are the symptoms, rate from 1 (Low)-5(High)?"]
    elif language == 'hi':
        return ["आप किन लक्षणों का सामना कर रहे हैं?", "आप कितने दिनों से इसका अनुभव कर रहे हैं?", "लक्षण कितने गंभीर हैं? 1(निम्न) - 5(उच्च)"]
    elif language == 'te':
        return ['మీరు ఏ లక్షణాలను ఎదుర్కొంటున్నారు?', 'మీరు ఎన్ని రోజుల నుండి అనుభవిస్తున్నారు?','లక్షణాలు ఎంత తీవ్రంగా ఉన్నాయి? 1(తక్కువ) - 5(అధిక)']
    # Add more languages as needed


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        symptoms = request.form['symptoms']
        print(symptoms)
        symptoms_list = symptoms.split("| ")  
        print(symptoms_list)
        if len(symptoms_list) != 4:
            return {'response': "Please reset the chat and try again!"}
        LANGUAGE = infer.detect_lang(symptoms_list[0])
        prediction = infer.translaste_and_predict(symptoms_list[1], LANGUAGE)
        return {'response': prediction}  # Sending back the prediction as JSON

@app.route('/get_questions', methods=['POST'])
def get_questions():
    if request.method == 'POST':
        symptoms = request.form['symptoms']
        print(symptoms)
        LANGUAGE = infer.detect_lang(symptoms) # Detect language of user input
        print(LANGUAGE)
        if LANGUAGE not in ALLOWED_LANGUAGES:
            return jsonify({'error': 'Unsupported language. Only English, Hindi and Telugu are supported. Please reset the chat and try again.'})
        questions = get_followup_questions(LANGUAGE)  # Get follow-up questions based on language
        return jsonify({'language': LANGUAGE, 'questions': questions})  # Send language and questions to frontend

if __name__ == '__main__':
    app.run(debug=True)
