class PromptsTemplates:
    """
    Class for managing various prompt templates and response evaluations in an HSE context.
    """

    def __init__(self, response_evaluator, report_generator):
        self.response_evaluator = response_evaluator
        self.report_generator = report_generator
        self.current_question = None
        self.current_response = None

    def set_current_question(self, question):
        """
        Sets the current question for context in follow-ups and evaluations.
        """
        self.current_question = question

    def set_current_response(self, response):
        """
        Sets the current question for context in follow-ups and evaluations.
        """
        self.current_response = response

    def reask(self):
        """
        Generates a prompt to reask the current question for more clarity.
        """
        return f"Could you please elaborate on your last response about '{self.current_question}'?"

    def move_to_next_question(self, next_question):
        """
        Provides a transition to the next question.
        """
        return f"Thank you for the information. Now, let's talk about {next_question}."

    def follow_up_for_details(self):
        """
        Generates a follow-up prompt for more detailed information.
        """
        return f"Thanks for sharing that. Can you provide more specific details about '{self.current_question}'?"

    def check_response_sufficiency(self, response):
        """
        Checks if the response is sufficient using the response evaluator.
        """
        return self.response_evaluator.is_response_sufficient(self.current_question, response)

    def generate_report(self, responses):
        """
        Generates a report based on the collected responses.
        """
        return self.report_generator.generate(responses)




class ResponseEvaluator:
    """
    Class to evaluate the sufficiency of responses using an AI model.
    """

    def __init__(self, ai_model):
        self.ai_model = ai_model

    def formulate_model_prompt(self, question, operator_response):
        """
        Formulates a prompt for the AI model to evaluate the response.
        """
        prompt = (
            f"Question: {question}\n"
            f"Operator's Response: {operator_response}\n"
            "Is this response sufficient for a detailed understanding? (Yes/No)\n"
        )
        return prompt

    def is_response_sufficient(self, question, operator_response):
        """
        Determines if the operator's response is sufficient.
        """
        prompt = self.formulate_model_prompt(question, operator_response)
        model_response = self.ai_model.generate_response(prompt)  # Assuming a method to get model's response

        # Interpreting the model's response
        return model_response.strip().lower() == "yes"
    



class QuestionsManager:
    """
    Manages the flow and state of questions in the interaction with operators.
    """

    def __init__(self, questions, prompts_templates, response_evaluator):
        self.questions = questions
        self.current_question_index = 0
        self.prompts_templates = prompts_templates
        self.response_evaluator = response_evaluator

    def get_current_question(self):
        """
        Returns the current question to be asked.
        """
        return self.questions[self.current_question_index]

    def update_question_flow(self, operator_response):
        """
        Determines the next step based on the operator's response.
        Returns a tuple (next_prompt, is_end_of_questions).
        """
        current_question = self.get_current_question()
        if self.response_evaluator.is_response_sufficient(current_question, operator_response):
            # Move to the next question
            self.current_question_index += 1
            if self.current_question_index < len(self.questions):
                next_question = self.get_current_question()
                return self.prompts_templates.move_to_next_question(next_question), False
            else:
                return "Thank you for your responses. That's all the questions we have.", True
        else:
            # Reask the current question
            return self.prompts_templates.reask(), False

# Example Usage
questions = [
    "Can you describe the safety measures in place in your area?",
    "How effective do you find the recent safety protocol changes?"
    # Add more questions as needed
]

# Assuming instances of PromptsTemplates and ResponseEvaluator are already created
questions_manager = QuestionsManager(questions, prompts_templates, response_evaluator)

# Simulating interaction
current_question = questions_manager.get_current_question()
print("Agent: ", current_question)  # Agent asks the first question

# Assuming an operator response
operator_response = "We have alarms."

# Determine the next step
next_prompt, is_end = questions_manager.update_question_flow(operator_response)
print("Agent: ", next_prompt)  # Agent either asks the next question or reasks based on the response







