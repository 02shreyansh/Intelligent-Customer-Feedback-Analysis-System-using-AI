# Bonus: AI Chatbot for Customer Insights
from transformers import pipeline

class FeedbackChatbot:
    def __init__(self):
        self.qa_pipeline = pipeline("question-answering", 
                                  model="distilbert-base-cased-distilled-squad")
        self.feedback_context = self.prepare_context()
    
    def prepare_context(self):
        """Prepare context from analyzed feedback"""
        positive_feedback = processed_df[processed_df['sentiment'] == 'positive']['feedback'].tolist()
        negative_feedback = processed_df[processed_df['sentiment'] == 'negative']['feedback'].tolist()
        
        context = """
        Customer Feedback Analysis Context:
        
        Positive Feedback Themes:
        - Product quality and satisfaction
        - Good customer service experience  
        - Value for money
        - User-friendly features
        
        Negative Feedback Themes:
        - Product quality issues
        - Poor customer service
        - Shipping and delivery problems
        - Mismatched expectations
        
        Common Suggestions:
        - Improve product durability
        - Enhance customer support responsiveness
        - Provide better product information
        - Streamline return processes
        """
        
        # Add sample feedbacks
        context += "\nSample Positive Feedbacks:\n" + "\n".join(positive_feedback[:3])
        context += "\nSample Negative Feedbacks:\n" + "\n".join(negative_feedback[:3])
        
        return context
    
    def ask_question(self, question):
        """Answer questions based on feedback analysis"""
        try:
            result = self.qa_pipeline(
                question=question,
                context=self.feedback_context,
                max_answer_len=100
            )
            return result['answer']
        except:
            return "I'm sorry, I couldn't process that question. Please try rephrasing."

# Test the chatbot
print("🤖 Testing AI Chatbot for Feedback Insights...")
chatbot = FeedbackChatbot()

test_questions = [
    "What are the main positive themes in customer feedback?",
    "What issues do customers complain about most?",
    "What suggestions do customers have for improvement?"
]

for question in test_questions:
    answer = chatbot.ask_question(question)
    print(f"\n❓ Question: {question}")
    print(f"🤖 Answer: {answer}")
    print("-" * 80)