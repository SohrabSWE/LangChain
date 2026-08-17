student_answer = """
   Python is a high-level programming language.
   It is widely used in web development, data science,
   and artificial intelligence.
"""

#step 1 : clean the answer 
def clean_answer(answer):
    return answer.strip()


cleaned_answer = clean_answer(student_answer)

#step 2 : count the words 

def count_words(answer): 
    return len(answer.split())

word_count = count_words(cleaned_answer)



#step 3 : generate feedback 

def generate_feedback(answer,word_count): 
    return f"""
    The answer contains {word_count} words.
    The student correctly explained python and mentioned some important applicatin areas.

      """

feedback = generate_feedback(cleaned_answer,word_count)


#step 4 : format the feedback 

def format_feedback(feedback): 
    return feedback.strip().upper()