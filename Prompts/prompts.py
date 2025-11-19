MISTAKE_CORRECTION_PROMPT = (
    lambda question, answer: f"""
    You are an expert math tutor who evaluates and corrects student solutions.

    Create a **multi-turn** student-teacher dialogue where:
    - The student presents an incorrect or partially correct solution.
    - The teacher first **evaluates the student's solution for correctness**.
    - If the student's response is incorrect, the teacher **identifies the exact step or reasoning error** where the first mistake occurs.
    - The teacher then **explains why that step is wrong** and provides the **correct reasoning chain** that leads to the correct numeric answer.

    The teacher should give clear, factual feedback—no vague hints or open-ended guidance.

    Question: {question}
    Ground truth answer: {answer}

    Return your response strictly in valid JSON with the following structure:
    [
      {{"student": "value"}},
      {{"teacher": "value"}},
      {{"student": "value"}},
      ...
    ]
"""
)


SCAFFOLDING_PROMPT = (
    lambda question, answer: f"""
    You are an expert math tutor who excels at scaffolding — guiding students to reason deeply,
    identify mistakes, and build understanding without directly giving away answers.

    Your goal is to create a **multi-turn** student-teacher dialogue that demonstrates excellent scaffolding.
    Each teacher response should build upon the student's previous message — encouraging reflection,
    probing for reasoning, and guiding them toward understanding. The teacher should never directly state
    the correct answer but should progressively help the student reason it out.
    

    Questions: {question}
    Ground truth answer: {answer}
    Return your response strictly in valid JSON with the following structure:
    [{{ "student": "value"}}, {{"teacher": "value"}}, {{"student": "value"}}, ...]
"""
)
SOCRATIC_QUESTIONING_PROMPT = (
    lambda question, answer: f"""
You are an expert math tutor who uses the **Socratic questioning** method to guide students toward understanding. 
Your goal is not to provide the answer, but to help the student reason it out by asking thoughtful, probing, and guiding questions.

Given a math problem, the student's answer, and an image, create a multi-turn student-teacher dialogue that demonstrates excellent Socratic questioning. 
Each teacher response should build on the student's previous statement and aim to:
- Decompose the problem into smaller, manageable reasoning steps.
- Ask guiding questions that prompt reflection or verification.
- Encourage the student to explain their thinking.
- Maintain a supportive and curious tone.

Question: {question}
Ground truth answer: {answer}

Return your response strictly in **valid JSON** with the structure:
[{{"student": "value", "teacher": "value"}}, {{"student": "value", "teacher": "value"}},... ]
"""
)
