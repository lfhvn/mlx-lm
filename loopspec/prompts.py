QUESTIONS = [
    "Write a Python function that returns the longest palindromic substring of s. Explain the complexity.",
    "Implement an LRU cache class in Python with get and put in O(1). Include a short test.",
    "Given an array of up to 1e5 integers, count inversions. Write the C++ solution.",
    "Write a bash script that finds the 10 largest files under a directory, skipping .git.",
    "Fix this bug: `def mean(xs): return sum(xs) / len(xs)` crashes on empty lists. Write a robust version with type hints and docstring.",
]


def chat_prompts(tok, questions=QUESTIONS):
    return [
        tok.apply_chat_template(
            [{"role": "user", "content": q}], add_generation_prompt=True
        )
        for q in questions
    ]


def long_prompt(tok, n=4000):
    filler = " ".join(
        f"step {i} checked the edge case and found it fine." for i in range(1500)
    )
    head = tok.encode(filler)[:n]
    tail = chat_prompts(tok, [QUESTIONS[1]])[0]
    return head + tail
